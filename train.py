import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# ==============================================================================
# 1. GLOBAL HYPERPARAMETERS & CONFIGURATION
# ==============================================================================

# --- Data Parameters ---
TRAIN_RATIO = 0.8
NUM_WINDOWS = 3                # Number of distinct test chunks to carve out
SEQ_LENGTH = 300               # Number of historical timesteps per sequence
BATCH_SIZE = 128               # Batch size for DataLoaders


# --- Model Architecture ---
LSTM_HIDDEN = 64               # Hidden dimensions for the LSTM
NUM_HEADS = 4                  # Self-Attention heads (LSTM_HIDDEN must be divisible by this)
DROPOUT = 0.3                  # Dropout probability for regularization
MLP_HIDDEN = 32                # Hidden dimension for the dense layer post-attention

# --- Training Parameters ---
EPOCHS = 150                   # Total training epochs
LEARNING_RATE = 0.005          # Initial learning rate for Adam
WEIGHT_DECAY = 1e-5            # L2 regularization factor
SCHEDULER_PATIENCE = 5         # Epochs to wait before reducing LR on plateau
SCHEDULER_FACTOR = 0.75        # Factor to multiply LR by on plateau

# --- PINN (Physics-Informed) Configuration ---
LAMBDA_PENALTY = 0.3           # Weight of the physics penalty against the base MAE loss
CURRENT_IDLE_THRESHOLD_AMP = 0.5 # Amps threshold below which battery is considered "idle"

# --- Feature Configuration ---
COLUMNS = ['Test_Time(s)', 'Voltage(V)', 'Current(A)', 'dV/dt(V/s)', 'Temperature (C)_1', 'SoC (%)']
FEATURES = ['Voltage(V)', 'Current(A)', 'dV/dt(V/s)', 'Temperature (C)_1']
TARGET = 'SoC (%)'
CURRENT_IDX = FEATURES.index('Current(A)') # Automatically find the index for Current(A)


# ========= DO NOT MODIFY
FILE_PATH = '/kaggle/input/datasets/leonardoluchini/calce-a123-dynamic-raw-joined/CALCE-dataset-1Hz.xlsx'
SHEETS = ["DST-US06-FUDS-25"]
OCV_KEEP_FRACTION = 0.5        # Downsampling factor for pure OCV profiles


# ==============================================================================
# 2. DATA PROCESSING & SEQUENCE GENERATION
# ==============================================================================

def create_sequences(data, target, seq_length=300):
    xs, ys = [], []
    for i in range(seq_length - 1, len(data)): 
        xs.append(data[i - seq_length + 1 : i + 1])
        ys.append(target[i])
    return np.array(xs), np.array(ys)

def process_and_split_data(filename, sheets, train_ratio, seq_length, num_windows):
    X_tr_list, y_tr_list, X_te_list, y_te_list = [], [], [], []
    train_raw_features, sheet_data_dict = [], {}

    # --- PASS 1: Calculate Windows & Fit Scaler ---
    for sheet in sheets:
        df = pd.read_excel(filename, sheet_name=sheet, usecols=COLUMNS)
        X_raw = df[FEATURES].values
        y_raw = df[TARGET].values
        total_len = len(X_raw)
        
        test_len = int(total_len * (1 - train_ratio))
        window_size = test_len // num_windows
        interval = total_len // num_windows
        
        test_windows = []
        for i in range(num_windows):
            w_start = i * interval + (interval - window_size) // 2
            w_end = w_start + window_size
            test_windows.append((w_start, w_end))
            
        train_mask = np.ones(total_len, dtype=bool) 
        for start, end in test_windows:
            train_mask[start:end] = False
            
        train_raw_features.append(X_raw[train_mask])
        
        sheet_data_dict[sheet] = {
            'X_raw': X_raw, 'y_raw': y_raw, 
            'test_windows': test_windows, 'total_len': total_len
        }

    scaler = MinMaxScaler()
    scaler.fit(np.vstack(train_raw_features))

    # --- PASS 2: Scale & Chunk Generation ---
    for sheet in sheets:
        data = sheet_data_dict[sheet]
        X_scaled = scaler.transform(data['X_raw'])
        step_size = int(1 / OCV_KEEP_FRACTION) if "OCV" in sheet else 1
        
        # Test Sequences (With lookback)
        for start, end in data['test_windows']:
            test_start_idx = max(0, start - seq_length + 1)
            X_test_chunk = X_scaled[test_start_idx:end]
            y_test_chunk = data['y_raw'][test_start_idx:end]
            
            if len(X_test_chunk) >= seq_length:
                xt, yt = create_sequences(X_test_chunk, y_test_chunk, seq_length)
                X_te_list.append(xt[::step_size])
                y_te_list.append(yt[::step_size])
                
        # Train Sequences (Between test chunks)
        train_chunks = []
        last_end = 0
        for start, end in data['test_windows']:
            if start > last_end:
                train_chunks.append((last_end, start))
            last_end = end
        if last_end < data['total_len']:
            train_chunks.append((last_end, data['total_len']))
            
        for t_start, t_end in train_chunks:
            X_train_chunk = X_scaled[t_start:t_end]
            y_train_chunk = data['y_raw'][t_start:t_end]
            
            if len(X_train_chunk) >= seq_length:
                xt, yt = create_sequences(X_train_chunk, y_train_chunk, seq_length)
                X_tr_list.append(xt[::step_size])
                y_tr_list.append(yt[::step_size])

    X_tr = np.concatenate(X_tr_list, axis=0) if X_tr_list else None
    y_tr = np.concatenate(y_tr_list, axis=0) if y_tr_list else None
    X_te = np.concatenate(X_te_list, axis=0) if X_te_list else None
    y_te = np.concatenate(y_te_list, axis=0) if y_te_list else None
    
    return X_tr, y_tr, X_te, y_te, scaler, sheet_data_dict


# ==============================================================================
# 3. MODEL ARCHITECTURE & PINN LOSS
# ==============================================================================

class PhysicsInformedBMSLoss(nn.Module):
    def __init__(self, lambda_penalty, current_zero_val, current_threshold):
        super(PhysicsInformedBMSLoss, self).__init__()
        self.mae = nn.L1Loss()
        self.lambda_penalty = lambda_penalty
        self.zero_val = current_zero_val
        self.threshold = current_threshold

    def forward(self, pred_t, pred_t_1, y_true, current_t_scaled):
        base_loss = self.mae(pred_t, y_true)
        delta_pred = torch.abs(pred_t - pred_t_1)
        is_idle = (torch.abs(current_t_scaled - self.zero_val) < self.threshold).float().unsqueeze(1)
        physics_penalty = torch.mean(delta_pred * is_idle)
        
        total_loss = base_loss + (self.lambda_penalty * physics_penalty)
        return total_loss, base_loss, physics_penalty


class BatteryAttnLSTMNet(nn.Module):
    def __init__(self, input_size, lstm_hidden, num_heads, dropout, mlp_hidden):
        super(BatteryAttnLSTMNet, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=lstm_hidden, 
                            num_layers=2, batch_first=True, dropout=dropout)
        
        self.attention = nn.MultiheadAttention(embed_dim=lstm_hidden, num_heads=num_heads, 
                                               batch_first=True, dropout=dropout)
        
        self.mlp = nn.Sequential(
            nn.Linear(lstm_hidden, mlp_hidden),
            nn.ReLU(),
            nn.Dropout(p=dropout)
        )
        self.fc_out = nn.Linear(mlp_hidden, 1)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out) 
        
        feat_t   = attn_out[:, -1, :]
        feat_t_1 = attn_out[:, -2, :]
        
        pred_t = self.fc_out(self.mlp(feat_t))
        pred_t_1 = self.fc_out(self.mlp(feat_t_1))
        
        return pred_t, pred_t_1


# ==============================================================================
# 4. TRAINING LOOP & EVALUATION
# ==============================================================================

def evaluate_metrics(model, dataloader, device):
    model.eval()
    mae_loss_fn = nn.L1Loss(reduction='sum')
    total_mae, num_samples = 0.0, 0
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            pred, _ = model(x)
            total_mae += mae_loss_fn(pred, y).item()
            num_samples += y.size(0)
    return total_mae / num_samples


def train_model(train_loader, test_loader, scaler):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[INIT] Device: {device.type.upper()} | Features: {scaler.n_features_in_}", flush=True)
    
    # Init Model
    model = BatteryAttnLSTMNet(
        input_size=scaler.n_features_in_, 
        lstm_hidden=LSTM_HIDDEN, 
        num_heads=NUM_HEADS, 
        dropout=DROPOUT,
        mlp_hidden=MLP_HIDDEN
    )

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model = model.to(device)
    
    # Optimizer & Scheduler
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=SCHEDULER_FACTOR, patience=SCHEDULER_PATIENCE
    )

    # Scaling values for PINN bounds
    dummy_array = np.zeros((1, scaler.n_features_in_))
    scaled_zero_amp = scaler.transform(dummy_array)[0, CURRENT_IDX]
    dummy_array[0, CURRENT_IDX] = CURRENT_IDLE_THRESHOLD_AMP 
    scaled_threshold_amp = abs(scaler.transform(dummy_array)[0, CURRENT_IDX] - scaled_zero_amp)

    criterion = PhysicsInformedBMSLoss(
        lambda_penalty=LAMBDA_PENALTY, 
        current_zero_val=scaled_zero_amp, 
        current_threshold=scaled_threshold_amp
    )
    
    fast_test_loader = DataLoader(test_loader.dataset, batch_size=256, shuffle=False)
    
    print("[TRAIN] Starting training loop...", flush=True)
    for epoch in range(EPOCHS):
        model.train()
        running_train_loss = 0.0
        
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            
            pred_t, pred_t_1 = model(batch_x) 
            current_t_scaled = batch_x[:, -1, CURRENT_IDX]
            
            loss, _, _ = criterion(pred_t, pred_t_1, batch_y, current_t_scaled)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running_train_loss += loss.item()
            
        test_mae = evaluate_metrics(model, fast_test_loader, device)
        scheduler.step(test_mae)
        
        if (epoch+1) % 5 == 0:
            epoch_loss = running_train_loss / len(train_loader)
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch+1:03d}/{EPOCHS} | Train Loss: {epoch_loss:.5f} | Test MAE: {test_mae:.5f}% | LR: {current_lr:.2e}", flush=True)

    print("\n[EVAL] Extracting final predictions...", flush=True)
    model.eval()
    all_y_pred, all_y_true = [], []
    with torch.no_grad():
        for batch_x_te, batch_y_te in fast_test_loader:
            batch_x_te = batch_x_te.to(device)
            pred_t_final, _ = model(batch_x_te)
            all_y_pred.append(pred_t_final.cpu().numpy())
            all_y_true.append(batch_y_te.cpu().numpy()) 
            
    return np.vstack(all_y_true), np.vstack(all_y_pred)


# ==============================================================================
# 5. VISUALIZATION MODULE
# ==============================================================================

def get_window_sizes(test_metadata, seq_length):
    sizes = []
    for sheet, data in test_metadata.items():
        step_size = int(1 / OCV_KEEP_FRACTION) if "OCV" in sheet else 1
        for start, end in data['test_windows']:
            test_start_idx = max(0, start - seq_length + 1)
            chunk_len = end - test_start_idx
            if chunk_len >= seq_length:
                num_seqs = chunk_len - seq_length + 1
                actual_len = len(range(0, num_seqs, step_size))
                sizes.append(actual_len)
    return sizes

def plot_analysis(y_true, y_pred, window_sizes):
    y_true_flat = np.array(y_true).flatten()
    y_pred_flat = np.array(y_pred).flatten()
    abs_error_raw = np.abs(y_true_flat - y_pred_flat)
    
    num_windows = len(window_sizes)
    num_subplots = num_windows + 1
    
    height_ratios = [2] * num_windows + [1.5]
    fig, axes = plt.subplots(num_subplots, 1, figsize=(16, 4 * num_subplots), gridspec_kw={'height_ratios': height_ratios})
    if num_subplots == 1: axes = [axes]
    
    current_idx = 0
    colors = ['blue', 'green', 'orange', 'purple', 'brown'] 
    
    for i, size in enumerate(window_sizes):
        y_t_win = y_true_flat[current_idx : current_idx + size]
        y_p_win = y_pred_flat[current_idx : current_idx + size]
        mae_win = np.mean(np.abs(y_t_win - y_p_win))
        
        ax = axes[i]
        color = colors[i % len(colors)]
        
        ax.plot(y_t_win, label='True SoC (%)', color=color, linewidth=2)
        ax.plot(y_p_win, label='Predicted SoC (%)', color='red', linestyle='--', alpha=0.9)
        ax.set_title(f'Test Window {i+1} | Window MAE: {mae_win:.3f}%', fontsize=14)
        ax.legend(); ax.grid(True, linestyle=':', alpha=0.6)
        
        current_idx += size
        
    err_ax = axes[num_windows]
    err_ax.plot(abs_error_raw, color='black', alpha=0.6)
    
    curr_bound = 0
    for i, size in enumerate(window_sizes[:-1]):
        curr_bound += size
        label = 'Window Split' if i == 0 else ""
        err_ax.axvline(x=curr_bound, color='red', linestyle='-', linewidth=2, alpha=0.7, label=label)
        
    err_ax.set_title('Global Absolute Error (Concatenated Test Set)', fontsize=14)
    if num_windows > 1: err_ax.legend()
    err_ax.grid(True, linestyle=':', alpha=0.6)

    plt.tight_layout()
    plt.show()

# ==============================================================================
# 6. MAIN EXECUTION
# ==============================================================================

if __name__ == "__main__":
    # 1. Process Data
    X_tr, y_tr, X_te, y_te, scaler, test_metadata = process_and_split_data(
        filename=FILE_PATH, sheets=SHEETS, train_ratio=TRAIN_RATIO, 
        seq_length=SEQ_LENGTH, num_windows=NUM_WINDOWS
    )

    if X_tr is not None:
        # 2. Build Loaders
        train_loader = DataLoader(
            TensorDataset(torch.tensor(X_tr, dtype=torch.float32), 
                          torch.tensor(y_tr, dtype=torch.float32).view(-1, 1)), 
            batch_size=BATCH_SIZE, shuffle=True
        )
                                  
        test_loader = DataLoader(
            TensorDataset(torch.tensor(X_te, dtype=torch.float32), 
                          torch.tensor(y_te, dtype=torch.float32).view(-1, 1)), 
            batch_size=BATCH_SIZE, shuffle=False
        )
                                 
        print(f"Data ready. Train samples: {len(X_tr)} | Test samples: {len(X_te)}")

        # 3. Train & Extract Predictions
        y_true, y_pred = train_model(train_loader, test_loader, scaler)

        # 4. Plot Results
        win_sizes = get_window_sizes(test_metadata, seq_length=SEQ_LENGTH)
        plot_analysis(y_true, y_pred, window_sizes=win_sizes)