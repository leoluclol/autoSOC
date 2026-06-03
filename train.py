import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler

# Ensures Kaggle can read .xlsx files without crashing
os.system('pip install openpyxl -q')

FEATURE_COLUMNS = [
    'Voltage (V)',
    'Current (A)',
    'Ah_roll600',
    'dV_dt',
    'T',
    'Current_dir',
    'Current_roll120',
    'time_since_dir_change'
]

# ==========================================
# 1. DATA PREPARATION
# ==========================================
def create_multibranch_sequences(data, target, fast_seq_length=100, slow_seq_length=150, slow_step=5):
    xs_fast, xs_slow, ys = [], [], []
    max_lookback = slow_seq_length * slow_step

    for i in range(max_lookback - 1, len(data)):
        x_f = data[i - fast_seq_length + 1 : i + 1]
        x_s = data[i - max_lookback + 1 : i + 1 : slow_step]
        xs_fast.append(x_f)
        xs_slow.append(x_s)
        ys.append(target[i])

    return np.array(xs_fast), np.array(xs_slow), np.array(ys)

def process_and_split_data(filename='/kaggle/input/datasets/leonardoluchini/calce-a123-dynamic-raw-joined/dataset_filled_1Hz_LFP.xlsx'):
    print(f"Loading data from {filename}...")
    df = pd.read_excel(filename, sheet_name='Temp_25C')
    df = df.drop_duplicates(subset=['Time (s)'], keep='first')

    df['dt'] = df['Time (s)'].diff().fillna(0)

    if 'Temperature (C)' in df.columns and 'T' not in df.columns:
        df = df.rename(columns={'Temperature (C)': 'T'})

    df['Ah_step'] = (df['Current (A)'] * df['dt']) / 3600.0
    df['Ah_roll600'] = df['Ah_step'].rolling(window=600, min_periods=1).sum()
    df['dV_dt'] = df['Voltage (V)'].diff().fillna(0)
    df['Current_dir'] = np.sign(df['Current (A)']).fillna(0)
    df['Current_roll120'] = df['Current (A)'].rolling(window=120, min_periods=1).sum()

    dir_vals = df['Current_dir'].to_numpy()
    time_since = np.zeros(len(dir_vals), dtype=int)
    last_sign = dir_vals[0] if dir_vals[0] != 0 else 0
    counter = 0
    time_since[0] = 0
    for i in range(1, len(dir_vals)):
        curr = dir_vals[i]
        if curr != 0:
            if last_sign == 0 or curr == last_sign:
                counter += 1
            else:
                counter = 0
            last_sign = curr
        else:
            counter += 1
        time_since[i] = counter
    df['time_since_dir_change'] = time_since

    target_col = 'SOC'
    df = df.dropna(subset=FEATURE_COLUMNS + [target_col])

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(df[FEATURE_COLUMNS].values)
    y_values = df[target_col].values

    Xt_f, Xt_s, Yt = create_multibranch_sequences(X_scaled, y_values, fast_seq_length=100, slow_seq_length=150, slow_step=5)

    split_idx = int(len(Xt_f) * 0.8)
    X_tr_f, X_tr_s, y_tr = Xt_f[:split_idx], Xt_s[:split_idx], Yt[:split_idx]
    X_te_f, X_te_s, y_te = Xt_f[split_idx:], Xt_s[split_idx:], Yt[split_idx:]

    return X_tr_f, X_tr_s, y_tr, X_te_f, X_te_s, y_te, scaler

# ==========================================
# 2. MODEL AND LOSS DEFINITION
# ==========================================
class PhysicsInformedBMSLoss(nn.Module):
    def __init__(self, lambda_penalty=0.0, current_zero_val=0.0, current_threshold=0.05):
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
        return base_loss + (self.lambda_penalty * physics_penalty)

class HysteresisAwareFusionNet(nn.Module):
    def __init__(self, input_size, fast_channels=32, fast_hidden=48, slow_hidden=48, dropout=0.25):
        super(HysteresisAwareFusionNet, self).__init__()
        self.fast_conv = nn.Conv1d(in_channels=input_size, out_channels=fast_channels, kernel_size=3, padding=1)
        self.fast_dropout = nn.Dropout(dropout)
        self.fast_gru = nn.GRU(fast_channels, fast_hidden, batch_first=True)

        self.slow_linear = nn.Linear(input_size, slow_hidden)
        self.slow_dropout = nn.Dropout(dropout)
        self.slow_gru = nn.GRU(slow_hidden, slow_hidden, batch_first=True)
        self.slow_norm = nn.LayerNorm(slow_hidden)

        self.dir_gate = nn.Sequential(
            nn.Linear(3, slow_hidden),
            nn.SiLU(),
            nn.Linear(slow_hidden, slow_hidden),
            nn.Sigmoid()
        )

        fusion_input = fast_hidden + slow_hidden
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input, 40),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(40, 18),
            nn.SiLU()
        )
        self.head = nn.Linear(18, 1)

    def _prepare_direction_gates(self, x_fast, idx_dir, idx_time):
        dir_t = x_fast[:, -1, idx_dir]
        dir_t_1 = x_fast[:, -2, idx_dir]
        time_since_t = x_fast[:, -1, idx_time] / 200.0
        time_since_t_1 = x_fast[:, -2, idx_time] / 200.0

        hist_t = x_fast[:, -25:-5, idx_dir]
        hist_t_1 = x_fast[:, -26:-6, idx_dir]

        dir_hist_t = hist_t.mean(dim=1)
        dir_hist_t_1 = hist_t_1.mean(dim=1)

        gate_t = self.dir_gate(torch.stack([dir_t, dir_hist_t, time_since_t], dim=1))
        gate_t_1 = self.dir_gate(torch.stack([dir_t_1, dir_hist_t_1, time_since_t_1], dim=1))
        return gate_t, gate_t_1

    def forward(self, x_fast, x_slow):
        xf = torch.relu(self.fast_conv(x_fast.permute(0, 2, 1)))
        xf = xf.permute(0, 2, 1)
        xf = self.fast_dropout(xf)
        fast_out, _ = self.fast_gru(xf)
        fast_feat_t = torch.relu(fast_out[:, -1, :])
        fast_feat_t_1 = torch.relu(fast_out[:, -2, :])

        slow_emb = torch.relu(self.slow_linear(x_slow))
        slow_emb = self.slow_dropout(slow_emb)
        slow_out, _ = self.slow_gru(slow_emb)
        slow_t = self.slow_norm(slow_out[:, -1, :])
        slow_t_1 = self.slow_norm(slow_out[:, -2, :])

        idx_dir = FEATURE_COLUMNS.index('Current_dir')
        idx_time = FEATURE_COLUMNS.index('time_since_dir_change')
        gate_t, gate_t_1 = self._prepare_direction_gates(x_fast, idx_dir, idx_time)

        gated_slow_t = slow_t * gate_t
        gated_slow_t_1 = slow_t_1 * gate_t_1

        fused_t = self.fusion(torch.cat([fast_feat_t, gated_slow_t], dim=1))
        fused_t_1 = self.fusion(torch.cat([fast_feat_t_1, gated_slow_t_1], dim=1))

        pred_t = self.head(fused_t)
        pred_t_1 = self.head(fused_t_1)

        return pred_t, pred_t_1

# ==========================================
# 3. TRAINING LOOP WITH METRICS
# ==========================================
def train_and_evaluate():
    t_start = time.time()

    X_tr_f, X_tr_s, y_tr, X_te_f, X_te_s, y_te, scaler = process_and_split_data()

    train_loader = DataLoader(TensorDataset(torch.tensor(X_tr_f, dtype=torch.float32),
                                            torch.tensor(X_tr_s, dtype=torch.float32),
                                            torch.tensor(y_tr, dtype=torch.float32).view(-1, 1)),
                              batch_size=96, shuffle=True)

    test_loader = DataLoader(TensorDataset(torch.tensor(X_te_f, dtype=torch.float32),
                                           torch.tensor(X_te_s, dtype=torch.float32),
                                           torch.tensor(y_te, dtype=torch.float32).view(-1, 1)),
                             batch_size=96, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HysteresisAwareFusionNet(input_size=len(FEATURE_COLUMNS)).to(device)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    optimizer = optim.Adam(model.parameters(), lr=3e-4, weight_decay=5e-6)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.75, patience=5)

    dummy_array = np.zeros((1, len(FEATURE_COLUMNS)))
    scaled_zero_amp = scaler.transform(dummy_array)[0, FEATURE_COLUMNS.index('Current (A)')]
    dummy_array[0, FEATURE_COLUMNS.index('Current (A)')] = 0.5
    scaled_half_amp = abs(scaler.transform(dummy_array)[0, FEATURE_COLUMNS.index('Current (A)')] - scaled_zero_amp)

    criterion = PhysicsInformedBMSLoss(lambda_penalty=0.1, current_zero_val=scaled_zero_amp, current_threshold=scaled_half_amp)

    epoch_num = 50
    total_steps = 0
    t_start_training = time.time()

    print(f"\nStarting Training on {device} - Model Parameters: {num_params / 1e6:.4f}M")

    for epoch in range(epoch_num):
        model.train()
        train_loss = 0.0

        for batch_xf, batch_xs, batch_y in train_loader:
            batch_xf, batch_xs, batch_y = batch_xf.to(device), batch_xs.to(device), batch_y.to(device)
            optimizer.zero_grad()

            pred_t, pred_t_1 = model(batch_xf, batch_xs)
            current_t_scaled = batch_xf[:, -1, FEATURE_COLUMNS.index('Current (A)')]

            loss = criterion(pred_t, pred_t_1, batch_y, current_t_scaled)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()
            total_steps += 1

        model.eval()
        epoch_test_mae = 0.0
        with torch.no_grad():
            for batch_xf_te, batch_xs_te, batch_y_te in test_loader:
                batch_xf_te, batch_xs_te, batch_y_te = batch_xf_te.to(device), batch_xs_te.to(device), batch_y_te.to(device)
                pred_t_eval, _ = model(batch_xf_te, batch_xs_te)
                epoch_test_mae += nn.L1Loss()(pred_t_eval, batch_y_te).item()

        epoch_test_mae /= len(test_loader)
        scheduler.step(epoch_test_mae)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:03d}/{epoch_num} | Train Loss: {train_loss/len(train_loader):.5f} | Test MAE: {epoch_test_mae:.5f}")

    training_time = time.time() - t_start_training

    model.eval()
    all_y_pred, all_y_true = [], []
    with torch.no_grad():
        for batch_xf_te, batch_xs_te, batch_y_te in test_loader:
            batch_xf_te, batch_xs_te = batch_xf_te.to(device), batch_xs_te.to(device)
            pred_t_final, _ = model(batch_xf_te, batch_xs_te)
            all_y_pred.append(pred_t_final.cpu().numpy())
            all_y_true.append(batch_y_te.numpy())

    y_pred = np.vstack(all_y_pred).flatten()
    y_true = np.vstack(all_y_true).flatten()

    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred)**2))
    max_err = np.max(np.abs(y_true - y_pred))

    if torch.cuda.is_available():
        peak_vram_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
    else:
        peak_vram_mb = 0.0

    total_time = time.time() - t_start

    print("\n" + "="*40)
    print("FINAL EVALUATION METRICS")
    print("="*40)
    print(f"test_mae_percent:   {mae * 100:.4f}")
    print(f"test_rmse_percent:  {rmse * 100:.4f}")
    print(f"max_error_percent:  {max_err * 100:.4f}")
    print("-" * 40)
    print(f"training_seconds:   {training_time:.1f}")
    print(f"total_seconds:      {total_time:.1f}")
    print(f"peak_vram_mb:       {peak_vram_mb:.1f}")
    print(f"num_steps:          {total_steps}")
    print(f"total_samples:      {len(y_tr) + len(y_te)}")
    print(f"num_params_M:       {num_params / 1e6:.4f}")
    print("="*40)

if __name__ == "__main__":
    try:
        train_and_evaluate()
    except FileNotFoundError:
        print("Error: Dataset not found. Verify that you are on Kaggle and check the path.")