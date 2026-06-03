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
    
    features_cols = ['Voltage (V)', 'Current (A)', 'Ah_roll600', 'dV_dt', 'T']
    target_col = 'SOC'
    
    df = df.dropna(subset=features_cols + [target_col])
    
    scaler = MinMaxScaler()
    X_data = df[features_cols].values
    X_scaled = scaler.fit_transform(X_data)
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

class BatteryAttentionFusionNet(nn.Module):
    def __init__(self, input_size, fast_channels=32, fast_hidden=64, slow_hidden=64, attn_heads=4, dropout=0.25):
        super(BatteryAttentionFusionNet, self).__init__()
        self.fast_conv = nn.Conv1d(in_channels=input_size, out_channels=fast_channels, kernel_size=3, padding=1)
        self.fast_pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.fast_norm = nn.LayerNorm(fast_channels)
        self.fast_lstm = nn.LSTM(input_size=fast_channels, hidden_size=fast_hidden, batch_first=True, num_layers=1)
        self.slow_gru = nn.GRU(input_size=input_size, hidden_size=slow_hidden, batch_first=True, num_layers=1)

        self.fast2attn = nn.Linear(fast_hidden, slow_hidden)

        self.attn = nn.MultiheadAttention(embed_dim=slow_hidden, num_heads=attn_heads, batch_first=True, dropout=dropout)

        self.context_gate = nn.Sequential(
            nn.Linear(4, slow_hidden),
            nn.SiLU(),
            nn.Linear(slow_hidden, slow_hidden),
            nn.Sigmoid()
        )

        fused_dim = fast_hidden + slow_hidden
        self.fusion = nn.Sequential(
            nn.LayerNorm(fused_dim),
            nn.Linear(fused_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        self.head = nn.Linear(32, 1)

    def _current_stats(self, segment):
        mean = segment.mean(dim=1)
        std = segment.std(dim=1, unbiased=False)
        max_val = segment.max(dim=1).values
        min_val = segment.min(dim=1).values
        return torch.stack([mean, std, max_val, min_val], dim=1)

    def _apply_attention(self, fast_feat, slow_seq, stats):
        query = self.fast2attn(fast_feat).unsqueeze(1)
        attn_out, _ = self.attn(query, slow_seq, slow_seq)
        attn_vec = attn_out.squeeze(1)
        gate = self.context_gate(stats)
        return attn_vec * gate

    def forward(self, x_fast, x_slow):
        xf = x_fast.permute(0, 2, 1)
        fast_conv = torch.relu(self.fast_conv(xf))
        fast_conv = self.fast_pool(fast_conv)
        fast_conv = fast_conv.permute(0, 2, 1)
        fast_conv = self.fast_norm(fast_conv)
        fast_out, _ = self.fast_lstm(fast_conv)

        slow_out, _ = self.slow_gru(x_slow)

        feat_fast_t = fast_out[:, -1, :]
        feat_fast_t_1 = fast_out[:, -2, :]

        feat_slow_t = slow_out[:, -1, :]
        feat_slow_t_1 = slow_out[:, -2, :]

        curr_t = x_fast[:, -5:, 1]
        curr_t_1 = x_fast[:, -6:-1, 1]
        stats_t = self._current_stats(curr_t)
        stats_t_1 = self._current_stats(curr_t_1)

        attn_t = self._apply_attention(feat_fast_t, slow_out, stats_t)
        attn_t_1 = self._apply_attention(feat_fast_t_1, slow_out, stats_t_1)

        combined_t = torch.cat((feat_fast_t, attn_t), dim=1)
        combined_t_1 = torch.cat((feat_fast_t_1, attn_t_1), dim=1)

        fused_t = self.fusion(combined_t)
        fused_t_1 = self.fusion(combined_t_1)

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
                              batch_size=128, shuffle=True)
                              
    test_loader = DataLoader(TensorDataset(torch.tensor(X_te_f, dtype=torch.float32), 
                                           torch.tensor(X_te_s, dtype=torch.float32), 
                                           torch.tensor(y_te, dtype=torch.float32).view(-1, 1)), 
                             batch_size=128, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BatteryAttentionFusionNet(input_size=5).to(device)
    
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.75, patience=5)

    dummy_array = np.zeros((1, 5))
    scaled_zero_amp = scaler.transform(dummy_array)[0, 1]
    dummy_array[0, 1] = 0.5 
    scaled_half_amp = abs(scaler.transform(dummy_array)[0, 1] - scaled_zero_amp)

    criterion = PhysicsInformedBMSLoss(lambda_penalty=0.1, current_zero_val=scaled_zero_amp, current_threshold=scaled_half_amp)
    
    epoch_num = 50
    total_steps = 0
    t_start_training = time.time()
    
    print(f"\nStarting Training on {device} - Model Parameters: {num_params / 1e6:.2f}M")
    
    for epoch in range(epoch_num):
        model.train()
        train_loss = 0.0
        
        for batch_xf, batch_xs, batch_y in train_loader:
            batch_xf, batch_xs, batch_y = batch_xf.to(device), batch_xs.to(device), batch_y.to(device)
            optimizer.zero_grad()
            
            pred_t, pred_t_1 = model(batch_xf, batch_xs) 
            current_t_scaled = batch_xf[:, -1, 1]
            
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
        
        if (epoch + 1) % 50 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:03d}/{epoch_num} | Train Loss: {train_loss/len(train_loader):.5f} | Test MAE: {epoch_test_mae:.5f}")

    training_time = time.time() - t_start_training

    # ==========================================
    # 4. FINAL EVALUATION AND METRICS
    # ==========================================
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
    # Removed the % symbol and MB/s strings to ease LLM parsing
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