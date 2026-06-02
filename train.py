"""
Autoresearch pretraining script per LSTM Multi-Branch (PINN) con Attention e Transformer.
Script unificato, ottimizzato per l'esecuzione su Kaggle Cloud.
"""

import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler

# Assicura che Kaggle possa leggere i file .xlsx senza crashare
os.system('pip install openpyxl -q')

# ==========================================
# 1. PREPARAZIONE DEI DATI
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
    print(f"Caricamento dati da {filename}...")
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
# 2. DEFINIZIONE DEL MODELLO E DELLA LOSS
# ==========================================
class Attention(nn.Module):
    def __init__(self, input_dim):
        super(Attention, self).__init__()
        self.attention_weights = nn.Parameter(torch.Tensor(input_dim, 1))
        nn.init.xavier_uniform_(self.attention_weights)

    def forward(self, x):
        scores = torch.matmul(x, self.attention_weights).squeeze(-1)
        alpha = torch.softmax(scores, dim=1)
        context = (x * alpha.unsqueeze(-1)).sum(dim=1)
        return context, alpha

class MultiBranchLSTMWithAttentionAndTransformer(nn.Module):
    def __init__(self, input_dim_fast, input_dim_slow, hidden_dim, num_layers):
        super(MultiBranchLSTMWithAttentionAndTransformer, self).__init__()
        
        self.lstm_fast = nn.LSTM(input_dim_fast, hidden_dim, num_layers=1, batch_first=True)
        self.lstm_slow = nn.LSTM(input_dim_slow, hidden_dim, num_layers=1, batch_first=True)
        
        self.attention_fast = Attention(hidden_dim)
        self.attention_slow = Attention(hidden_dim)
        
        self.transformer_layer = nn.TransformerEncoderLayer(d_model=hidden_dim * 2, nhead=2)
        self.transformer_encoder = nn.TransformerEncoder(self.transformer_layer, num_layers=1)
        
        self.fc = nn.Linear(hidden_dim * 2, 1)

    def forward(self, x_fast, x_slow):
        out_fast, _ = self.lstm_fast(x_fast)
        out_slow, _ = self.lstm_slow(x_slow)
        
        context_fast, _ = self.attention_fast(out_fast)
        context_slow, _ = self.attention_slow(out_slow)
        
        combined = torch.cat((context_fast, context_slow), dim=1).unsqueeze(1)
        transformer_out = self.transformer_encoder(combined).squeeze(1)
        
        out = self.fc(transformer_out)
        return out

class PhysicsInformedBMSLoss(nn.Module):
    def __init__(self, lambda_penalty=0.0, current_zero_val=0.0, current_threshold=0.05):
        super(PhysicsInformedBMSLoss, self).__init__()
        self.mae = nn.L1Loss()
        self.lambda_penalty = lambda_penalty
    
    def forward(self, pred, target):
        return self.mae(pred, target)

# ==========================================
# 3. FUNZIONE DI TRAINING E VALUTAZIONE
# ==========================================
def train_and_evaluate():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Caricamento e preparazione dei dati
    X_tr_f, X_tr_s, y_tr, X_te_f, X_te_s, y_te, scaler = process_and_split_data()
    train_dataset = TensorDataset(torch.tensor(X_tr_f, dtype=torch.float32), torch.tensor(X_tr_s, dtype=torch.float32), torch.tensor(y_tr, dtype=torch.float32))
    test_dataset = TensorDataset(torch.tensor(X_te_f, dtype=torch.float32), torch.tensor(X_te_s, dtype=torch.float32), torch.tensor(y_te, dtype=torch.float32))
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    # Definizione del modello
    input_dim_fast = X_tr_f.shape[2]
    input_dim_slow = X_tr_s.shape[2]
    hidden_dim = 64  # Ridotto da 128 a 64
    num_layers = 1   # Ridotto il numero di layer
    model = MultiBranchLSTMWithAttentionAndTransformer(input_dim_fast, input_dim_slow, hidden_dim, num_layers).to(device)
    
    # Definizione della loss e dell'ottimizzatore
    criterion = PhysicsInformedBMSLoss(lambda_penalty=0.1)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=10, verbose=True)
    
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_steps = 0
    epoch_num = 100
    t_start_training = time.time()
    
    print(f"\nInizio Addestramento su {device} - Parametri Modello: {num_params / 1e6:.2f}M")
    
    # Ciclo di addestramento
    for epoch in range(epoch_num):
        model.train()
        train_loss = 0.0
        for batch_xf, batch_xs, batch_y in train_loader:
            batch_xf, batch_xs, batch_y = batch_xf.to(device), batch_xs.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            pred = model(batch_xf, batch_xs)
            loss = criterion(pred, batch_y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            total_steps += 1
        
        # Valutazione sul set di test
        model.eval()
        epoch_test_mae = 0.0
        with torch.no_grad():
            for batch_xf_te, batch_xs_te, batch_y_te in test_loader:
                batch_xf_te, batch_xs_te, batch_y_te = batch_xf_te.to(device), batch_xs_te.to(device), batch_y_te.to(device)
                pred_t_eval = model(batch_xf_te, batch_xs_te)
                epoch_test_mae += nn.L1Loss()(pred_t_eval, batch_y_te).item()
        
        epoch_test_mae /= len(test_loader)
        scheduler.step(epoch_test_mae)
        
        if (epoch + 1) % 50 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:03d}/{epoch_num} | Train Loss: {train_loss/len(train_loader):.5f} | Test MAE: {epoch_test_mae:.5f}")

    training_time = time.time() - t_start_training

    # ==========================================
    # 4. VALUTAZIONE FINALE E METRICHE
    # ==========================================
    model.eval()
    all_y_pred, all_y_true = [], []
    with torch.no_grad():
        for batch_xf_te, batch_xs_te, batch_y_te in test_loader:
            batch_xf_te, batch_xs_te = batch_xf_te.to(device), batch_xs_te.to(device)
            pred_t_final = model(batch_xf_te, batch_xs_te)
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

    total_time = time.time() - t_start_training

    print("\n" + "="*40)
    print("FINAL EVALUATION METRICS")
    print("="*40)
    print(f"test_mae_percent:   {mae * 100:.4f}")
    print(f"peak_vram_mb:       {peak_vram_mb:.1f}")

if __name__ == "__main__":
    try:
        train_and_evaluate()
    except FileNotFoundError:
        print("Errore: Dataset non trovato. Verifica di essere su Kaggle e controlla il path.")