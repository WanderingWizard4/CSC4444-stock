import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import multiprocessing as mp
import platform
import warnings
warnings.filterwarnings("ignore")

from challenger_model import ChallengerAgent
from sequence_generator import SequenceGenerator


# Cross-platform setup
if platform.system() == "Darwin":
    mp.set_start_method('fork', force=True)
elif platform.system() == "Windows":
    mp.set_start_method('spawn', force=True)


class TradingDataset(Dataset):
    def __init__(self, master_dfs: dict, seq_gen: SequenceGenerator, tickers: list):
        self.master_dfs = master_dfs
        self.seq_gen = seq_gen
        self.tickers = tickers
        self.WINDOW_SIZE = seq_gen.window_size
        
        self.valid_indices = list(range(len(list(master_dfs.values())[0]) - self.WINDOW_SIZE - 1))
        print(f"Dataset ready with {len(self.valid_indices):,} samples")

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        i = self.valid_indices[idx]
        features_list = []
        labels_list = []
        
        for ticker in self.tickers:
            df = self.master_dfs[ticker]
            start_idx = max(0, i - self.WINDOW_SIZE + 1)
            window_df = df.iloc[start_idx:i+1][self.seq_gen.feature_cols].copy()

            if len(window_df) < self.WINDOW_SIZE:
                pad = pd.DataFrame(0, index=range(self.WINDOW_SIZE - len(window_df)),
                                 columns=self.seq_gen.feature_cols)
                window_df = pd.concat([pad, window_df], ignore_index=True)

            # CRITICAL: Replace any NaNs/Infs from rolling features
            window_values = window_df.values
            window_values = np.nan_to_num(window_values, nan=0.0, posinf=1.0, neginf=-1.0)
            features_list.append(window_values)

            # Label
            label_idx = i + self.WINDOW_SIZE
            label = df.iloc[label_idx]['label'] if label_idx < len(df) else 0
            labels_list.append(label)

        stacked = np.stack(features_list, axis=0)
        state = torch.FloatTensor(stacked).permute(1, 0, 2).reshape(1, self.WINDOW_SIZE, -1)
        
        labels = torch.tensor(labels_list, dtype=torch.float32)
        return state.squeeze(0), labels


def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"✅ Using CUDA: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = torch.device("cpu")   # ← TEMPORARY: Force CPU for stability
        print("⚠️  MPS causing NaNs → Falling back to CPU (more stable for LSTM)")
    else:
        device = torch.device("cpu")
        print("⚠️ Using CPU")
    return device


def normalize_features(tensor: torch.Tensor) -> torch.Tensor:
    # More robust normalization
    mean = tensor.mean(dim=(0, 1), keepdim=True)
    std = tensor.std(dim=(0, 1), keepdim=True) + 1e-8
    normalized = (tensor - mean) / std
    return torch.nan_to_num(normalized, nan=0.0, posinf=1.0, neginf=-1.0)


def train_agent(master_dfs, num_epochs=5, batch_size=32):
    TICKERS = list(master_dfs.keys())
    seq_gen = SequenceGenerator(window_size=60)
    input_dim = seq_gen.get_feature_count() * len(TICKERS)

    model = ChallengerAgent(input_dim=input_dim, hidden_dim=128, num_stocks=len(TICKERS))
    device = get_device()
    model = model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)  # Lower LR

    dataset = TradingDataset(master_dfs, seq_gen, TICKERS)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=min(8, mp.cpu_count() - 2),
        pin_memory=False,   # Disabled for stability
        persistent_workers=True
    )

    print(f"\nStarting training on {device} | Batch size: {batch_size}\n")

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0

        print(f"--- Epoch {epoch+1}/{num_epochs} ---")

        for batch_idx, (states, target_labels) in enumerate(dataloader):
            states = states.to(device)
            target_labels = target_labels.to(device)
            states = normalize_features(states)

            optimizer.zero_grad()
            weights = model(states)

            # Label-aware target
            target = torch.ones_like(weights) / len(TICKERS)
            target = target + (target_labels * 0.12)
            target = torch.clamp(target, min=0.01)
            target = target / (target.sum(dim=1, keepdim=True) + 1e-8)

            loss = F.mse_loss(weights, target)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Prevent exploding gradients
            optimizer.step()

            total_loss += loss.item()

            if batch_idx % 200 == 0:
                print(f"  Batch {batch_idx:5d} | Loss: {loss.item():.6f} | "
                      f"Mean Weight: {weights.mean().item():.4f}")

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1} completed. Avg Loss: {avg_loss:.6f}\n")

    torch.save(model.state_dict(), "challenger_model.pth")
    print("✅ Training finished! Model saved.")


if __name__ == "__main__":
    print("Run via main_runner.py")