import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from challenger_model import ChallengerAgent
from trading_environment import Portfolio
from sequence_generator import SequenceGenerator
from agent_bridge import AgentBridge

def normalize_features(tensor: torch.Tensor) -> torch.Tensor:
    """Z-score normalization to help the model converge."""
    mean = tensor.mean(dim=(0, 1), keepdim=True)
    std = tensor.std(dim=(0, 1), keepdim=True) + 1e-8
    return (tensor - mean) / std

def train_agent(master_dfs):
    """
    master_dfs: The dictionary of dataframes from your pipeline
    """
    # 1. INITIALIZE COMPONENTS
    TICKERS = list(master_dfs.keys())
    seq_gen = SequenceGenerator(window_size=60)
    WINDOW_SIZE = 60
    input_dim = seq_gen.get_feature_count() * len(TICKERS)
    
    model = ChallengerAgent(input_dim=input_dim, hidden_dim=128, num_stocks=len(TICKERS))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    bridge = AgentBridge(tickers=TICKERS)
    
    # 2. PRE-PROCESS DATA (The "Fast Lane" Strategy)
    print("Converting DataFrames to high-speed NumPy tensors...")
    feature_data = []
    price_data = []
    
    for ticker in TICKERS:
        df = master_dfs[ticker]
        feature_data.append(df[seq_gen.feature_cols].values)
        price_data.append(df['close'].values)
    
    # [Num_Tickers, Timesteps, Num_Features]
    full_features = np.stack(feature_data, axis=0) 
    # [Num_Tickers, Timesteps]
    full_prices = np.stack(price_data, axis=0)
    
    # Labels (The "Answers")
    print("Creating training sequences and labels...")
    try:
        _, labels, _ = seq_gen.create_sequences(master_dfs)
    except Exception as e:
        print(f"Error creating sequences: {e}")
        _, labels, _ = seq_gen.create_sequences({TICKERS[0]: master_dfs[TICKERS[0]]})
    
    num_samples = len(labels)
    print(f"Training on {num_samples:,} samples")
    
    # 3. THE TRAINING LOOP
    num_epochs = 1
    current_prices = {}
    
    for epoch in range(num_epochs):
        print(f"\n--- Starting Epoch {epoch+1}/{num_epochs} ---")
        portfolio = Portfolio(initial_cash=10000.00)
        model.train() 
        
        # SAFETY FIX: The loop must not exceed the length of the actual price/index arrays.
        # We use the first ticker's length as our physical boundary.
        max_physical_steps = len(master_dfs[TICKERS[0]])
        loop_limit = max_physical_steps - WINDOW_SIZE
        
        # Start the loop using the safer limit
        for i in range(loop_limit):
            # Define current indices relative to the full data block
            current_idx = i + WINDOW_SIZE
            
            # FAST NUMPY SLICING (Instead of Pandas .iloc)
            # Shape: [Num_Tickers, Window, Features]
            state_np = full_features[:, i:current_idx, :]
            
            # Convert to Tensor and Normalize
            # Reshape to [1, Window, Total_Features]
            state = torch.FloatTensor(state_np).permute(1, 0, 2).reshape(1, WINDOW_SIZE, -1)
            state = normalize_features(state)
            
            # THE BRAIN MAKES A CHOICE
            action_weights = model(state) 
            
            # THE AGENT EXECUTES THE CHOICE
            # This is where the crash happened—now safe due to loop_limit
            current_dt = master_dfs[TICKERS[0]].index[current_idx]
            
            for t_idx, ticker in enumerate(TICKERS):
                price = full_prices[t_idx, current_idx]
                current_prices[ticker] = float(price) if not np.isnan(price) else 0.0

            # Execute trades
            bridge.execute_allocation(portfolio, action_weights, current_prices, current_dt)
            
            # 4. CALCULATE LOSS
            target_label = torch.zeros((1, len(TICKERS)))
            target_label[0, 0] = labels[i] # Target the first ticker
            
            loss = F.mse_loss(action_weights, target_label) 
            
            # 5. BACKPROPAGATION
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if i % 1000 == 0:
                val = portfolio.get_current_value(current_prices)
                # Updated print to show progress against the real limit
                print(f"   Step {i}/{loop_limit} | Loss {loss.item():.6f} | Portfolio: ${val:.2f}")
        
        final_val = portfolio.get_current_value(current_prices)
        print(f"Epoch {epoch+1} complete. Final Value: ${final_val:.2f}")
        
    # --- THE BRAIN SAVE ---
    torch.save(model.state_dict(), "challenger_model.pth")
    print("\n" + "="*50)
    print("SUCCESS: Brain saved to challenger_model.pth!")
    print("="*50)
    
if __name__ == "__main__":
    print("This script is now a module. Please run main_runner.py with TRAINING_MODE = True.")