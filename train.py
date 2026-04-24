import torch
import torch.nn as nn
import torch.nn.functional as F
from challenger_model import ChallengerAgent
from trading_environment import Portfolio
from sequence_generator import SequenceGenerator
from agent_bridge import AgentBridge

def train_agent(master_dfs):
    """
    master_dfs: The dictionary of dataframes from your pipeline
    """
    # 1. INITIALIZE COMPONENTS
    TICKERS = list(master_dfs.keys())
    # 9 features: open, high, low, close, volume, macd, rsi_20, sentiment, buzz
    model = ChallengerAgent(input_dim=9, hidden_dim=128, num_stocks=len(TICKERS))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    bridge = AgentBridge(tickers=TICKERS)
    
    # 2. PREPARE DATA (The "Textbooks")
    seq_gen = SequenceGenerator(window_size=60)
    feature_cols = ['open', 'high', 'low', 'close', 'volume', 'macd', 'rsi_20', 'news_sentiment', 'news_buzz']
    
    # We use the first ticker (e.g., NVDA) as the primary training baseline
    print(f"Creating training sequences for {TICKERS[0]}...")
    sequences, labels = seq_gen.create_sequences(master_dfs[TICKERS[0]], feature_cols)
    
    # 3. THE TRAINING LOOP
    num_epochs = 10 
    
    for epoch in range(num_epochs):
        print(f"\n--- Starting Epoch {epoch+1}/{num_epochs} ---")
        portfolio = Portfolio(initial_cash=10000.00)
        current_prices = {} 
        
        model.train() # Set to training mode
        for i in range(len(sequences)):
            # Convert sequence to PyTorch tensor [Batch, Window, Features]
            state = torch.FloatTensor(sequences[i]).unsqueeze(0) 
            
            # THE BRAIN MAKES A CHOICE
            action_weights = model(state) 
            
            # THE AGENT EXECUTES THE CHOICE
            # Use i+60 because the sequence covers the 60 minutes prior to this point
            current_dt = master_dfs[TICKERS[0]].index[i + 60]
            current_prices = {t: master_dfs[t].iloc[i + 60]['close'] for t in TICKERS}
            
            # Execute trades based on the model's weight output
            bridge.execute_allocation(portfolio, action_weights, current_prices, current_dt)
            
            # 4. CALCULATE THE "GRADE" (Loss/Reward)
            # Compare weights to the Triple Barrier Label
            target_label = torch.zeros((1, len(TICKERS)))
            target_label[0, 0] = labels[i] 
            
            loss = F.mse_loss(action_weights, target_label) 
            
            # 5. BACKPROPAGATION
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if i % 500 == 0:
                print(f"  Step {i}: Loss {loss.item():.6f} | Portfolio: ${portfolio.get_current_value(current_prices):.2f}")
            
        print(f"Epoch {epoch+1} complete. Final Value: ${portfolio.get_current_value(current_prices):.2f}")

    # --- THE BRAIN SAVE ---
    # Save the weights so main_runner.py can load them later
    torch.save(model.state_dict(), "challenger_model.pth")
    print("\n" + "="*40)
    print("SUCCESS: Brain saved to challenger_model.pth!")
    print("="*40)

if __name__ == "__main__":
    print("This script is now a module. Please run main_runner.py with TRAINING_MODE = True.")