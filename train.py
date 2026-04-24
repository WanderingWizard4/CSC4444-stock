import torch
import torch.nn as nn
import torch.nn.functional as F  # Fix for F.mse_loss
from challenger_model import ChallengerAgent
from trading_environment import Portfolio
from sequence_generator import SequenceGenerator
from agent_bridge import AgentBridge

def train_agent(master_dfs):
	"""
	master_dfs: The dictionary of dataframes from your pipeline
	"""
	# 1. INITIALIZE COMPONENTS
	# Updated to 9 features: close, volume, macd, rsi, news_sentiment, news_buzz + 3 others from OHLC
	TICKERS = list(master_dfs.keys())
	model = ChallengerAgent(input_dim=9, hidden_dim=128, num_stocks=len(TICKERS))
	optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
	bridge = AgentBridge(tickers=TICKERS)
    
	# 2. PREPARE DATA (The "Textbooks")
	seq_gen = SequenceGenerator(window_size=60)
	# Using the first ticker (e.g., NVDA) to generate sequences for training
	# Note: feature_cols must match your input_dim
	feature_cols = ['open', 'high', 'low', 'close', 'volume', 'macd', 'rsi_20', 'news_sentiment', 'news_buzz']
    
	# Generate sequences from one of your processed dataframes
	# In a full run, you'd aggregate these across all 30 tickers
	sequences, labels = seq_gen.create_sequences(master_dfs[TICKERS[0]], feature_cols)
    
	# 3. THE TRAINING LOOP
	num_epochs = 10 
    
	for epoch in range(num_epochs):
		print(f"Starting Epoch {epoch+1}")
		portfolio = Portfolio(initial_cash=10000.00)
		current_prices = {}
        
		for i in range(len(sequences)):
			# Convert sequence to PyTorch tensor [Batch, Window, Features]
			state = torch.FloatTensor(sequences[i]).unsqueeze(0) 
            
			# THE BRAIN MAKES A CHOICE
			action_weights = model(state) 
            
			# THE AGENT EXECUTES THE CHOICE
			# Get current prices for all stocks at this specific minute
			# We use i+60 because the sequence represents the 60 minutes LEADING UP to now
			current_dt = master_dfs[TICKERS[0]].index[i + 60]
			current_prices = {t: master_dfs[t].iloc[i + 60]['close'] for t in TICKERS}
            
			# Use the Bridge to turn weights into portfolio actions
			bridge.execute_allocation(portfolio, action_weights, current_prices, current_dt)
            
			# 4. CALCULATE THE "GRADE" (Loss/Reward)
			# Compare model weights to the Triple Barrier Label (y)
			# We need to reshape the label to match action_weights
			target_label = torch.zeros((1, len(TICKERS)))
			target_label[0, 0] = labels[i] # Target the first stock for this simple test
            
			loss = F.mse_loss(action_weights, target_label) 
            
			# 5. BACKPROPAGATION
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
            
			if i % 500 == 0:
				print(f"  Step {i}: Loss {loss.item():.6f} | Portfolio: ${portfolio.get_current_value(current_prices):.2f}")
            
		print(f"Epoch {epoch+1} complete. Final Value: ${portfolio.get_current_value(current_prices):.2f}")

if __name__ == "__main__":
	# This assumes you have already run your pipeline to get master_dfs
	# For a test run, you can import your pipeline runner here
	print("Please run this via main_runner.py to ensure master_dfs is populated.")
