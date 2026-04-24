import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
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
	
	seq_gen = SequenceGenerator(window_size=60)
	WINDOW_SIZE = 60
	input_dim = seq_gen.get_feature_count() * len(TICKERS)
	
	model = ChallengerAgent(input_dim=input_dim, hidden_dim=128, num_stocks=len(TICKERS))
	optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
	bridge = AgentBridge(tickers=TICKERS)
	
	# 2. PREPARE DATA (The "Textbooks")
	print("Creating training sequences across all tickers...")
	try:
		_, labels, _ = seq_gen.create_sequences(master_dfs)
	except Exception as e:
		print(f"Error creating sequences: {e}")
		#fallback: use first ticker only
		_, labels, _ = seq.gen.create_sequences({TICKERS[0]: master_dfs[TICKERS[0]]})
		
	print(f"Training on {len(labels):,} samples")
	
	# 3. THE TRAINING LOOP
	num_epochs = 10
	
	for epoch in range(num_epochs):
		print(f"\n--- Starting Epoch {epoch+1}/{num_epochs} ---")
		portfolio = Portfolio(initial_cash=10000.00)
		model.train() 
		
		for i in range(len(labels)):
			try:
				features_list = []
				current_idx = i + WINDOW_SIZE + 1
				
				for ticker in TICKERS:
					df = master_dfs[ticker]
					start_idx = max(0, current_idx - WINDOW_SIZE + 1)
					window_df = df.iloc[start_idx:current_idx][seq_gen.feature_cols].copy()

					if len(window_df) < WINDOW_SIZE:
						pad = pd.DataFrame(0, index=range(WINDOW_SIZE - len(window_df)), columns=seq_gen.feature_cols)
						window_df = pd.concat([pad, window_df], ignore_index=True)
						
					features_list.append(window_df.values)

				stacked = np.stack(features_list,axis = 0)  #[num_tickers, window, features]
				# Convert sequence to PyTorch tensor [Batch, Window, Features]
				state = torch.FloatTensor(stacked).permute(1,0,2).reshape(1, WINDOW_SIZE, -1)
		
			except (IndexError, KeyError):
				continue
			
			# THE BRAIN MAKES A CHOICE
			action_weights = model(state) 
			
			# THE AGENT EXECUTES THE CHOICE
			try:
				ref_df = list(master_dfs.values())[0]
				current_dt = ref_df.index[min(current_idx, len(ref_df)-1)]
				
				current_prices = {}
				for t in TICKERS:
					df_t = master_dfs[t]
					if current_dt in master_df_t.index:
						price = df_t.loc[current_dt, 'close']
						if not pd.isna(price):
							current_prices[t] = float(price)
					else:
						current_prices[t] = float(df_t['close'].iloc[-1])
			except Exception:
				continue
			
			# Execute trades based on the model's weight output
			bridge.execute_allocation(portfolio, action_weights, current_prices, current_dt)
			
			# 4. CALCULATE THE "GRADE" (Loss/Reward)
			# Compare weights to the Triple Barrier Label
			target_label = torch.zeros((1, len(TICKERS)))
			if i < len(labels):
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
	print("\n" + "="*50)
	print("SUCCESS: Brain saved to challenger_model.pth!")
	print("="*50)
	
if __name__ == "__main__":
    print("This script is now a module. Please run main_runner.py with TRAINING_MODE = True.")