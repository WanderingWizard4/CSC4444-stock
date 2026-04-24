import torch
import torch.nn as nn
import torch.nn.functional as F

class ChallengerAgent(nn.Module):
	def __init__(self, input_dim, hidden_dim=128, num_stocks=30):
		super(ChallengerAgent, self).__init__()
        
		# LSTM
		# Patterns the past price data (OHLCV + Technicals)
		self.lstm = nn.LSTM(
			input_size=input_dim, 
			hidden_size=hidden_dim, 
			num_layers=2, 
			batch_first=True, 
			dropout=0.2
		)
        
		# GRU
		# Takes the LSTM's output features and determines the final signal
		self.gru = nn.GRU(
			input_size=hidden_dim, 
			hidden_size=64, 
			num_layers=1, 
			batch_first=True
		)
        
		# DECISION LAYER
		# Maps GRU state for selected stocks to weights
		self.fc = nn.Linear(64, num_stocks)
        
	def forward(self, x):
		"""
		x shape: [batch, window_size, features] 
		(e.g., 32 samples, 60 minutes of history, 9 features)
		"""
		# LSTM processes the entire 60-minute history
		# lstm_out shape: [batch, window_size, hidden_dim]
		lstm_out, _ = self.lstm(x)
        
		# Feed those patterns into the GRU 
		# (This uses the 'patterns' from the LSTM as features)
		gru_out, h_n = self.gru(lstm_out)
        
		# Take the last output of the sequence for the final decision
		# h_n shape: [1, batch, 64] -> remove the first dim
		last_hidden_state = h_n.squeeze(0)
        
		# Convert to 30 stock allocations (Continuous weights)
		output = self.fc(last_hidden_state)
        
		# Softmax ensures our 30 weights add up to 1.0 (Total Portfolio)
		return F.softmax(output, dim=1)

# Example usage for your 30 stocks
# 9 features: O, H, L, C, V, MACD, RSI, Sentiment, Buzz
model = ChallengerAgent(input_dim=9, hidden_dim=128, num_stocks=30)
