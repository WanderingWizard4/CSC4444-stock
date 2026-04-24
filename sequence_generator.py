import numpy as np
import torch

class SequenceGenerator:
	def __init__(self, window_size=60):
		self.window_size = window_size

	def create_sequences(self, ticker_dfs, feature_cols):
		"""
		ticker_dfs: Dictionary of {ticker: dataframe} from your pipeline
		feature_cols: List of columns to use (e.g., ['close', 'rsi_20', 'news_sentiment'])
		"""
		X = []
		y = []
        
		# We need to make sure all tickers have the same timestamps
		# For simplicity, we'll iterate through one ticker and find overlapping windows
		main_ticker = list(ticker_dfs.keys())[0]
		df_main = ticker_dfs[main_ticker]
        
		print(f"Generating sequences using a {self.window_size} minute window...")
        
		for i in range(len(df_main) - self.window_size):
			# 1. Get the 'Target' (the Labeler output for the NEXT minute)
			# This is what the NN is trying to predict
			target_label = df_main['label'].iloc[i + self.window_size]
            
			# 2. Extract the 'Window' of features
			# Shape: [window_size, num_features]
			window = df_main[feature_cols].iloc[i : i + self.window_size].values
            
			X.append(window)
			y.append(target_label)
            
		return np.array(X), np.array(y)

# Example of how this looks for the NN
# If features = 9 and window = 60
# X shape will be: [Number of Samples, 60, 9]
