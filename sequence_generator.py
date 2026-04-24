import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import torch

class SequenceGenerator:
	def __init__(self, window_size=60):
		self.window_size = window_size
		self.feature_cols = [
			#price, returns, volume
			'open', 'high', 'low', 'close', 'volume',
			'return_5', 'return_10', 'return_20',
			'log_return_5', 'log_return_10', 'log_return_20',

			#volatility
			'vol_5', 'vol_10', 'vol_20',

			#Technical indicators

			#RSI 
			'rsi_5', 'rsi_10', 'rsi_20', 
			
			#MACD
			'macd', 'macd_signal', 'macd_hist',
			
			#ATR
			'atr',

			#EMA-based features
			'price_ema_ratio_5', 'price_ema_ratio_10', 'price_ema_ratio_20',

			#raw EMA
			'ema_5', 'ema_10', 'ema_20',

			#sentiment 
			'news_sentiment', 'news_buzz'
		]

	def create_sequences(self, 
						 master_data: Dict[str, pd.DataFrame]) -> Tuple[np.ndarray, np.array, List[str]]:
		
		"""
		create seequences across All tickers.
		Returns:
			X: shape [num_samples, window_size, num_features]
			Y: shape [num_samples] (labels from TripleBarrier)
			feature_names: list of features used
		"""
		if not master_data:
			raise ValueError("master_data is empty")

		print(f"Generating sequences with window = {self.window_size} using {len(self.feature_cols)} features...")
		
		X = []
		y = []
		used_tickers = []
        
		for ticker, df in master_data.items():
			if df.empty or 'label' not in df.columns:
				print(f"   Skipping {ticker} (no label column)")
				continue

			#ensure all required features
			missing = [col for col in self.feature_cols if col not in df.columns]
			if missing:
				print(f"   Warning: {ticker} missing features: {missing[:5]}...")
				#Use available features for the ticker
				available_features = [col for col in self.features_cols if col in df.columns]
				if not available_features:
					continue
				feature_data = df[available_features].values
			else:
				feature_data = df[self.feature_cols].values

			labels = df['label'].values

			#Create rolling windows
			for i in range(len(df) - self.window_size):
				window = feature_data[i : i + self.window_size]
				target_label = labels[i + self.window_size]

				X.append(window)
				y.append(target_label)
				used_tickers.append(ticker)

		X = np.array(X, dtype=np.float32)
		y = np.array(y, dtype=np.int8)

		print(f"Generated {len(X):,} sequences from {len(set(used_tickers))} tickers")
		print(f"   Shape: X={X.shape}, y={y.shape}")

		return X, y, self.feature_cols

	def get_feature_count(self)-> int:
		'''return number of featuires the model should expect'''
		return len(self.feature_cols)
