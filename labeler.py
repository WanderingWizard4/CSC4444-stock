import pandas as pd
import numpy as np

class TripleBarrierLabeler:
	def __init__(self, pt_multiplier=1.0, sl_multiplier=1.0, vertical_barrier_mins=30):
		"""
		pt_multiplier: How many standard deviations for Profit Take
		sl_multiplier: How many standard deviations for Stop Loss
		vertical_barrier_mins: Max minutes to hold the trade
		"""
		self.pt = pt_multiplier
		self.sl = sl_multiplier
		self.window = vertical_barrier_mins

	def compute_volatility(self, df, span=100):
		"""Dynamic volatility helps adjust barriers to market speed"""
		# Calculate daily volatility scaled to 1-min timeframe
		returns = df['close'].pct_change()
		vol = returns.ewm(span=span).std()
		return vol.dropna()

	def label_data(self, df):
		"""Main labeling logic"""
		print(f"Labeling {len(df)} rows...")
        
		# 1. Get dynamic volatility (so barriers aren't static)
		vol = self.compute_volatility(df)
        
		labels = []
		prices = df['close'].values
		vol_values = vol.reindex(df.index).bfill().values
        
		# 2. Iterate and look ahead (the 'Barriers')
		# Note: Vectorize for speed @ production, but this is clearer for testing
		for i in range(len(prices) - self.window):
			start_price = prices[i]
			current_vol = vol_values[i]
            
			# Define barriers based on current volatility
			upper_barrier = start_price * (1 + current_vol * self.pt)
			lower_barrier = start_price * (1 - current_vol * self.sl)
            
			# Look ahead in the 'window'
			future_prices = prices[i+1 : i+1+self.window]
            
			label = 0 # Default: Vertical Barrier (Hold/Neutral)
			for p in future_prices:
				if p >= upper_barrier:
					label = 1  # Buy/Profit hit
					break
				elif p <= lower_barrier:
					label = -1 # Sell/Loss hit
					break
			labels.append(label)
        
		# Pad the end with Neutral labels since we can't look past the end of the data
		labels += [0] * self.window
		df['label'] = labels
		return df

if __name__ == "__main__":
	# Quick test logic
	print("Testing Labeler...")
	test_data = pd.DataFrame({'close': np.random.normal(100, 1, 1000)})
	labeler = TripleBarrierLabeler()
	labeled_df = labeler.label_data(test_data)
	print(f"Labels generated: {labeled_df['label'].value_counts().to_dict()}")
