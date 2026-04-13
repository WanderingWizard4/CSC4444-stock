import pandas as pd
from stock_data_loader import StockDataLoader as SDL

class MultiTimeFrameFeatures:
	def __init__(self, loader: SDL):
		self.loader = loader

	def resample_ohlcv(self, df:pd.DataFrame, rule: str) -> pd.DataFrame:
		"""Helper to resample with proper OHLCV aggregation"""
		if df.empty:
			return df
		return df.resample(rule).agg({
			'open': 'first',
			'high': 'max',
			'low': 'min',
			'close': 'last',
			'volume': 'sum'
		}).dropna()

	def create_features(
		self,
		ticker: str, 
		start: str = None, 
		end: str = None, 
		lookback_days: int = 60) -> dict:

		'''Main method to load raw data and create multiple resampled time frames'''

		#load raw 1 min 
		raw_df = self.loader.load1min(ticker=ticker, start=start, end=end)

		if raw_df.empty:
			print(f"No data loaded for {ticker}")
			return{}

		timeframes_list = ['5min', '15min', '30min', '1h', '1W', '1ME']

		data = {}

		for tf in timeframes_list:
			#create a readable key for labels
			key = (
				tf.replace('h', 'hour')
				.replace('D', 'day') 
				.replace('W', 'week')
				.replace('ME', 'month')
			)

			resampled = self.resample_ohlcv(raw_df, tf)
			data[key] = resampled

			print(f"Created {key:>8} -> {len(resampled):>5,} bars")

		print(f"\nMulti-timeframe features created for {ticker} "
			  f"({len(data)} timeframes)\n")

		return data
