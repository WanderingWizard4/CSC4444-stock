import pandas as pd
import numpy as np

class RollingFeatures:
	'''
	Creates rolling widows features on top of multi-timeframe data.
	Most of the neural net's inputs will come from here. 
	'''

	def __init__(self):
		self.default_config={
			'5min': [5,10,20],
			'15min': [5,10,20],
			'30min': [5,10,20],
			'1hour': [5,10,20,50],
			'1day': [5,10,20,50],
			'1week': [5,10,20],
			'1month': [5,10,20]
		}

	def add_rolling_features(
		self,
		df: pd.DataFrame,
		tf_key:str) -> pd.DataFrame:

		'''
		Add rolling features to one timeframe's dataframe. 
		tf_key example: '5min', '1hour', '1day'
		'''

		if df.empty:
			return df

		df =df.copy()
		windows = self.default_config.get(tf_key, [5,10,20])

		for window in windows:

			#Momentum and returns
			df[f'return_{window}'] = df['close'].pct_change(window)
			df[f'log_return_{window}'] = np.log(df['close'] /df['close'].shift(window))

			# Volatility
			df[f'vol_{window}'] = df[f'log_return_{window}'].rolling(window).std()

			# RSI
			df[f'rsi_{window}'] = self._rsi(df['close'], window)

			# EMA Ratio
			ema = df['close'].ewm(span=window, adjust=False).mean()
			df[f'ema_{window}'] = ema
			df[f'price_ema_ratio_{window}'] = df['close'] / ema

		#MACD
		df = self._add_macd(df)

		#ATR
		df = self._add_atr(df)

		return df.dropna(how='all')

	def _rsi(self, series: pd.Series, window:int)-> pd.Series:
		delta = series.diff()
		gain = delta.where(delta > 0, 0).rolling(window).mean()
		loss = -delta.where(delta < 0, 0).rolling(window).mean()
		rs = gain / loss
		return 100 - (100 / (1 + rs))

	def _add_macd(self, df:pd.DataFrame, window:int = 14) -> pd.DataFrame:
		ema12 = df['close'].ewm(span=12, adjust=False).mean()
		ema26 = df['close'].ewm(span=26, adjust=False).mean()
		df['macd'] = ema12 - ema26
		df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
		df['macd_hist'] = df['macd'] - df['macd_signal']
		return df

	def _add_atr(self, df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
		high_low = df['high'] - df['low']
		high_close = np.abs(df['high'] - df['close'].shift())
		low_close = np.abs(df['low'] - df['close'].shift())
		tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
		df['atr'] = tr.rolling(window).mean()
		return df

	def _add_vwap(self, df:pd.DataFrame) -> pd.DataFrame:
		df = df.copy()
		df['cum_vol'] = df['volume'].cumsum()
		df['cum_price_vol'] = (df['close'] * df['volume']).cumsum()
		df['vwap'] = df['cum_price_vol'] / df['cum_vol']
		return df.drop(columns=['cum_vol', 'cum_price_col'], errors='ignore')

	def process(self, multi_tf_data: dict[str, pd.DataFrame]) -> dict[str,pd.DataFrame]:
		'''Apply rolling features to all timeframes'''
		processed = {}
		print ("Adding rolling features...\n")
		for tf_key, df in multi_tf_data.items():
			print(f"Processing {tf_key:>8}...", end=" ")
			processed[tf_key] = self.add_rolling_features(df, tf_key)
			print(f" ...{processed[tf_key].shape[1]} columns")

		print(f"\nRolling features added to {len(processed)} timeframes.\n")
		return processed
