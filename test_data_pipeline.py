from stock_data_loader import StockDataLoader
from feature_engineering import MultiTimeFrameFeatures
from rolling_features import RollingFeatures

def main():
	# Adjust the path if needed
	SDL = StockDataLoader(base_path="../OHLC 1 minute data/extracted_files")
	
	# Create multi-timeframe data (use short period first to test)
	mfe = MultiTimeFrameFeatures(SDL)
	multi_tf = mfe.create_features("AAPL", start="2024-01-01", end="2024-03-31")
	
	print("\n" + "="*60)
	rf = RollingFeatures()
	processed = rf.process(multi_tf)
	
	# Show summary
	for tf, df in processed.items():
		print(f"{tf:>8} : {df.shape[0]:>6,} bars | {df.shape[1]} columns")
		if len(df) > 0:
			print(f"   Last row MACD: {df['macd'].iloc[-1]:.4f} | RSI_20: {df['rsi_20'].iloc[-1]:.2f}")
		else:
			print("-" * 60)

if __name__ == '__main__':
	main()
