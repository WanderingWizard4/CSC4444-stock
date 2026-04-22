from stock_data_loader import StockDataLoader
from feature_engineering import MultiTimeFrameFeatures
from rolling_features import RollingFeatures
from sentiment_loader import SentimentLoader
from labeler import TripleBarrierLabeler

def main():
	# Adjust the path if needed
	SDL = StockDataLoader(base_path="../OHLC 1 minute data/extracted_files")
	
	# Create multi-timeframe data (use short period first to test)
	mfe = MultiTimeFrameFeatures(SDL)
	multi_tf = mfe.create_features("AAPL", start="2024-01-01", end="2024-03-31")
	
	# Initialize the sentiment loader
	print("Integrating Finnhub sentiment data...")
	sl = SentimentLoader()
	for tf in multi_tf:
		multi_tf[tf] = sl.add_sentiment_to_df(multi_tf[tf], "AAPL")
	
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

	# This creates the "Target" for the AI to learn from
	print("\n" + "="*60)
	print("Labeling data with Triple Barrier method...")
	tbl = TripleBarrierLabeler(pt_multiplier=2, sl_multiplier=2, vertical_barrier_mins=60)

	#Label the 5min timeframe as a starting point
	if '5min' in processed:
		final_df = tbl.label_data(processed['5min'])
		print(f"Final labeled data shape: {final_df.shape}")
		print(f"Label distribution:\n{final_df['label'].value_counts()}")

if __name__ == '__main__':
	main()
