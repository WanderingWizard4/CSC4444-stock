import pandas as pd
from dotenv import load_dotenv
from stock_data_loader import StockDataLoader
from feature_engineering import MultiTimeFrameFeatures
from rolling_features import RollingFeatures
from sentiment_loader import SentimentLoader
from labeler import TripleBarrierLabeler

def main():
	# Load environment variables
	load_dotenv() 

	TICKERS = [
		'NVDA', 'AAPL', 'MSFT', 'AMZN', 'WMT', 'JPM', 'V', 'JNJ', 'CAT', 'CVX', 
		'CSCO', 'PG', 'HD', 'KO', 'UNH', 'MRK', 'GS', 'AXP', 'MCD', 'IBM', 
		'VZ', 'AMGN', 'DIS', 'BA', 'CRM', 'HON', 'SHW', 'MMM', 'NKE', 'TRV'
	]
    
	DATA_PATH = "../OHLC 1 minute data/extracted_files"
    
	SDL = StockDataLoader(base_path=DATA_PATH)
	sl = SentimentLoader()
	rf = RollingFeatures()
	tbl = TripleBarrierLabeler(pt_multiplier=2.0, sl_multiplier=1.0, vertical_barrier_mins=60)
    
	master_training_data = {}

	print(f"Starting pipeline for {len(TICKERS)} tickers...")

	for ticker in TICKERS:
		try:
			print(f"\nProcessing {ticker}...")
            
			# LOAD & RESAMPLE 
			mfe = MultiTimeFrameFeatures(SDL)
			multi_tf = mfe.create_features(ticker, start="2024-01-01", end="2024-03-31")
            
			# INTEGRATE SENTIMENT
			for tf in multi_tf:
				multi_tf[tf] = sl.add_sentiment_to_df(multi_tf[tf], ticker)
            
			# ROLLING FEATURES
			processed = rf.process(multi_tf)
            
			# GENERATE CONTINUOUS LABELS
			if '5min' in processed:
				labeled_df = tbl.label_data(processed['5min'])
				master_training_data[ticker] = labeled_df
                
				print(f"Done with {ticker}: {labeled_df.shape[0]} bars ready.")
            
		except Exception as e:
			print(f"Error processing {ticker}: {e}")
			continue

	# SUMMARY FOR THE TEAM
	print("\n" + "="*60)
	print("PIPELINE COMPLETE")
	print(f"Successfully processed {len(master_training_data)} / {len(TICKERS)} stocks.")
    
	# Save a sample, comment out and update for full output data
	if 'NVDA' in master_training_data:
		master_training_data['NVDA'].to_csv("NVDA_training_sample.csv")
		print("Saved NVDA_training_sample.csv for review.")

if __name__ == '__main__':
	main()
