from stock_data_loader import StockDataLoader
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv


load_dotenv()
def main():
	SDL = StockDataLoader(base_path="../OHLC 1 minute data/extracted_files")

	aapl = SDL.load1min("aapl", "2002-01-01", "2025-12-31")
	print(aapl.head())
	print(aapl.tail())
	print(f"\nShape: {aapl.shape}")
	print(f"Date range: {aapl.index[0]} -> {aapl.index[-1]}")
	print(f"Memory Usage: {aapl.memory_usage(deep=True).sum() / 1024**2:.1f} MB")

if __name__ == "__main__":
	main()

	
