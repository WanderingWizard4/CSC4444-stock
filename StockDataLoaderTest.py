from stock_data_loader import StockDataLoader
import pandas as pd
import os
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()

def main():
	data_path = os.getenv('DATA_PATH', '../OHLC 1 minute data/extracted_files')
	SDL = StockDataLoader(base_path=data_path)

	aapl = SDL.load1min("aapl", "2024-01-01", "2024-01-31")
	print(aapl.head())
	print(aapl.tail())
	print(f"\nShape: {aapl.shape}")
	print(f"Date range: {aapl.index[0]} -> {aapl.index[-1]}")
	print(f"Memory Usage: {aapl.memory_usage(deep=True).sum() / 1024**2:.1f} MB")

if __name__ == "__main__":
	main()

	
