from stock_data_loader import StockDataLoader
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv


load_dotenv()
def main():
	SDL = StockDataLoader(base_path="../OHLC 1 minute data/extracted_files")
	# Updated for Jordan's computer file structure
	# DATA_PATH = r"D:\OHLC 1992-2025-2 tar files"
	# SDL = StockDataLoader(base_path=DATA_PATH)

	aapl = SDL.load1min("aapl", "2024-01-01", "2024-01-31")
	print(aapl.head())
	print(aapl.tail())
	print(f"\nShape: {aapl.shape}")
	print(f"Date range: {aapl.index[0]} -> {aapl.index[-1]}")
	print(f"Memory Usage: {aapl.memory_usage(deep=True).sum() / 1024**2:.1f} MB")

if __name__ == "__main__":
	main()

	
