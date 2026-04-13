import pandas as pd
from pathlib import Path
from typing import Union, List, Dict, Optional
from datetime import datetime

class StockDataLoader:
    def __init__(self, base_path: str = "OHLC 1 minute data/extracted_files"):
        self.base_path = Path(base_path)
        if not self.base_path.exists():
            raise FileNotFoundError(f"Base path not found: {self.base_path}")
        print(f"StockDataLoader initialized with base path: {self.base_path}")

    def _get_file_path(self, year: int, month: int, ticker: str) -> Path:
        month_str = f"{month:02d}"
        return self.base_path / str(year) / f"{year}-{month_str}" / f"{ticker.upper()}.csv"

    def load1min(self, 
                 ticker: str, 
                 start: str = "1992-01-01", 
                 end: str = "2026-3-31",
                 tz: str = "US/Eastern") -> pd.DataFrame:

        """Load 1-minute OHLCV data for a single ticker."""
        start_dt = pd.to_datetime(start).tz_localize(tz)
        end_dt = pd.to_datetime(end).tz_localize(tz)
        
        data_frames = []
        current = start_dt.replace(day=1)  # Start at first day of the month
        
        while current <= end_dt:
            year = current.year
            month = current.month
            file_path = self._get_file_path(year, month, ticker)
            
            if file_path.exists():
                try:
                    df = pd.read_csv(file_path, dtype={'timestamp': 'int64'})
                    
                    # Convert Unix timestamp (seconds) to datetime
                    df['datetime'] = pd.to_datetime(df['timestamp'], unit='s', utc=True).dt.tz_convert(tz)
                    df = df.set_index('datetime')
                    df = df[['open', 'high', 'low', 'close', 'volume']]
                    
                    # Filter to requested range
                    mask = (df.index >= start_dt) & (df.index <= end_dt)
                    df = df[mask]
                    
                    if not df.empty:
                        data_frames.append(df)
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")
            # else:
            #     print(f"Missing file: {file_path}")  # uncomment for debugging
            
            # Move to next month
            if current.month == 12:
                current = current.replace(year=current.year + 1, month=1)
            else:
                current = current.replace(month=current.month + 1)
        
        if not data_frames:
            print(f"Warning: No data found for {ticker} between {start} and {end}")
            return pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume'])
        
        result = pd.concat(data_frames).sort_index()
        return result

    def load(self, 
             tickers: Union[str, List[str]], 
             start: str = "1992-01-01", 
             end: str = "2026-03-31") -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
        """Load one or multiple tickers. Returns dict if multiple."""
        if isinstance(tickers, str):
            return self.load_one(tickers, start, end)
        
        results = {}
        for t in tickers:
            df = self.load_one(t, start, end)
            if not df.empty:
                results[t.upper()] = df
            else:
                print(f"Skipped {t} - no data")
        return results
