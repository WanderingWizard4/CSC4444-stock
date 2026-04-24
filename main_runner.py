import pandas as pd
import torch
from datetime import datetime
from dotenv import load_dotenv

# Import your group's components
from stock_data_loader import StockDataLoader
from feature_engineering import MultiTimeFrameFeatures
from rolling_features import RollingFeatures
from sentiment_loader import SentimentLoader
from labeler import TripleBarrierLabeler
from challenger_model import ChallengerAgent
from trading_environment import Portfolio
from agent_bridge import AgentBridge

def main():
    load_dotenv()
    
    # 1. SETUP CONSTANTS
    TICKERS = [
        'NVDA', 'AAPL', 'MSFT', 'AMZN', 'WMT', 'JPM', 'V', 'JNJ', 'CAT', 'CVX', 
        'CSCO', 'PG', 'HD', 'KO', 'UNH', 'MRK', 'GS', 'AXP', 'MCD', 'IBM', 
        'VZ', 'AMGN', 'DIS', 'BA', 'CRM', 'HON', 'SHW', 'MMM', 'NKE', 'TRV'
    ]
    DATA_PATH = "../OHLC 1 minute data/extracted_files"
    START_DATE = "2024-01-01"
    END_DATE = "2024-03-31" # Test window
    
    # 2. INITIALIZE ENGINE COMPONENTS
    sdl = StockDataLoader(base_path=DATA_PATH)
    sl = SentimentLoader()
    rf = RollingFeatures()
    bridge = AgentBridge(tickers=TICKERS)
    
    # Initialize Portfolios (The Showdown Contestants)
    challenger_portfolio = Portfolio(initial_cash=10000.00)
    control_portfolio = Portfolio(initial_cash=10000.00)
    
    # Initialize the Brain (9 features)
    model = ChallengerAgent(input_dim=9, hidden_dim=128, num_stocks=len(TICKERS))
    
    # 3. DATA PREP LOOP (The Pipeline)
    master_data = {}
    print(f"--- Step 1: Processing Data for {len(TICKERS)} stocks ---")
    
    for ticker in TICKERS:
        try:
            mfe = MultiTimeFrameFeatures(sdl)
            # We use 5min as our standard trading interval
            multi_tf = mfe.create_features(ticker, start=START_DATE, end=END_DATE)
            df_5m = sl.add_sentiment_to_df(multi_tf['5min'], ticker)
            processed_df = rf.process({'5min': df_5m})['5min']
            
            master_data[ticker] = processed_df
            print(f"Loaded {ticker}")
        except Exception as e:
            print(f"Skipping {ticker} due to error: {e}")

    # 4. THE SIMULATION / BACKTEST
    print(f"\n--- Step 2: Starting Agent Showdown ---")
    
    # Check if data actually loaded before continuing
    if not master_data or 'AAPL' not in master_data:
        print("Error: No data loaded. Check your data path and tickers.")
        return

    # We use the index of the first ticker as our 'clock'
    timeline = master_data['AAPL'].index
    
    # INITIALIZE variables before the loop to fix VS Code "undefined" warnings
    current_prices = {t: 0.0 for t in TICKERS}
    dt = timeline[0]

    for dt in timeline:
        # A. Update current prices for all tickers
        current_prices = {t: master_data[t].loc[dt, 'close'] for t in master_data if dt in master_data[t].index}
        
        # B. Check for 'Payday' (Every Friday)
        if dt.weekday() == 4 and dt.hour == 15 and dt.minute == 55:
            print(f"Payday! Adding $100 to both agents at {dt}")
            challenger_portfolio.payday(100)
            control_portfolio.payday(100)

        # C. CHALLENGER AGENT DECISION
        # features: Placeholder for the 60-min window sequence (Batch, Window, Features)
        features = torch.randn(1, 60, 9) 
        challenger_weights = model(features)
        bridge.execute_allocation(challenger_portfolio, challenger_weights, current_prices, dt)
        
        # D. SECONDARY CONTROL DECISION (Equal weights)
        control_weights = bridge.get_control_weights()
        bridge.execute_allocation(control_portfolio, control_weights, current_prices, dt)

    # 5. FINAL RESULTS
    final_c = challenger_portfolio.get_current_value(current_prices)
    final_s = control_portfolio.get_current_value(current_prices)
    
    print("\n" + "="*40)
    print(f"FINAL RESULTS ({END_DATE})")
    print(f"Challenger Portfolio: ${final_c:,.2f}")
    print(f"Secondary Control:    ${final_s:,.2f}")
    
    # Handle division by zero just in case
    if final_s > 0:
        delta = ((final_c / final_s) - 1) * 100
        print(f"Performance Delta:    {delta:.2f}%")
    
    print("="*40)

if __name__ == "__main__":
    main()