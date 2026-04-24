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

# --- CONFIGURATION SWITCH ---
# Set this to True to train the brain on 2024 data
# Set this to False to test the brain on 2025 data
TRAINING_MODE = True 

def align_all_dataframes(master_data: dict, timeline):
    '''Make sure every ticker has the same index as the main timeline'''
    aligned = {}
    for ticker, df in master_data.items():
        df = df.reindex(timeline, method='ffill')
        df = df.fillna(method='bfill')
        aligned[ticker] = df
    return aligned
    
def main():
    load_dotenv()
    
    # 1. SETUP CONSTANTS
    TICKERS = [
        'NVDA', 'AAPL', 'MSFT', 'AMZN', 'WMT', 'JPM', 'V', 'JNJ', 'CAT', 'CVX', 
        'CSCO', 'PG', 'HD', 'KO', 'UNH', 'MRK', 'GS', 'AXP', 'MCD', 'IBM', 
        'VZ', 'AMGN', 'DIS', 'BA', 'CRM', 'HON', 'SHW', 'MMM', 'NKE', 'TRV'
    ]
    FEATURES = ['open', 'high', 'low', 'close', 'volume', 'macd', 'rsi_20', 'news_sentiment', 'news_buzz']
    
    DATA_PATH = "../OHLC 1 minute data/extracted_files"
    
    # --- DATE LOGIC BASED ON MODE ---
    if TRAINING_MODE:
        START_DATE = "2024-01-01"
        END_DATE = "2024-12-31"
        print(f"MODE: TRAINING (Studying {START_DATE} to {END_DATE})")
    else:
        START_DATE = "2025-01-01"
        END_DATE = "2025-03-31"
        print(f"MODE: BACKTEST (Testing {START_DATE} to {END_DATE})")
    
    # 2. INITIALIZE ENGINE COMPONENTS
    sdl = StockDataLoader(base_path=DATA_PATH)
    sl = SentimentLoader()
    rf = RollingFeatures()
    bridge = AgentBridge(tickers=TICKERS)
    
    model = ChallengerAgent(input_dim=9, hidden_dim=128, num_stocks=len(TICKERS))
    
    # Only load the brain if we are in Backtest mode
    if not TRAINING_MODE:
        try:
            model.load_state_dict(torch.load("challenger_model.pth"))
            model.eval()
            print("SUCCESS: Trained brain 'challenger_model.pth' loaded.")
        except FileNotFoundError:
            print("WARNING: No trained brain found. The Challenger will use random weights.")

    # 3. DATA PREP LOOP
    master_data = {}
    print(f"--- Step 1: Processing Data for {len(TICKERS)} stocks ---")

    for ticker in TICKERS:
        try:
            mfe = MultiTimeFrameFeatures(sdl)
            multi_tf = mfe.create_features(ticker, start=START_DATE, end=END_DATE)
            df_5m = sl.add_sentiment_to_df(multi_tf['5min'], ticker)
            processed_df = rf.process({'5min': df_5m})['5min']
            master_data[ticker] = processed_df
            print(f"Loaded {ticker}")
        except Exception as e:
            print(f"Skipping {ticker} due to error: {e}")

    # --- 4. EXECUTION BRANCH ---
    if TRAINING_MODE:
        print(f"\n--- Step 2: Handoff to Training Loop ---")
        from train import train_agent
        train_agent(master_data)
    else:
        print(f"\n--- Step 2: Starting Agent Showdown (Backtest) ---")
        
        if not master_data or 'AAPL' not in master_data:
            print("Error: No data loaded.")
            return

        timeline = master_data['AAPL'].index
        master_data = align_all_dataframes(master_data, timeline)
        
        challenger_portfolio = Portfolio(initial_cash=10000.00)
        control_portfolio = Portfolio(initial_cash=10000.00)
        current_prices = {t: 0.0 for t in TICKERS}

        for dt in timeline:
            current_prices = {t: master_data[t].loc[dt, 'close'] for t in master_data}
            
            if dt.weekday() == 4 and dt.hour == 15 and dt.minute == 55:
                print(f"[{dt.date()}] Payday! +$100")
                challenger_portfolio.payday(100)
                control_portfolio.payday(100)

            current_idx = master_data['AAPL'].index.get_loc(dt)
            if current_idx >= 60:
                window_data = master_data['AAPL'].iloc[current_idx-60 : current_idx][FEATURES].values
                features_tensor = torch.FloatTensor(window_data).unsqueeze(0) 
                with torch.no_grad():
                    challenger_weights = model(features_tensor)
                bridge.execute_allocation(challenger_portfolio, challenger_weights, current_prices, dt)

            control_weights = bridge.get_control_weights()
            bridge.execute_allocation(control_portfolio, control_weights, current_prices, dt)

        # FINAL RESULTS (Only for Backtest)
        final_c = challenger_portfolio.get_current_value(current_prices)
        final_s = control_portfolio.get_current_value(current_prices)
        print("\n" + "="*40)
        print(f"FINAL RESULTS ({END_DATE})")
        print(f"Challenger Portfolio: ${final_c:,.2f}")
        print(f"Secondary Control:    ${final_s:,.2f}")
        if final_s > 0:
            delta = ((final_c / final_s) - 1) * 100
            print(f"Performance Delta:    {delta:.2f}%")
        print("="*40)

if __name__ == "__main__":
    main()