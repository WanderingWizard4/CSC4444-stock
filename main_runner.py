import pandas as pd
import torch
import os
import numpy as np
import multiprocessing as mp
import platform
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
from datetime import datetime
from dotenv import load_dotenv
from stock_data_loader import StockDataLoader
from feature_engineering import MultiTimeFrameFeatures
from rolling_features import RollingFeatures
from sentiment_loader import SentimentLoader
from labeler import TripleBarrierLabeler
from challenger_model import ChallengerAgent
from trading_environment import Portfolio
from agent_bridge import AgentBridge
from sequence_generator import SequenceGenerator


#============ CONFIG==================
TRAINING_MODE = True   #True = train  | False = backtest
REBALANCE_EVERY = 12   #12 *5min = hourly rebalancing
WINDOW_SIZE = 60
#======================================

#Multiprocessing Wizardry
if platform.system() == "Darwin": #macOS
	mp.set_start_method('fork', force=True)
elif platform.system() == "Windows":
	mp.set_start_method('spawn', force=True)   #windows
else:
	pass
pd.options.mode.chained_assignment = None #supress pandas warnings


def process_single_ticker(ticker:str, start_date:str, end_date: str, data_path: str):
	'''Process one ticker completely- meant for parallel runing'''
	try:
		print(f" Processing {ticker}...")
		sdl = StockDataLoader(base_path=data_path)
		mfe = MultiTimeFrameFeatures(sdl)
		sl = SentimentLoader()
		rf = RollingFeatures()
		tbl = TripleBarrierLabeler(pt_multiplier=1.5, sl_multiplier=1.0, vertical_barrier_mins=60)
		
		# load and resample
		multi_tf = mfe.create_features(ticker, start = start_date, end = end_date)
		if not multi_tf or '5min' not in multi_tf:
			print(f" no 5min data for {ticker}")
			return ticker, None

		#add sentiment
		df_5m = sl.add_sentiment_to_df(multi_tf['5min'], ticker)

		#rolling features
		processed = rf.process({'5min': df_5m})
		df_processed = processed['5min']

		#label 
		labeled_df = tbl.label_data(df_processed)

		print(f" {ticker} done - {len(labeled_df):,} bars, labels: {labeled_df['label'].value_counts().to_dict()}")
		return ticker, labeled_df
		
	except Exception as e:
		print(f" Error processing {ticker}: {e}")
		return ticker, None
		
def normalize_features(tensor: torch.Tensor) -> torch.Tensor:
	'''
	Z-score normalization across feature dimension
	Helps LSTM/GRU handle vastly different scales of features (price vs voume, vs RSI etc)
	'''
	mean = tensor.mean(dim=(0,1), keepdim=True)
	std = tensor.std(dim=(0,1), keepdim=True) + 1e-8 #avoid division by zero
	return (tensor - mean) / std
	
def get_current_sequence(master_data: dict, timeline_idx: int, seq_gen: SequenceGenerator, window_size=60):
	'''
	Extract most recent window of features for all tickers @ current timestep
	Return tensor of shape[1, window_size, num_features * num_tickers]
	'''

	features_list = []
	for ticker in master_data.keys():
		df = master_data[ticker]
		if len(df) <= timeline_idx or timeline_idx <0:
			#Pad with zeros to start
			window = pd.DataFrame(0, index=range(window_size), columns = seq_gen.feature_cols)
		else:
			start_idx = max(0, timeline_idx - window_size + 1)
			window_df = df.iloc[start_idx:timeline_idx +1][seq_gen.feature_cols].copy()

			#pad front if not enough history
			if len(window_df) < window_size:
				pad = pd.DataFrame(0, index=range(window_size - len(window_df)), columns=seq_gen.feature_cols)

				window_df = pd.concat([pad, window_df], ignore_index=True)
				
			window = window_df

		features_list.append(window.values)

	#stack all tickers   -> shape[num_tickers, window_size, num_features]
	stacked = np.stack(features_list, axis=0)

	batch_tensor = torch.FloatTensor(stacked).permute(1,0,2).reshape(1, window_size, -1)

	return batch_tensor

def align_all_dataframes(master_data: dict, timeline):
	'''Make sure every ticker has the same index as the main timeline'''
	aligned = {}
	for ticker, df in master_data.items():
		if df.empty:
			aligned[ticker] = df
			continue
		df_aligned = df.reindex(timeline, method='ffill').bfill()
		aligned[ticker] = df_aligned
	return aligned
	


def align_all_dataframes(master_data: dict, timeline):
	'''Make sure every ticker has the same index as the main timeline'''
	aligned = {}
	for ticker, df in master_data.items():
		if df.empty:
			aligned[ticker]=df
			continue
		df_aligned = df.reindex(timeline, method='ffill').bfill()
		aligned[ticker] = df
	return aligned
    
def main():
	load_dotenv()
	
	# Setup Constants tickers from DJIA + SPY
	TICKERS = [
		'NVDA', 'AAPL', 'MSFT', 'AMZN', 'WMT', 'JPM', 'V', 'JNJ', 'CAT', 'CVX', 
		'CSCO', 'PG', 'HD', 'KO', 'UNH', 'MRK', 'GS', 'AXP', 'MCD', 'IBM', 
		'VZ', 'AMGN', 'DIS', 'BA', 'CRM', 'HON', 'SHW', 'MMM', 'NKE', 'TRV', 'SPY'
	]
	DATA_PATH = "../OHLC 1 minute data/extracted_files"
	
	# --- Date Logic Based on Mode ---
	if TRAINING_MODE:
		START_DATE = "2018-01-01"
		END_DATE = "2023-12-31"
		print(f"MODE: TRAINING (Studying {START_DATE} to {END_DATE})")
	else:
		START_DATE = "2024-01-01"
		END_DATE = "2026-03-31"
		print(f"MODE: BACKTEST (Testing {START_DATE} to {END_DATE})")

	# Data Prep Loop (The Pipeline)
	master_data = {}
	num_workers = min(12, mp.cpu_count())
	
	print(f"--- Step 1: Processing Data for {len(TICKERS)} stocks in parallel ---")

	print("\n---Generateing Triple Barrier Labels---")
	with ProcessPoolExecutor(max_workers=num_workers) as executor:
		future_to_ticker = {
			executor.submit(process_single_ticker, ticker, START_DATE, END_DATE, DATA_PATH): ticker for ticker in TICKERS
		}
		for future in as_completed(future_to_ticker):
			ticker, df = future.result()
			if df is not None and not df.empty:
				master_data[ticker] = df
				print(f" Loaded {ticker} into master_data")
			else:
				print(f" skipped{ticker}")
	print(f"\nSuccessfully processed {len(master_data)} / {len(TICKERS)} tickers")

	# Simulator / Backtest
	# Check if data actually loaded before continuing
	if not master_data:
		print("Error: No data loaded. womp womp...")
		return

	# We use the index of the first ticker as our 'clock'
	print("Aligning data and preparing model...")
	timeline = list(master_data.values())[0].index
	master_data = align_all_dataframes(master_data, timeline)

	seq_gen = SequenceGenerator(window_size=WINDOW_SIZE)
	input_dim = seq_gen.get_feature_count() * len(TICKERS)

	#create model with correct input dimension
	model = ChallengerAgent(input_dim=input_dim, hidden_dim=128, num_stocks=len(TICKERS))
	print(f"Model initialized with {input_dim} features (multi-ticker)")
	model_path = "challenger_model.pth"
	if os.path.exists(model_path):
		model.load_state_dict(torch.load(model_path, map_location = 'cpu'))
		print(f"Successfully loaded trained weights from {model_path}")
	else:
		print(f"Warning: {model_path} not found - using untrained model")
	
	bridge = AgentBridge(tickers=TICKERS)
	
    
	# Initialize variables
	current_prices = {t: 0.0 for t in TICKERS}

	if TRAINING_MODE:
		print(f"\n---Starting Training Phase---")
		from train_parallel import train_agent
		train_agent(master_data, num_epochs=15, batch_size=64)
	else:
		print(f"\n--- Step 2: Starting Agent Showdown ---")
		challenger_portfolio = Portfolio(initial_cash=10000.00)
		control_portfolio = Portfolio(initial_cash=10000.00)
		
		for i, dt in enumerate(timeline):
			#Update current prices for all tickers
			current_prices = {}
			for t in TICKERS:
				df_t = master_data[t]
				if df_t.empty:
					print (f"Warning: {t} has no data - skipping")
					current_prices[t] = 0.0
					continue
					
				if dt in df_t.index:
					price = df_t.loc[dt, 'close']
				else:
					#if missing use last known price
					price = df_t['close'].asof(dt)
				if pd.isna(price) or price <= 0:
					#use last known price for this ticker
					price = df_t['close'].iloc[-1]

				current_prices[t] = float(price)
					
	        
			#Check for 'Payday' (Every Friday)
			if dt.weekday() == 4 and dt.hour == 15 and dt.minute == 55:
				print(f"Payday! Adding $1000 to both agents at {dt}")
				challenger_portfolio.payday(1000)
				control_portfolio.payday(1000)
				#Control immediately invests cash in SPY
				bridge.buy_spy_on_payday(control_portfolio, current_prices, dt)
	
			#Challenger Agent Decision
			if i% REBALANCE_EVERY == 0:
				features = get_current_sequence(master_data, i, seq_gen, WINDOW_SIZE)
				features = normalize_features(features)
				with torch.no_grad():
					challenger_weights = model(features)
			else:
				challenger_weights = None
			
			bridge.execute_allocation(challenger_portfolio, challenger_weights, current_prices, dt)
			control_weights = bridge.get_control_weights()
			bridge.execute_allocation(control_portfolio, control_weights, current_prices, dt)
		
			#record Equity (for pretty pictures for report)
			challenger_portfolio.record_equity(dt, challenger_portfolio.get_current_value(current_prices))
			control_portfolio.record_equity(dt, control_portfolio.get_current_value(current_prices))
		
		# final results (Only for Backtest)
		final_c = challenger_portfolio.get_current_value(current_prices)
		final_s = control_portfolio.get_current_value(current_prices)
		
		print("\n" + "="*60)
		print(f"FINAL RESULTS ({END_DATE})")
		print(f"Challenger Portfolio: ${final_c:,.2f}")
		print(f"Secondary Control:    ${final_s:,.2f}")
		if final_s > 0:
			delta = ((final_c / final_s) - 1) * 100
			print(f"Performance Delta:    {delta:.2f}%")
		print("="*60)

		#save outputs
		challenger_portfolio.save_equity_curve("challenger_equity.csv")
		challenger_portfolio.save_final_portfolio("challenger_final.csv", current_prices)
		challenger_portfolio.save_trade_history("challenger_trades.csv")
		

if __name__ == "__main__":
    main()