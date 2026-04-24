import pandas as pd
import torch
import os
import numpy as np
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
TRAINING_MODE = True   #train on 2024 | False = backtest on 2025
REBALANCE_EVERY = 12   #12 *5min = hourly rebalancing
WINDOW_SIZE = 60
#======================================

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
		df_aligned = df.reindex(timeline, method='ffill')
		df_aligned = df_aligned.bfill()
		aligned[ticker] = df_aligned
	return aligned
	
# --- CONFIGURATION SWITCH ---
# Set this to True to train the brain on 2024 data
# Set this to False to test the brain on 2025 data
TRAINING_MODE = True 

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
	
	# 1. SETUP CONSTANTS tickers from DJIA + SPY
	TICKERS = [
		'NVDA', 'AAPL', 'MSFT', 'AMZN', 'WMT', 'JPM', 'V', 'JNJ', 'CAT', 'CVX', 
		'CSCO', 'PG', 'HD', 'KO', 'UNH', 'MRK', 'GS', 'AXP', 'MCD', 'IBM', 
		'VZ', 'AMGN', 'DIS', 'BA', 'CRM', 'HON', 'SHW', 'MMM', 'NKE', 'TRV', 'SPY'
	]
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
	
	# 3. DATA PREP LOOP (The Pipeline)
	master_data = {}
	print(f"--- Step 1: Processing Data for {len(TICKERS)} stocks ---")

	print("\n---Generateing Triple Barrier Labels---")
	
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

	tbl = TripleBarrierLabeler(pt_multiplier=1.5, sl_multiplier=1.0, vertical_barrier_mins=60)
	
	for ticker in list(master_data.keys()):
		try:
			master_data[ticker] = tbl.label_data(master_data[ticker])
			print(f"Labels added for {ticker} | Unique labels: {master_data[ticker]['label'].value_counts().to_dict()}")
		except Exception as e:
			print(f"Failed to label {ticker}: {e}")
		
	# 4. THE SIMULATION / BACKTEST
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
	
	bridge = AgentBridge(tickers=TICKERS)
	
    
	# INITIALIZE variables before the loop to fix VS Code "undefined" warnings
	current_prices = {t: 0.0 for t in TICKERS}

	if TRAINING_MODE:
		print(f"\n---Starting Training Phase---")
		from train import train_agent
		train_agent(master_data)
	else:
		print(f"\n--- Step 2: Starting Agent Showdown ---")
		challenger_portfolio = Portfolio(initial_cash=10000.00)
		control_portfolio = Portfolio(initial_cash=10000.00)
		
		for i, dt in enumerate(timeline):
			#Update current prices for all tickers
			current_prices = {t: master_data[t].loc[dt, 'close'] for t in TICKERS}
	        
			#Check for 'Payday' (Every Friday)
			if dt.weekday() == 4 and dt.hour == 15 and dt.minute == 55:
				print(f"Payday! Adding $1000 to both agents at {dt}")
				challenger_portfolio.payday(1000)
				control_portfolio.payday(1000)
	
			#CHALLENGER AGENT DECISION
			if i% REBALANCE_EVERY ==0:
				# features: Placeholder for the 60-min window sequence (Batch, Window, Features)
				features = get_current_sequence(master_data, i, seq_gen, window_size=60)
				features = normalize_features(features)
				challenger_weights = model(features)
			else:
				challenger_weights = None
			
			bridge.execute_allocation(challenger_portfolio, challenger_weights, current_prices, dt)
			control_weights = bridge.get_control_weights()
			bridge.execute_allocation(control_portfolio, control_weights, current_prices, dt)
		
			#record Equity (for pretty pictures for report)
			challenger_portfolio.record_equity(dt, challenger_portfolio.get_current_value(current_prices))
			control_portfolio.record_equity(dt, control_portfolio.get_current_value(current_prices))
		
		# 5. FINAL RESULTS (Only for Backtest)
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
	

if __name__ == "__main__":
    main()