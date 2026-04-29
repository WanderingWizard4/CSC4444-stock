from collections import defaultdict
import datetime
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

@dataclass
class Trade:
	'''Represents a single trade'''
	datetime: datetime.datetime
	ticker: str
	quantity: float
	price: float
	buy_sell: str

class Portfolio:
	'''Portfolio tracks Cash, stock holdings, trade history, and profits and loss.'''
	def __init__(self, initial_cash: float = 10000.00):
		self.cash: float = initial_cash
		self.stock_holdings: Dict[str, List[Tuple[ datetime.datetime, float, float]]] = defaultdict(list)
		self.trade_history: List[Trade] = []
		self.profit_loss: float = 0.0

	def buy(self, dt:datetime.datetime, ticker:str, quantity: float, price:float)->bool:
		'''Buy if enough cash. Returns true on success'''
		if quantity <= 0:
			return False
		cost = quantity * price
		if self.cash < cost:
			return False
		self.cash -= cost
		self.stock_holdings[ticker].append((dt, quantity, price))
		self.trade_history.append(Trade(dt, ticker, quantity, price, 'buy'))
		return True

	def sell(self, dt:datetime.datetime, ticker: str, quantity: float, price: float)-> bool:
		'''
		-Sell min(quantity requested, shares owned)
		-simplified so that sales always succeed if you own shares (real markets work differently)
		-FIFO for realized profit and loss calculaion
		'''
		
		if quantity <= 0 or ticker not in self.stock_holdings or not self.stock_holdings[ticker]:
			return False
		shares_owned = self.get_total_shares(ticker)
		actual_sold = min(quantity, shares_owned)

		if actual_sold <= 0:
			return False

		#FIFO for accounting and calculating profit and loss
		remaining = actual_sold
		while remaining > 0 and self.stock_holdings[ticker]:
			buy_dt, lot_qty, buy_price = self.stock_holdings[ticker][0]
			sell_qty = min(remaining, lot_qty)     #can't sell more than you have

			#realized profit and loss
			pnl = sell_qty * (price - buy_price)
			self.profit_loss += pnl

			#update lot
			if sell_qty >= lot_qty:
				self.stock_holdings[ticker].pop(0)
			else:
				self.stock_holdings[ticker][0] = (buy_dt, lot_qty - sell_qty, buy_price)

			remaining -= sell_qty

		self.cash += actual_sold * price
		self.trade_history.append(Trade(dt, ticker, actual_sold, price, 'sell'))
		return True

	def payday(self, paycheck: float):
		'''Add paycheck to cash'''
		self.cash += paycheck

	def get_current_value(self, current_prices: Dict[str, float]) -> float:
		'''Total portfolio value = cash + market value of stock holdings'''
		value = self.cash
		for ticker, lots in self.stock_holdings.items():
			price = current_prices.get(ticker, 0.0)
			for _, qty, _ in lots:
				value += qty * price
		return value

	def get_position_summary(self)-> Dict[str, Dict]:
		'''For reinforcement learning State: shares owned and avg cost per ticker'''
		summary = {}
		for ticker, lots in self.stock_holdings.items():
			total_shares = sum(qty for _, qty, _ in lots)
			if total_shares > 0:
				total_cost = sum(qty * buy_price for _, qty, buy_price in lots)
				avg_cost = total_cost / total_shares
				summary[ticker] = {"shares": total_shares, "avg_cost": avg_cost}
		return summary

	def get_total_shares(self, ticker: str)-> float:
		'''calculates how many shares of a ticker owned'''
		if ticker not in self.stock_holdings:
			return 0.0
		return sum(qty for _, qty, _ in self. stock_holdings[ticker])
		
	def save_trade_history(self, filepath: str):
		'''Save complete trade history to CSV'''
		if not self.trade_history:
			print(f"No trades to save for {filepath}")
			return

		records = []
		for trade in self.trade_history:
			records.append({
				'datetime': trade.datetime,
				'ticker': trade.ticker,
				'action': trade.buy_sell,
				'quantity': trade.quantity,
				'price': trade.price,
				'trade_value': round(trade.quantity * trade.price, 2),
			})
		trades_df = pd.DataFrame(records)
		column_order = ['datetime', 'ticker', 'action', 'quantity', 'price', 'trade_value']
		trades_df = trades_df[column_order]
		trades_df.to_csv(filepath, index=False)
		print(f"Trade history saved: {os.path.basename(filepath)} ({len(trades_df)} trades)")

	# Data collection for report
	def record_equity(self, dt, current_value: float):
		'''Record portfolio value at each timestep'''
		if not hasattr(self, 'equity_curve'):
			self.equity_curve = []
		self.equity_curve.append({'datetime': dt, 'portfolio_value': current_value})

	def get_equity_curve(self) -> pd.DataFrame:
		'''Return equity curve as DataFrame'''
		if not hasattr(self, 'equity_curve'):
			return pd.DataFrame(columns=['datetime', 'portfolio_value'])
		return pd.DataFrame(self.equity_curve)

	def save_final_portfolio(self, filepath: str, current_prices: Dict[str, float]):
		'''Save final holdings and summary'''
		holdings = []
		for ticker, lots in self.stock_holdings.items():
			total_shares = sum(qty for _, qty, _ in lots)
			if total_shares > 0:
				current_price = current_prices.get(ticker, 0.0)
				if current_price == 0.0:
					print(f"   Warning: No price found for {ticker} in final portfolio save.")
				total_cost = sum(qty * buy_price for _, qty, buy_price in lots)
				market_value = total_shares * current_price
				
				holdings.append({
					'ticker': ticker, 
					'shares': round(total_shares, 6),
					'avg_cost': round(total_cost / total_shares, 4),
					'current_price': round(current_price, 4),
					'market_value': round(market_value, 2),
					'unrealized_pnl': round(market_value - total_cost, 2)
				})

		holdings_df = pd.DataFrame(holdings)
		summary = {
			'cash': round(self.cash, 2),
			'total_market_value': round(sum(h['market_value'] for h in holdings) if holdings else 0,2),
			'total_value': round(self.get_current_value(current_prices), 2),
			'realized_pnl': round(self.profit_loss, 2),
			'num_trades': len(self.trade_history)
		}

		base = filepath.replace('.csv', '')
		if not holdings_df.empty:
			holdings_df.to_csv(f"{base}_holdings.csv", index=False)
		pd.DataFrame([summary]).to_csv(f"{base}_summary.csv", index=False)

		print(f"Final portfolio summary and Holdings saved")

	def save_equity_curve(self, filepath: str):
		'''Save equity curve to CSV and plot'''
		df = self.get_equity_curve()
		if df.empty:
			print("No equity data to save")
			return

		df.to_csv(filepath, index=False)

		#make plot
		plt.figure(figsize=(12,7))
		sns.set_style("darkgrid")

		plt.plot(df['datetime'], df['portfolio_value'], linewidth=2.5, color= 'blue')
		plt.title('Portfolio Equity Curve', fontsize = 16, fontweight='bold')
		plt.xlabel('Date')
		plt.ylabel('Portfolio Value ($)')
		plt.xticks(rotation=45)


		#Annotate start and end
		start_val = df['portfolio_value'].iloc[0]
		end_val = df['portfolio_value'].iloc[-1]
		plt.annotate(f'Start: ${start_val:,.0f}',
					 xy=(df['datetime'].iloc[-1], end_val),
					 xytext=(10, -15), textcoords='offset points',
					 arrowprops=dict(arrowstyle='->', color='red'))
		plt.tight_layout()
		plot_path = filepath.replace('.csv', '.png')
		plt.savefig(plot_path, dpi=200, bbox_inches='tight')
		plt.close()

		print(f"Equity curve saved: {os.path.basename(filepath)} + plot")
				   
	