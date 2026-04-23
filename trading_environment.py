#trading_environment.py
#necessary components for the simulated trading environment

from collections import defaultdict
import datetime
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

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
	

			