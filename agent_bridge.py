import torch
import datetime

class AgentBridge:
	def __init__(self, tickers, trade_penalty=0.001, confidence_threshold=0.05):
		self.tickers = tickers
		self.trade_penalty = trade_penalty # 0.1% slippage/fee simulation
		self.confidence_threshold = confidence_threshold

	def execute_allocation(self, portfolio, weights, current_prices, dt):
		"""
		Translates model weights (0.0 to 1.0) into Portfolio buy/sell actions.
        
		weights: PyTorch tensor or list of 30 values from the ChallengerNet
		current_prices: Dict of {ticker: price} at the current timestamp
		dt: current datetime.datetime object
		"""
		if weights is None:
			return
			
		# Confidence Check

		if torch.is_tensor(weights):
			max_val, _ = torch.max(weights.view(-1), dim=0)
			confidence = max_val.item()
		else:
			confidence = max(weights)

		# If confidence is very low, skip trading to avoid noise
		if confidence < self.confidence_threshold:
			return
		

		# Calculate Total Net Worth (Cash + Market Value of all stocks)
		total_value = portfolio.get_current_value(current_prices)
		
		# Get the value of SPY holdings to exclude from rebalancing
		spy_shares = portfolio.get_total_shares('SPY')
		spy_value = spy_shares * current_prices.get('SPY', 0.0)
		rebalance_value = total_value - spy_value  # Only rebalance the non-SPY portion

		# Determine Target Dollars for each stock
		# We assume weights[i] is the % of total_value we want in ticker[i]
		allocations = {}
		w = 0.0
		for i, ticker in enumerate(self.tickers):
			if torch.is_tensor(weights):
				# weights[0][i] if it's a batch tensor, otherwise weights[i]
				w = weights[0][i].item() if torch.is_tensor(weights) else weights[i]
				
			allocations[ticker] = rebalance_value * w
			
		# Sell First (To free up cash)
		# It is standard practice to sell before buying so the 'buy' checks don't fail for lack of cash
		for ticker in self.tickers:
			if ticker == 'SPY':
				continue  # Skip SPY for rebalancing
			target_dollars = allocations[ticker]
			current_shares = portfolio.get_total_shares(ticker)
			current_dollars = current_shares * current_prices.get(ticker, 0.0)
            
			if current_dollars > target_dollars +1.0:
				diff_dollars = current_dollars - target_dollars
				shares_to_sell = diff_dollars / current_prices[ticker]
				portfolio.sell(dt, ticker, shares_to_sell, current_prices[ticker])
				
		# Buy Second (Using the freed-up cash)
		for ticker in self.tickers:
			if ticker == 'SPY':
				continue  # Skip SPY for rebalancing
			target_dollars = allocations[ticker]
			current_shares = portfolio.get_total_shares(ticker)
			current_dollars = current_shares * current_prices.get(ticker, 0.0)
            
			if target_dollars > current_dollars + 1.0:
				diff_dollars = target_dollars - current_dollars
				shares_to_buy = diff_dollars / current_prices[ticker]
				portfolio.buy(dt, ticker, shares_to_buy, current_prices[ticker])
				
	def get_control_weights(self):
		"""Returns equal weights (1/30) for the Secondary Control Agent"""
		val = 1.0 / len(self.tickers)
		return [val] * len(self.tickers)

	def buy_spy_on_payday(self, portfolio, current_prices, dt):
		'''Pure SPY buy-and-hold for Control Agent'''
		if 'SPY' not in current_prices or current_prices['SPY'] <= 0:
			print("   Warning: No SPY price available")
			return False
	
		spy_price = current_prices['SPY']
		cash_available = portfolio.cash
	
		if cash_available > 50:   # small threshold to avoid tiny buys
			shares = cash_available / spy_price
			success = portfolio.buy(dt, 'SPY', shares, spy_price)
			if success:
				print(f"Control Bought {shares:.2f} SPY @ ${spy_price:.2f} | Cash left: ${portfolio.cash:.2f}")
				return True
		return False
