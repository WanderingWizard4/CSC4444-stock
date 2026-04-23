import torch
import datetime

class AgentBridge:
    def __init__(self, tickers, trade_penalty=0.001):
        self.tickers = tickers
        self.trade_penalty = trade_penalty # 0.1% slippage/fee simulation

    def execute_allocation(self, portfolio, weights, current_prices, dt):
        """
        Translates model weights (0.0 to 1.0) into Portfolio buy/sell actions.
        
        weights: PyTorch tensor or list of 30 values from the ChallengerNet
        current_prices: Dict of {ticker: price} at the current timestamp
        dt: current datetime.datetime object
        """
        # 1. Calculate Total Net Worth (Cash + Market Value of all stocks)
        total_value = portfolio.get_current_value(current_prices)
        
        # 2. Determine Target Dollars for each stock
        # We assume weights[i] is the % of total_value we want in ticker[i]
        allocations = {}
        for i, ticker in enumerate(self.tickers):
            # weights[0][i] if it's a batch tensor, otherwise weights[i]
            w = weights[0][i].item() if torch.is_tensor(weights) else weights[i]
            allocations[ticker] = total_value * w

        # 3. SELL FIRST (To free up cash)
        # It is standard practice to sell before buying so the 'buy' checks don't fail for lack of cash
        for ticker in self.tickers:
            target_dollars = allocations[ticker]
            current_shares = portfolio.get_total_shares(ticker)
            current_dollars = current_shares * current_prices[ticker]
            
            if current_dollars > target_dollars:
                diff_dollars = current_dollars - target_dollars
                shares_to_sell = diff_dollars / current_prices[ticker]
                
                # Small buffer: don't trade if it's less than $1.00 (saves on 'penalties')
                if diff_dollars > 1.0:
                    portfolio.sell(dt, ticker, shares_to_sell, current_prices[ticker])

        # 4. BUY SECOND (Using the freed-up cash)
        for ticker in self.tickers:
            target_dollars = allocations[ticker]
            current_shares = portfolio.get_total_shares(ticker)
            current_dollars = current_shares * current_prices[ticker]
            
            if target_dollars > current_dollars:
                diff_dollars = target_dollars - current_dollars
                shares_to_buy = diff_dollars / current_prices[ticker]
                
                if diff_dollars > 1.0:
                    # The portfolio.buy method handles cash checks automatically
                    portfolio.buy(dt, ticker, shares_to_buy, current_prices[ticker])

    def get_control_weights(self):
        """Returns equal weights (1/30) for the Secondary Control Agent"""
        val = 1.0 / len(self.tickers)
        return [val] * len(self.tickers)