import torch
import torch.nn as nn
from challenger_model import ChallengerAgent
from trading_environment import Portfolio
from sequence_generator import SequenceGenerator

def train_agent():
    # 1. INITIALIZE COMPONENTS
    # 9 features (OHLCV, MACD, RSI, Sentiment, Buzz)
    model = ChallengerAgent(input_dim=9, hidden_dim=128, num_stocks=30)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Initialize the "Challenger" Portfolio
    portfolio = Portfolio(initial_cash=10000.00)
    
    # 2. THE TRAINING LOOP (The "School" years)
    num_epochs = 10 # How many times to repeat the whole history
    
    for epoch in range(num_epochs):
        print(f"Starting Epoch {epoch+1}")
        
        # Reset portfolio for a fresh start each epoch
        portfolio = Portfolio(initial_cash=10000.00)
        
        # 3. THE "STEP" LOOP (Minute-by-minute learning)
        # Assuming you have your sequences ready from the SequenceGenerator
        for i in range(len(sequences)):
            # Convert sequence to PyTorch tensor
            state = torch.FloatTensor(sequences[i]).unsqueeze(0) # [1, 60, 9]
            
            # THE BRAIN MAKES A CHOICE
            action_weights = model(state) # Output: 30 weights adding to 1.0
            
            # THE AGENT EXECUTES THE CHOICE
            # Loop through the 30 stocks and tell the portfolio to buy/sell 
            # based on the weights and the current prices
            current_prices = get_prices_at_minute(i) 
            
            # (Self-Correction: Here is where the Agent 'talks' to the Portfolio)
            execute_trades(portfolio, action_weights, current_prices)
            
            # 4. CALCULATE THE "GRADE" (Loss/Reward)
            # We use the Labeler's output as the 'Correct Answer'
            target_label = torch.FloatTensor([labels[i]])
            loss = F.mse_loss(action_weights, target_label) 
            
            # 5. BACKPROPAGATION (The actual learning)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        print(f"Epoch {epoch+1} complete. Portfolio Value: {portfolio.get_current_value(current_prices)}")

if __name__ == "__main__":
    train_agent()