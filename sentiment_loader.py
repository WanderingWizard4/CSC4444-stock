import os
import finnhub
from dotenv import load_dotenv
import pandas as pd

load_dotenv()

class SentimentLoader:
	def __init__(self):
		# 1. Initialize the official Finnhub client
		api_key = os.getenv('FINNHUB_API_KEY')
		self.finnhub_client = finnhub.Client(api_key=api_key)

	def fetch_sentiment(self, ticker: str):
		"""Pulls pre-calculated news sentiment using official library"""
		try:
			# 2. Call the specific sentiment endpoint
			data = self.finnhub_client.news_sentiment(ticker.upper())
            
			# Extract scores (bullishPercent is the 0-1 scale we want)
			sentiment_score = data.get('sentiment', {}).get('bullishPercent', 0.5)
			buzz_score = data.get('buzz', {}).get('articlesInLastWeek', 0)
            
			return {
				'sentiment_score': sentiment_score,
				'buzz_score': buzz_score
			}
		except Exception as e:
			print(f"Finnhub Error for {ticker}: {e}")
			return {'sentiment_score': 0.5, 'buzz_score': 0}

	def add_sentiment_to_df(self, df: pd.DataFrame, ticker: str):
		"""Broadcases the sentiment to your teammate's OHLC dataframe"""
		if df.empty:
			return df
            
		scores = self.fetch_sentiment(ticker)
		df['news_sentiment'] = scores['sentiment_score']
		df['news_buzz'] = scores['buzz_score']
        
		return df
