import requests
from datetime import datetime, timedelta
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import warnings
warnings.filterwarnings("ignore")
import yfinance as yf
import talib
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import BayesianRidge

stocks = [
    'Bitcoin',
    'Ethereum',
    'Binance Coin',
    'Cardano',
    'Solana',
    'Ripple (XRP)',
    'Polkadot',
    'Dogecoin',
    'Shiba Inu',
    'Litecoin',
    'Chainlink',
    'Uniswap',
    'Stellar',
    'Polygon (MATIC)',
    'Avalanche',
    'Terra (LUNA)',
    'VeChain',
    'Tron',
    'Cosmos (ATOM)',
    'Tezos',
    'Monero',
    'Algorand',
    'Aave',
    'Elrond',
    'IOTA',
    'Filecoin'
]

def get_news(api_key, company_name, num_articles=100):
    start_date = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
    endpoint = "https://newsapi.org/v2/everything"
    params = {
        'apiKey': api_key,
        'q': company_name,
        'ortBy': 'publishedAt',
        'language': 'en',
        'pageSize': num_articles,
        'from': start_date,
        'to': datetime.now().strftime('%Y-%m-%d')
    }
    try:
        response = requests.get(endpoint, params=params)
        if response.status_code == 200:
            data = response.json()
            articles = data['articles']
            all_texts = ""
            for article in articles:
                if company_name.lower() in article['title'].lower():
                    title = article['title']
                    description = article['description']
                    url = article['url']
                    print(f"Title: {title}")
                    print(f"Description: {description}")
                    print(f"URL: {url}")
                    print("\n" + "="*50 + "\n")
                    all_texts += f"{title} {description} "
            return all_texts.strip()
        else:
            print(f"Error: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

api_key = '1f30615161cf4037b369adb5e4e0efb6'

nltk.download('vader_lexicon')
analyzer = SentimentIntensityAnalyzer()

df_sentiment = pd.DataFrame(columns=['Company', 'Compound Sentiment', 'Positive Sentiment', 'Negative Sentiment', 'Neutral Sentiment', 'Buy/Sell/Hold'])

def main():
    for company_name in stocks:
        text = get_news(api_key, company_name, num_articles=100)
        if text:
            sentiment = analyzer.polarity_scores(text)
            print(f"Text: {text}")
            print(f"The compound Sentiment for {company_name} is: {sentiment['compound']} the positive Sentiment is: {sentiment['pos']} the negative sentiment is {sentiment['neg']} and neutral sentiment is {sentiment['neu']}")
            if sentiment['pos'] > sentiment['neg']*1.5:
                print(f'Buy {company_name}')
                buy_sell_hold = 1
            elif sentiment['pos']*1.5 < sentiment['neg']:
                print(f'Sell {company_name}')
                buy_sell_hold = -1
            else:
                print(f'Hold {company_name}')
                buy_sell_hold = 0
            df_sentiment.loc[len(df_sentiment)] = [company_name, sentiment['compound'], sentiment['pos'], sentiment['neg'], sentiment['neu'], buy_sell_hold]
    print(df_sentiment)
    df_sentiment.to_csv(os.path.join('data','sentiment_analysis.csv'))

main()
