from pybit.unified_trading import HTTP
import Creds
import pandas as pd
import numpy as np
import yfinance as yf
import talib
import datetime
import os
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import BayesianRidge
import matplotlib.pyplot as plt
import time

#here in this code orders will be placed in real time

class Broker:
    def __init__(self):
        self.session = HTTP(
            testnet=True,
            api_key=Creds.KEY,
            api_secret=Creds.SECRET
        )

    def get_kline_data(self, category='spot', symbol='BTCUSDT', interval=60):
        data = self.session.get_kline(
            category=category,
            symbol=symbol,
            interval=interval
        )
        b = data['result']['list']
        df = pd.DataFrame(b, columns=['datetime', 'Open', 'High', 'Low', 'Close', 'Volume', 'tradeamount'])
        df['datetime'] = pd.to_datetime(df['datetime'], unit='ms')
        df.set_index('datetime', inplace=True)
        df = df.astype(float)
        return df

    def get_account_balance(self, accountType='UNIFIED'):
        balance = self.session.get_wallet_balance(accountType=accountType)
        print(balance)

    def place_order(self, category='spot', symbol='BTCUSDT', side='Buy', orderType='Market', qty='0.1'):
        order_response = self.session.place_order(
            category=category,
            symbol=symbol,
            side=side,
            orderType=orderType,
            qty=qty
        )
        print(order_response)
        return order_response


class ADXMOMENTUM(Broker):
    def __init__(self):
        super().__init__()
        self.data = None
        self.split = None

    def Functions(self):
        self.data['returns'] = self.data['Close'].pct_change(1)
        self.data['sma160'] = talib.SMA(self.data['Close'], timeperiod=160)
        self.data['sma80'] = talib.SMA(self.data['Close'], timeperiod=80)
        self.data['sma7'] = talib.SMA(self.data['Close'], timeperiod=7)
        self.data['ADX'] = talib.ADX(self.data['High'], self.data['Low'], self.data['Close'], timeperiod=14)
        self.data['ATR'] = talib.ATR(self.data['High'], self.data['Low'], self.data['Close'], timeperiod=10)
        self.data['Momentum'] = talib.MOM(self.data['Close'], timeperiod=14)
        self.data['rsi'] = talib.RSI(self.data['Close'], timeperiod=14)
        self.data['Donchian'] = talib.MAX(self.data['High'], timeperiod=20) - talib.MIN(self.data['Low'], timeperiod=20)
        self.data['DonchianChange'] = self.data['Donchian'].pct_change()
        self.data['Trend_prediction'] = np.where(self.data['DonchianChange'] > 0, 1, np.where(self.data['DonchianChange'] < 0, -1, 0))
        
        self.data = self.data.dropna()
        return self.data

    def Training(self):
        self.split = int(len(self.data) * 0.8)
        
        # Choose one regressor. Uncomment the one you want to use.
        reg = GradientBoostingRegressor(n_estimators=300, learning_rate=0.3, max_depth=15, random_state=123)
        # reg = LinearRegression()
        # reg = DecisionTreeRegressor()
        # reg = RandomForestRegressor()
        # reg = SVR()
        # reg = MLPRegressor()
        # reg = KNeighborsRegressor()
        # reg = BayesianRidge()

        x_train = self.data[['sma7', 'ADX', 'Momentum']].iloc[:self.split]
        y_train = self.data['returns'].iloc[:self.split]

        x_train.dropna(inplace=True)
        y_train.dropna(inplace=True)
        
        reg.fit(x_train, y_train)
        
        self.data['predict'] = reg.predict(self.data[['sma7', 'ADX', 'Momentum']])
        self.data['position_gb'] = np.sign(self.data['predict'].shift(1))
        self.data['strategy_gb'] = self.data['returns'] * self.data['position_gb']
        
        self.data['predict'].plot(label='Gradient Boosting Regressor')
        (self.data['strategy_gb'].iloc[self.split:].cumsum() * 100).plot(label='Gradient Boosting Strategy')

    def Signal(self):
        self.data.loc[:, 'Signal'] = 0
        for i in range(int(self.split), int(len(self.data))):
            if (((self.data['sma7'].iloc[i] - self.data['sma7'].iloc[i - 20]) / 20) > 0) and (
                    ((self.data['ADX'].iloc[i] - self.data['ADX'].iloc[i - 20]) / 20) > 0) and (
                    self.data['Momentum'].iloc[i] > 0) and self.data['position_gb'].iloc[i] == 1:
                self.data['Signal'].iloc[i] = 1
            elif (((self.data['sma7'].iloc[i] - self.data['sma7'].iloc[i - 20]) / 20) < 0) and (
                    ((self.data['ADX'].iloc[i] - self.data['ADX'].iloc[i - 20]) / 20) < 0) and (
                    self.data['Momentum'].iloc[i] < 0) and self.data['position_gb'].iloc[i] == -1:
                self.data['Signal'].iloc[i] = -1
        return self.data

    def ExecuteTrades(self):
        if self.data['Signal'].iloc[-1] == 1:
            print(self.data['Signal'])
            print('Placing a buy order')
            print(self.data['Signal'].iloc[-1])
            self.place_order()
        elif self.data['Signal'].iloc[-1] == -1:
            print('Placing a sell order')
            print(self.data['Signal'].iloc[-1])
            self.place_order(side='sell')
        else:
            print("No order has been placed")
            print(self.data['Signal'].iloc[-1])

def main():
    stocks = ['ETHUSDT']
    while True:
        for stock in stocks:
            session = ADXMOMENTUM()
            session.data = session.get_kline_data(symbol=stock)
            session.Functions()
            session.Training()
            session.Signal()
            session.ExecuteTrades()
            time.sleep(60)  

main()
