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
import math


#the funciton of this code is to find the logicpl and compile it with machine learning model

startdate = datetime.datetime(2024, 4, 22)
enddate = datetime.datetime(2024, 6, 20)
stocks=['NVDA']

#msft,nvda,adani,aapl-winning


class StrategyName:
    def __init__(self):
        self.data = None
        self.split = None  
    
    def FetchData(self, ticker):
        self.data = yf.download(ticker, start=startdate, end=enddate, interval='15m')
        self.data.drop('Adj Close', axis=1, inplace=True)
        self.data['returns'] = self.data['Close'].pct_change()
        return self.data

    def Functions(self):
        self.data['sma160'] = talib.SMA(self.data['Close'], timeperiod=160)
        self.data['sma80'] = talib.SMA(self.data['Close'], timeperiod=80)
        self.data['sma7'] = talib.SMA(self.data['Close'], timeperiod=7)
        self.data['ADX'] = talib.ADX(self.data['High'], self.data['Low'], self.data['Close'], timeperiod=14)
        self.data['ATR'] =talib.ATR(self.data['High'],self.data['Low'],self.data['Close'] ,timeperiod=10)
        self.data['Momentum'] = talib.MOM(self.data['Close'], timeperiod=14)
        self.data['rsi'] = talib.RSI(self.data['Close'], timeperiod=14)
        self.data['Donchian'] = talib.MAX(self.data['High'], timeperiod=20) - talib.MIN(self.data['Low'], timeperiod=20)
        self.data['DonchianChange'] = self.data['Donchian'].pct_change()
        self.data['Trend_prediction'] = np.where(self.data['DonchianChange'] > 0, 1, np.where(self.data['DonchianChange'] < 0, -1, 0))
        
        self.data = self.data.dropna()
        return self.data
    
    def Training(self):
        self.split = len(self.data) * 0.8
        # reg = GradientBoostingRegressor(n_estimators=300, learning_rate=0.3, max_depth=15, random_state=123)
        """reg = GradientBoostingRegressor()"""#`10857`
        """reg = LinearRegression()""" #10500
        reg = DecisionTreeRegressor()#10857
        """reg = RandomForestRegressor()"""#10857
        """reg = SVR()"""#10150
        """reg = MLPRegressor()"""#10758
        """reg = KNeighborsRegressor()"""#11400
        """reg = BayesianRidge()"""

        x_train = self.data[['sma7', 'ADX', 'Momentum']].iloc[:int(self.split)]
        y_train = self.data['returns'].iloc[:int(self.split)]

        x_train.dropna(inplace=True)
        y_train.dropna(inplace=True)
        
        reg.fit(x_train, y_train)
        
        self.data['predict'] = reg.predict(self.data[['sma7', 'ADX', 'Momentum']])
        self.data['position_gb'] = np.sign(self.data['predict'].shift(1))
        self.data['strategy_gb'] = self.data['returns'] * self.data['position_gb']
        
        self.data['predict'].plot(label='Gradient Boosting Regressor')
        plt.legend()
        plt.title('Stock Return Predictions')
        plt.xlabel('Date')
        plt.ylabel('Returns')
        plt.show()
        
        (self.data['strategy_gb'].iloc[int(self.split):].cumsum() * 100).plot(label='Gradient Boosting Strategy')
        plt.legend()
        plt.title('Cumulative Strategy Returns')
        plt.xlabel('Date')
        plt.ylabel('Cumulative Returns (%)')
        plt.show()

    def calculate_performance(self, position_column):    
        no_of_wins = 0
        no_of_losses = 0
        no_of_noreturn = 0

        for i in range(int(self.split), int(len(self.data)) - 1):
            position = self.data[position_column].iloc[i]
            next_return = self.data['returns'].iloc[i + 1]

            if position == 1:
                if next_return > 0:
                    no_of_wins += 1
                else:
                    no_of_losses += 1
            elif position == -1:
                if next_return < 0:
                    no_of_wins += 1
                else:
                    no_of_losses += 1
            else:
                no_of_noreturn += 1

        total_trades = no_of_wins + no_of_losses
        win_rate = (no_of_wins / total_trades) * 100 if total_trades > 0 else 0
        error_rate = (no_of_losses / total_trades) * 100 if total_trades > 0 else 0

        return no_of_wins, no_of_losses, no_of_noreturn, win_rate, error_rate

    def Signal(self):
        self.data.loc[:, 'Signal'] = 0  
        for i in range(int(self.split), int(len(self.data))):
            if (((self.data['sma7'].iloc[i] - self.data['sma7'].iloc[i - 20]) / 20) > 0) and (
                    ((self.data['ADX'].iloc[i] - self.data['ADX'].iloc[i - 20]) / 20) > 0) and (
                    self.data['Momentum'].iloc[i] > 0) and self.data['position_gb'].iloc[i] == 1:
                self.data['Signal'].iloc[i] = 1
            elif (((self.data['sma7'].iloc[i] - self.data['sma7'].iloc[i - 20]) / 20) < 0) and (
                    ((self.data['ADX'].iloc[i] - self.data['ADX'].iloc[i - 20]) / 20) < 0) and (
                    self.data['Momentum'].iloc[i] < 0)  == -1:
                self.data['Signal'].iloc[i] = -1
        return self.data

    def ExecuteTrades(self):
        pass

    def LogicPl(self):
        self.capital=10000
        positions = []
        final_capital = self.capital
        for i in range(int(len(self.data))):
            if self.data['Signal'].iloc[i] == 1:
                entry_price = self.data['Close'].iloc[i]
                quantity = math.floor(self.capital / entry_price)
                take_profit = self.data['Close'].iloc[i]+self.data['ATR'].iloc[i]*2
                stop_loss = self.data['Close'].iloc[i]-self.data['ATR'].iloc[i]*1
                for j in range(i, int(len(self.data))):
                    if self.data['Close'].iloc[j] >= take_profit:
                        exit_price = take_profit
                        pl = (exit_price - entry_price) * quantity
                        final_capital += pl
                        positions.append({
                            'EntryPrice': entry_price,
                            'ExitPrice': exit_price,
                            'PnL': pl,
                            'Quantity': quantity
                        })
                        break
                    elif self.data['Close'].iloc[j] <= stop_loss:
                        exit_price = stop_loss
                        pl = (exit_price - entry_price) * quantity
                        final_capital += pl
                        positions.append({
                            'EntryPrice': entry_price,
                            'ExitPrice': exit_price,
                            'PnL': pl,
                            'Quantity': quantity
                        })
                        break
        print(positions)
        positions = pd.DataFrame(positions)
        positions.to_csv('file.csv', index=False)
        print(final_capital)

    def PlotGraph(self,ticker):
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 5), sharex=True)

        ax1.plot(self.data['Close'], label='Close Price')
        ax1.plot(self.data['sma7'], label='SMA 7')
        
        
        ax1.set_title(f'{ticker} Stock Price and Indicators')
        ax1.legend(loc='upper left')

        ax2.plot(self.data['predict'], label='trendprediction')
        ax2.plot(self.data['Momentum'], label='Momentum')
        ax2.plot(self.data['ATR'], label='ATR')
        ax2.set_title(f'{ticker} Stock Predictions')
        ax2.legend(loc='upper left')
        
        ax3.plot(self.data['ADX'], label='ADX')
        ax3.set_title(f'{ticker}, Stock ADX')
        ax3.legend(loc='upper left')

        plt.show()     


def main():
    for stock in stocks:
        session = StrategyName()
        session.FetchData(stock)
        session.Functions()
        session.Training()  
        session.Signal()
        session.LogicPl()
        session.PlotGraph(stock)
            
        session.data.to_csv(os.path.join('data', f'file_{stock}.csv'))
        print(session.data)
        
        performance_gb = session.calculate_performance("position_gb")
        print(f'Gradient Boosting Model:\nNumber of wins: {performance_gb[0]}\nNumber of losses: {performance_gb[1]}\nNumber of no returns: {performance_gb[2]}\nWin rate: {performance_gb[3]:.2f}%\nError rate: {performance_gb[4]:.2f}%\n')
        session.data.to_csv(os.path.join('data', 'file.csv'))

main()
