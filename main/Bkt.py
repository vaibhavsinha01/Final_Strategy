import pandas as pd
import numpy as np
import yfinance as yf
import talib
from backtesting import Backtest, Strategy
import datetime
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import BayesianRidge
import matplotlib.pyplot as plt

# The function of this code is to check the backtesting results for the code
startdate = datetime.datetime(2024, 4, 30)
enddate = datetime.datetime(2024, 6, 27)
# startdate=datetime.datetime(2018,1,1)
# enddate=datetime.datetime(2021,1,1)
# startdate = datetime.datetime(2024, 6, 17)
# enddate = datetime.datetime(2024, 6, 24)

def optimfunc(series):
    if series['# Trades'] < 10:
        return -1
    else:
        return ['Final Capital [$]']

def Fetchdata(stock):
    data = yf.download(stock, start=startdate, end=enddate, interval='15m')
    """data=yf.download(stock,start=startdate,end=enddate)"""
    """data = yf.download(stock, start=startdate, end=enddate, interval='1m')"""
    data.drop('Adj Close', axis=1, inplace=True)
    return data

class ADXMOMENTUM(Strategy):
    """stlo=97
    tkpr=103"""
    multiplier1 = 200
    multiplier2 = 200
    

    def init(self):
        self.sma7 = self.I(talib.SMA, self.data.Close, timeperiod=7)
        self.sma80 = self.I(talib.SMA, self.data.Close, timeperiod=80)
        self.sma160 = self.I(talib.SMA, self.data.Close, timeperiod=160)
        self.dema160 = self.I(talib.DEMA, self.data.Close, timeperiod=160)
        self.pp =(self.data.High+self.data.Low+self.data.Close)/3
        self.r1 = 2*self.pp - self.data.Low
        self.r2 = self.pp + self.data.High - self.data.Low
        self.r3 = self.data.High + 2 * (self.pp  - self.data.Low)
        self.s1 = 2*self.pp - self.data.High
        self.s2 = self.pp - (self.data.High - self.data.Low)
        self.s3 = self.data.Low - 2 * (self.data.High  - self.pp)
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, timeperiod=10)
        self.adx = self.I(talib.ADX, self.data.High, self.data.Low, self.data.Close, timeperiod=14)
        self.MOM = self.I(talib.MOM, self.data.Close, timeperiod=14)
        self.rsi = self.I(talib.RSI, self.data.Close, timeperiod=14)
        self.donchian = self.I(talib.MAX, self.data.High, timeperiod=20) - self.I(talib.MIN, self.data.Low, timeperiod=20)
        self.Train()  # Call the Train method

    def Train(self):
        # Create a DataFrame with necessary data
        close_series = pd.Series(self.data.Close, index=self.data.index)
        self.indicator_df = pd.DataFrame(index=self.data.index)
        self.indicator_df['sma7'] = self.sma7
        self.indicator_df['ADX'] = self.adx
        self.indicator_df['MOM'] = self.MOM
        self.indicator_df['dema160'] = self.dema160

        # Calculate returns
        self.indicator_df['returns'] = close_series.pct_change(1)
        
        # Drop rows with NaN values
        self.indicator_df.dropna(inplace=True)

        # Split the data into training and testing sets
        split = int(len(self.indicator_df) * 0.8)
        
        # Training data
        x_train = self.indicator_df[['sma7', 'ADX', 'MOM', 'dema160']].iloc[:split]
        y_train = self.indicator_df['returns'].iloc[:split]
        
        # Create and train the linear regression model
        # reg = LinearRegression() #111
        # reg = DecisionTreeRegressor() #111
        # reg = RandomForestRegressor() #108
        # reg = GradientBoostingRegressor() #108
        # reg = SVR() #108
        reg = KNeighborsRegressor() #108
        # reg = MLPRegressor() #none
        # reg = BayesianRidge() #111

        reg.fit(x_train, y_train)
        
        # Predict on the entire dataset
        self.indicator_df['predict'] = reg.predict(self.indicator_df[['sma7', 'ADX', 'MOM', 'dema160']])
        self.indicator_df["position_reg"] = np.sign(self.indicator_df["predict"])

    def next(self):
        if (((self.sma7[-1] - self.sma7[-11]) / 20) > 0) and (((self.adx[-1] - self.adx[-11]) / 20) > 0) and (self.MOM > 0) and (self.dema160 < self.data.Close) and (self.indicator_df['position_reg'].iloc[-1] == -1):
            self.position.close()
            self.buy(sl=self.data.Close - self.atr * (self.multiplier2 / 100), tp=self.data.Close + self.atr * (self.multiplier1 / 100))
            # self.buy(sl=self.s3*0.9, tp=self.r1*1.1)

        elif (((self.sma7[-1] - self.sma7[-11]) / 20) < 0) and (((self.adx[-1] - self.adx[-11]) / 20) < 0) and (self.MOM < 0) and (self.dema160 > self.data.Close) and (self.indicator_df['position_reg'].iloc[-1] == -1):
            self.position.close()
            self.sell(tp=self.data.Close - self.atr * (self.multiplier2 / 100), sl=self.data.Close + self.atr * (self.multiplier1 / 100))
            # self.sell(sl=self.r1*1.1, tp=self.s3*0.9)

def main():
    data = Fetchdata('googl')
    bt = Backtest(data, ADXMOMENTUM, cash=10000)
    bt.run()
    st=bt.optimize(
        multiplier1=range(100, 500, 100),
        multiplier2=range(100, 500, 100),
        maximize=optimfunc,
        max_tries=1000
    )
    print(st)
    bt.plot()

main()
