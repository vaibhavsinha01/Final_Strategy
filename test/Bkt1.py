import pandas as pd
import numpy as np
import yfinance as yf
import talib
from backtesting import Backtest,Strategy
from backtesting.test import EURUSD,GOOG
import matplotlib.pyplot as plt
import datetime

#the function of this code is to check the backtesting results for the code

startdate=datetime.datetime(2024,4,25)
enddate=datetime.datetime(2024,6,22)

def optimfunc(series):
    if series['# Trades']<10:
        return -1
    else:
        return ['Final Capital [$]']

def Fetchdata(stock):
    data=yf.download(stock,start=startdate,end=enddate,interval='15m')
    data.drop('Adj Close',axis=1,inplace=True)
    return data

class ADXMOMENTUM(Strategy):
    """stlo=97
    tkpr=103"""
    multiplier1=200
    multiplier2=200

    def init(self):
        self.sma7=self.I(talib.SMA,self.data.Close,timeperiod=7)
        self.sma80=self.I(talib.SMA,self.data.Close,timeperiod=80)
        self.sma160=self.I(talib.SMA,self.data.Close,timeperiod=160)
        self.dema160=self.I(talib.DEMA,self.data.Close,timeperiod=160)
        self.atr=self.I(talib.ATR,self.data.High,self.data.Low,self.data.Close,timeperiod=10)
        self.adx=self.I(talib.ADX,self.data.High,self.data.Low,self.data.Close,timeperiod=14)
        self.MOM=self.I(talib.MOM,self.data.Close,timeperiod=14)
        self.rsi=self.I(talib.RSI,self.data.Close,timeperiod=14)
        self.donchian=self.I(talib.MAX,self.data.High,timeperiod=20)-self.I(talib.MIN,self.data.Low,timeperiod=20)

    def next(self):
        if (((self.sma7[-1] - self.sma7[-11]) / 20) > 0) and (((self.adx[-1] - self.adx[-11]) / 20) > 0) and (self.MOM > 0) and (self.dema160<self.data.Close):
            self.position.close()
            """self.buy(sl=(self.stlo*self.data.Close)/100,tp=(self.tkpr*self.data.Close)/100)"""
            self.buy(sl=self.data.Close-self.atr*(self.multiplier2/100),tp=self.data.Close+self.atr*(self.multiplier1/100))

        elif (((self.sma7[-1] - self.sma7[-11]) / 20) < 0) and (((self.adx[-1] - self.adx[-11]) / 20) < 0) and (self.MOM < 0) and (self.dema160>self.data.Close):
            self.position.close()
            """self.sell(tp=(self.stlo*self.data.Close)/100,sl=(self.tkpr*self.data.Close)/100)"""
            self.sell(tp=self.data.Close-self.atr*(self.multiplier2/100),sl=self.data.Close+self.atr*(self.multiplier1/100))

def main():
    data=Fetchdata('adanient.ns')
    bt=Backtest(data,ADXMOMENTUM,cash=10000)
    bt.run()
    bt.optimize(
        multiplier1=range(100,300,10),
        multiplier2=range(100,300,10),
        maximize=optimfunc
    )
    print(bt)
    bt.plot()

main()

#alternative for backtesting
#1)
"""stlo=range(90,98,1),
tkpr=range(102,110,1),"""
#2)
"""multiplier1=range(100,300,10),
multiplier2=range(100,300,10),"""