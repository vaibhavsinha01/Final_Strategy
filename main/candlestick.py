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

# Define start and end dates for data fetching
startdate = datetime.datetime(2024, 4, 28)
enddate = datetime.datetime(2024, 6, 25)
ticker = 'adanient.ns'

# Function to fetch data
def FetchData(stock):
    data = yf.download(stock, start=startdate, end=enddate, interval='15m')
    data.drop('Adj Close', axis=1, inplace=True)
    return data

# Function to add technical indicators to the data
def Functions(data):
    data['returns'] = data['Close'].pct_change(1)
    data['DEMA200'] = talib.DEMA(data['Close'], timeperiod=200)
    data['Doji'] = talib.CDLDOJI(data['Open'], data['High'], data['Low'], data['Close'])
    data['Engulfing'] = talib.CDLENGULFING(data['Open'], data['High'], data['Low'], data['Close'])
    data['Hammer'] = talib.CDLHAMMER(data['Open'], data['High'], data['Low'], data['Close'])
    data['HangingMan'] = talib.CDLHANGINGMAN(data['Open'], data['High'], data['Low'], data['Close'])
    data['Harami'] = talib.CDLHARAMI(data['Open'], data['High'], data['Low'], data['Close'])
    data['MorningStar'] = talib.CDLMORNINGSTAR(data['Open'], data['High'], data['Low'], data['Close'])
    data['EveningStar'] = talib.CDLEVENINGSTAR(data['Open'], data['High'], data['Low'], data['Close'])
    data['Piercing'] = talib.CDLPIERCING(data['Open'], data['High'], data['Low'], data['Close'])
    data['DarkCloudCover'] = talib.CDLDARKCLOUDCOVER(data['Open'], data['High'], data['Low'], data['Close'])
    data['ShootingStar'] = talib.CDLSHOOTINGSTAR(data['Open'], data['High'], data['Low'], data['Close'])
    data['InvertedHammer'] = talib.CDLINVERTEDHAMMER(data['Open'], data['High'], data['Low'], data['Close'])
    data['Marubozu'] = talib.CDLMARUBOZU(data['Open'], data['High'], data['Low'], data['Close'])
    data['ThreeWhiteSoldiers'] = talib.CDL3WHITESOLDIERS(data['Open'], data['High'], data['Low'], data['Close'])
    data['ThreeBlackCrows'] = talib.CDL3BLACKCROWS(data['Open'], data['High'], data['Low'], data['Close'])
    data['DragonflyDoji'] = talib.CDLDRAGONFLYDOJI(data['Open'], data['High'], data['Low'], data['Close'])
    data['GravestoneDoji'] = talib.CDLGRAVESTONEDOJI(data['Open'], data['High'], data['Low'], data['Close'])
    data['BeltHold'] = talib.CDLBELTHOLD(data['Open'], data['High'], data['Low'], data['Close'])
    data['Kicking'] = talib.CDLKICKING(data['Open'], data['High'], data['Low'], data['Close'])
    data.dropna(inplace=True)
    return data

# Function to generate trading signals
def Signal(data):
    signals = ['Signal1', 'Signal2', 'Signal3', 'Signal4', 'Signal5', 'Signal6', 'Signal7', 'Signal8', 'Signal9', 'Signal10',
               'Signal11', 'Signal12', 'Signal13', 'Signal14', 'Signal15', 'Signal16', 'Signal17', 'Signal18', 'Signal19', 'Signal20']
    for signal in signals:
        data[signal] = 0

    for i in range(4, len(data)):
        if data['Close'].iloc[i] > data['Close'].iloc[i-1]:
            data.loc[data.index[i], 'Signal1'] = 1
        if data['Close'].iloc[i-1] > data['Close'].iloc[i-2]:
            data.loc[data.index[i], 'Signal2'] = 1
        if data['Close'].iloc[i-2] > data['Close'].iloc[i-3]:
            data.loc[data.index[i], 'Signal3'] = 1
        if data['Doji'].iloc[i] != 0:
            data.loc[data.index[i], 'Signal4'] = 1
        if data['Engulfing'].iloc[i] != 0:
            data.loc[data.index[i], 'Signal5'] = 1
        if data['HangingMan'].iloc[i] != 0:
            data.loc[data.index[i], 'Signal6'] = 1
        if data['Harami'].iloc[i] != 0:
            data.loc[data.index[i], 'Signal7'] = 1
        if data['MorningStar'].iloc[i] != 0:
            data.loc[data.index[i], 'Signal8'] = 1
        if data['EveningStar'].iloc[i] != 0:
            data.loc[data.index[i], 'Signal9'] = 1
        if data['Piercing'].iloc[i] != 0:
            data.loc[data.index[i], 'Signal10'] = 1
        if data['DarkCloudCover'].iloc[i] != 0:
            data.loc[data.index[i], 'Signal11'] = 1
        if data['ShootingStar'].iloc[i] != 0:
            data.loc[data.index[i], 'Signal12'] = 1
        if data['InvertedHammer'].iloc[i] != 0:
            data.loc[data.index[i], 'Signal13'] = 1
        if data['Marubozu'].iloc[i] != 0:
            data.loc[data.index[i], 'Signal14'] = 1
        if data['ThreeWhiteSoldiers'].iloc[i] != 0:
            data.loc[data.index[i], 'Signal15'] = 1
        if data['ThreeBlackCrows'].iloc[i] != 0:
            data.loc[data.index[i], 'Signal16'] = 1
        if data['DragonflyDoji'].iloc[i] != 0:
            data.loc[data.index[i], 'Signal17'] = 1
        if data['GravestoneDoji'].iloc[i] != 0:
            data.loc[data.index[i], 'Signal18'] = 1
        if data['BeltHold'].iloc[i] != 0:
            data.loc[data.index[i], 'Signal19'] = 1
        if data['Kicking'].iloc[i] != 0:
            data.loc[data.index[i], 'Signal20'] = 1

    data.dropna(inplace=True)
    return data

# Fetch, process, and generate signals
a = FetchData(ticker)
b = Functions(a)
df = Signal(b)

# Split the data
split = int(len(df) * 0.8)

# Train a Decision Tree Regressor
x_train = df[['DEMA200', 'Signal1', 'Signal2', 'Signal3', 'Signal4', 'Signal5', 'Signal6', 'Signal7', 'Signal8', 'Signal9',
              'Signal10', 'Signal11', 'Signal12', 'Signal13', 'Signal14', 'Signal15', 'Signal16', 'Signal17', 'Signal18',
              'Signal19', 'Signal20']].iloc[:split]
y_train = df['returns'].iloc[:split]

# reg = DecisionTreeRegressor()
# reg = LinearRegression()
# reg = RandomForestRegressor()
# reg = KNeighborsRegressor()
# reg = MLPRegressor() #best for long time
# reg = BayesianRidge()
# reg = GradientBoostingRegressor()
reg = SVR() #best model for 15m

reg.fit(x_train, y_train)

# Make predictions
df['predict'] = reg.predict(df[['DEMA200', 'Signal1', 'Signal2', 'Signal3', 'Signal4', 'Signal5', 'Signal6', 'Signal7',
                                'Signal8', 'Signal9', 'Signal10', 'Signal11', 'Signal12', 'Signal13', 'Signal14',
                                'Signal15', 'Signal16', 'Signal17', 'Signal18', 'Signal19', 'Signal20']])
df['position_reg'] = np.sign(df['predict'])
df['strategy_reg'] = df['position_reg'] * df['returns'].shift(1)

# Calculate win and loss rates
df['win'] = np.where(df['strategy_reg'] > 0, 1, 0)
df['loss'] = np.where(df['strategy_reg'] < 0, 1, 0)
df['no_return'] = np.where(df['strategy_reg'] == 0, 1, 0)

num_wins = df['win'].sum()
num_losses = df['loss'].sum()
num_no_returns = df['no_return'].sum()

total_trades = num_wins + num_losses
win_rate = num_wins / total_trades if total_trades > 0 else 0
loss_rate = num_losses / total_trades if total_trades > 0 else 0

print(f"Decision Tree Model:")
print(f"Number of wins: {num_wins}")
print(f"Number of losses: {num_losses}")
print(f"Win rate: {win_rate:.2f}")
print(f"Loss rate: {loss_rate:.2f}")

# Plotting the strategy performance
plt.figure(figsize=(12, 6))
plt.plot((df['strategy_reg'] + 1).cumprod(), label='Strategy')
plt.plot((df['returns'] + 1).cumprod(), label='Market')
plt.legend()
plt.title('Strategy vs Market Performance')
plt.show()
