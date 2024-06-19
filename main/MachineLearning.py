import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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


# MLP doesn't work and the best is probably random forest.
# for AAPL the SVR/MLP model is showing a win rate of 100 % so check that.
# Download data
data = yf.download('AAPL')
data.drop('Adj Close', axis=1, inplace=True)

def feature_engineering(data):
    data['returns'] = data['Close'].pct_change(1)
    data['RSI'] = talib.RSI(data['Close'], timeperiod=2)
    data['StochRSI'], _ = talib.STOCHRSI(data['Close'], timeperiod=14)
    data['EMA200'] = talib.EMA(data['Close'], timeperiod=200)
    data['MACD'], data['MACDSignal'], data['MACDHistory'] = talib.MACD(data['Close'], fastperiod=12, slowperiod=26)
    data['ATR'] = talib.ATR(data['High'],data['Low'],data['Close'],timeperiod=14)
    data['MaximaMinima'] = 0

    for i in range(4, len(data) - 4):
        is_maxima = True
        is_minima = True
        for j in range(i - 3, i + 4):
            if data['Close'].iloc[i] < data['Close'].iloc[j]:
                is_maxima = False
            if data['Close'].iloc[i] > data['Close'].iloc[j]:
                is_minima = False
        if is_maxima:
            data.loc[data.index[i], 'MaximaMinima'] = 1
        if is_minima:
            data.loc[data.index[i], 'MaximaMinima'] = -1

    data.dropna(inplace=True)
    return data

# Apply feature engineering
df = feature_engineering(data)

# Split the data
split = int(0.9 * len(df))

# Check for NaN values in the DataFrame before splitting
df.dropna(inplace=True)

x_train = df[['MACD', 'MACDSignal', 'RSI', 'StochRSI', 'Volume', 'EMA200', 'MaximaMinima','ATR']].iloc[:split]
y_train = df['returns'].iloc[:split]  # No need to use double brackets for y_train
x_test = df[['MACD', 'MACDSignal', 'RSI', 'StochRSI', 'Volume', 'EMA200', 'MaximaMinima','ATR']].iloc[split:]
y_test = df['returns'].iloc[split:]  # No need to use double brackets for y_test

# Ensure no NaN values in training data after splitting
x_train.dropna(inplace=True)
y_train.dropna(inplace=True)

# Train the models
reg = LinearRegression()
reg1 = DecisionTreeRegressor()  # Adding random_state for reproducibility
reg2 = RandomForestRegressor()
reg3 = GradientBoostingRegressor()
reg4 = SVR()
reg5 = KNeighborsRegressor()
reg6 = MLPRegressor()  # Increased max_iter for convergence
reg7 = BayesianRidge()

reg.fit(x_train, y_train)
reg1.fit(x_train, y_train)
reg2.fit(x_train, y_train)
reg3.fit(x_train, y_train)
reg4.fit(x_train, y_train)
reg5.fit(x_train, y_train)
reg6.fit(x_train, y_train)
reg7.fit(x_train, y_train)

# Predict
df['predict'] = reg.predict(df[['MACD', 'MACDSignal', 'RSI', 'StochRSI', 'Volume', 'EMA200', 'MaximaMinima','ATR']])
df['predict1'] = reg1.predict(df[['MACD', 'MACDSignal', 'RSI', 'StochRSI', 'Volume', 'EMA200', 'MaximaMinima','ATR']])
df['predict2'] = reg2.predict(df[['MACD', 'MACDSignal', 'RSI', 'StochRSI', 'Volume', 'EMA200', 'MaximaMinima','ATR']])
df['predict3'] = reg3.predict(df[['MACD', 'MACDSignal', 'RSI', 'StochRSI', 'Volume', 'EMA200', 'MaximaMinima','ATR']])
df['predict4'] = reg4.predict(df[['MACD', 'MACDSignal', 'RSI', 'StochRSI', 'Volume', 'EMA200', 'MaximaMinima','ATR']])
df['predict5'] = reg5.predict(df[['MACD', 'MACDSignal', 'RSI', 'StochRSI', 'Volume', 'EMA200', 'MaximaMinima','ATR']])
df['predict6'] = reg6.predict(df[['MACD', 'MACDSignal', 'RSI', 'StochRSI', 'Volume', 'EMA200', 'MaximaMinima','ATR']])
df['predict7'] = reg7.predict(df[['MACD', 'MACDSignal', 'RSI', 'StochRSI', 'Volume', 'EMA200', 'MaximaMinima','ATR']])

# Plot the prediction
df['predict'].plot(label='Linear Regression Predictions')
df['predict1'].plot(label='Decision Tree Predictions')
df['predict2'].plot(label='Random Forest Regressor')
df['predict3'].plot(label='Gradient Boosting Regressor')
df['predict4'].plot(label='SVR')
df['predict5'].plot(label='K Neighbors Regressor')
df['predict6'].plot(label='MLP Regressor')
df['predict7'].plot(label='Bayesian Ridge')
plt.legend()
plt.title('Stock Return Predictions')
plt.xlabel('Date')
plt.ylabel('Returns')
plt.show()

# Compute positions and strategy returns
df["position_reg"] = np.sign(df["predict"])
df["position_tree"] = np.sign(df["predict1"])
df["position_rf"] = np.sign(df["predict2"])
df["position_gb"] = np.sign(df["predict3"])
df["position_svr"] = np.sign(df["predict4"])
df["position_knn"] = np.sign(df["predict5"])
df["position_mlp"] = np.sign(df["predict6"])
df["position_br"] = np.sign(df["predict7"])

df["strategy_reg"] = df["returns"] * df["position_reg"].shift(1)
df["strategy_tree"] = df["returns"] * df["position_tree"].shift(1)
df["strategy_rf"] = df["returns"] * df["position_rf"].shift(1)
df["strategy_gb"] = df["returns"] * df["position_gb"].shift(1)
df["strategy_svr"] = df["returns"] * df["position_svr"].shift(1)
df["strategy_knn"] = df["returns"] * df["position_knn"].shift(1)
df["strategy_mlp"] = df["returns"] * df["position_mlp"].shift(1)
df["strategy_br"] = df["returns"] * df["position_br"].shift(1)

# Plot cumulative strategy returns
(df["strategy_reg"].iloc[split:].cumsum() * 100).plot(label='Linear Regression Strategy')
(df["strategy_tree"].iloc[split:].cumsum() * 100).plot(label='Decision Tree Strategy')
(df["strategy_rf"].iloc[split:].cumsum() * 100).plot(label='Random Forest Strategy')
(df["strategy_gb"].iloc[split:].cumsum() * 100).plot(label='Gradient Boosting Strategy')
(df["strategy_svr"].iloc[split:].cumsum() * 100).plot(label='SVR Strategy')
(df["strategy_knn"].iloc[split:].cumsum() * 100).plot(label='K Neighbors Strategy')
(df["strategy_mlp"].iloc[split:].cumsum() * 100).plot(label='MLP Strategy')
(df["strategy_br"].iloc[split:].cumsum() * 100).plot(label='Bayesian Ridge Strategy')
plt.legend()
plt.title('Cumulative Strategy Returns')
plt.xlabel('Date')
plt.ylabel('Cumulative Returns (%)')
plt.show()

# Function to calculate number of wins, losses, and error rates
def calculate_performance(position_column):
    no_of_wins = 0
    no_of_losses = 0
    no_of_noreturn = 0
    no_of_error = 0

    for i in range(int(0.9*len(df)), len(df)-1):
        if df[position_column].iloc[i-1] == 1 and df['returns'].iloc[i] > 0:
            no_of_wins += 1
        elif df[position_column].iloc[i-1] == -1 and df['returns'].iloc[i] < 0:
            no_of_losses += 1
        elif df[position_column].iloc[i-1] != 1 and df[position_column].iloc[i-1] != -1:
            no_of_noreturn += 1
        else:
            no_of_error += 1

    win_rate = (no_of_wins / (no_of_wins + no_of_losses)) * 100
    error_rate = (no_of_error / len(df)) * 100

    return no_of_wins, no_of_losses, no_of_noreturn, win_rate, error_rate

# Calculate performance for each model
performance_reg = calculate_performance("position_reg")
performance_tree = calculate_performance("position_tree")
performance_rf = calculate_performance("position_rf")
performance_gb = calculate_performance("position_gb")
performance_svr = calculate_performance("position_svr")
performance_knn = calculate_performance("position_knn")
performance_mlp = calculate_performance("position_mlp")
performance_br = calculate_performance("position_br")

# Print results for each model
print(f'Linear Regression Model:\nNumber of wins: {performance_reg[0]}\nNumber of losses: {performance_reg[1]}\nNumber of no returns: {performance_reg[2]}\nWin rate: {performance_reg[3]:.2f}%\nError rate: {performance_reg[4]:.2f}%\n')
print(f'Decision Tree Model:\nNumber of wins: {performance_tree[0]}\nNumber of losses: {performance_tree[1]}\nNumber of no returns: {performance_tree[2]}\nWin rate: {performance_tree[3]:.2f}%\nError rate: {performance_tree[4]:.2f}%\n')
print(f'Random Forest Model:\nNumber of wins: {performance_rf[0]}\nNumber of losses: {performance_rf[1]}\nNumber of no returns: {performance_rf[2]}\nWin rate: {performance_rf[3]:.2f}%\nError rate: {performance_rf[4]:.2f}%\n')
print(f'Gradient Boosting Model:\nNumber of wins: {performance_gb[0]}\nNumber of losses: {performance_gb[1]}\nNumber of no returns: {performance_gb[2]}\nWin rate: {performance_gb[3]:.2f}%\nError rate: {performance_gb[4]:.2f}%\n')
print(f'SVR Model:\nNumber of wins: {performance_svr[0]}\nNumber of losses: {performance_svr[1]}\nNumber of no returns: {performance_svr[2]}\nWin rate: {performance_svr[3]:.2f}%\nError rate: {performance_svr[4]:.2f}%\n')
print(f'K Neighbors Model:\nNumber of wins: {performance_knn[0]}\nNumber of losses: {performance_knn[1]}\nNumber of no returns: {performance_knn[2]}\nWin rate: {performance_knn[3]:.2f}%\nError rate: {performance_knn[4]:.2f}%\n')
print(f'MLP Model:\nNumber of wins: {performance_mlp[0]}\nNumber of losses: {performance_mlp[1]}\nNumber of no returns: {performance_mlp[2]}\nWin rate: {performance_mlp[3]:.2f}%\nError rate: {performance_mlp[4]:.2f}%\n')
print(f'Bayesian Ridge Model:\nNumber of wins: {performance_br[0]}\nNumber of losses: {performance_br[1]}\nNumber of no returns: {performance_br[2]}\nWin rate: {performance_br[3]:.2f}%\nError rate: {performance_br[4]:.2f}%\n')
