from pybit.unified_trading import HTTP
import Creds

#the funciton of this code is to hold funcitons of the broker wrapper

class Broker:
    def __init__(self):
        self.session = HTTP(
            testnet=True,
            api_key=Creds.KEY,  
            api_secret=Creds.SECRET
)

    def get_kline_data(self,category='spot',symbol='BTCUSDT',interval=60):
        print(self.session.get_kline(
        category=category,
        symbol=symbol,
        interval=interval   #here we can add start and end to get more accurate data and interval is 1 minute.
))

    def get_account_balance(self,accountType='UNIFIED'):
        print(self.session.get_wallet_balance(
        accountType=accountType   #here we can add coin balance to get the balance of that specified coin also for more coin we can seperate the input by commas
))

    def get_coin_balance(self,accountType='UNIFIED',coin='BTC'):
        print(self.session.get_coin_balance(
        accountType=accountType,
        coin=coin
))
    
    def place_order(self,category='spot',symbol='BTCUSDT',side='Buy',orderType='Market',qty='0.1'):
        print(self.session.place_order(
        category=category,
        symbol= symbol,
        side= side,
        orderType=orderType,
        qty=qty  #different parameters are there for limit order type also only market type are supported by some TP&SL 
))

    def cancel_all_order(self,category='spot',settleCoin='USDT'):
        print(self.session.cancel_all_orders(
        category=category,
        settleCoin=settleCoin
))

    def cancel_order(self,category='spot',symbol='BTCUSDT',orderId='c6f055d9-7f21-4079-913d-e6523a9cfffa'):
        print(self.session.cancel_order(
        category=category,
        symbol=symbol,
        orderId=orderId # here either orderID or orderTypeId is required
))

    def get_ticker(self,category='spot',symbol='BTCUSDT'):
        print(self.session.get_tickers(
        category=category,
        symbol=symbol
))

    def get_trade_history(self,category='spot'):
        print(self.session.get_executions(
        category=category
))

    def set_leverage(self,category="spot",symbol="BTCUSDT",buyLeverage="10",sellLeverage="10"):
        print(self.session.set_leverage(
        category=category,
        symbol=symbol,
        buyLeverage=buyLeverage, 
        sellLeverage=sellLeverage  #set the leverage for both
))

    def set_trading_stop(self,category="spot",symbol="BTCUSDT",positionIdx=0):
        print(self.session.set_trading_stop(
        category=category,
        symbol=symbol,
        positionIdx=positionIdx,# 0 for one way mode 1 for buy hedge and 2 for sell hedge
        # takeProfit="0.6",
        # stopLoss="0.2", #here level 0 cancels the stoploss or the takeprofit
        # tpTriggerBy="MarkPrice",
        # slTriggerB="IndexPrice",
        # tpslMode="Partial",
        # tpOrderType="Limit",
        # slOrderType="Limit",
        # tpSize="50",
        # slSize="50",
        # tpLimitPrice="0.57",
        # slLimitPrice="0.21"
        # this is like the above mode just that it has a lot of conditions and is the newer version and is better
))
    
    def get_fee_rate(self,category='spot',symbol='BTCUSDT'):
        print(self.session.get_fee_rates(
            category=category,
            symbol=symbol
))


