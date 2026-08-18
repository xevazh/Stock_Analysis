import pandas as pd
import numpy as np

from python import wealth

df = pd.read_csv("practice_20stocks_180days.csv")
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values(['date','ticker'])
print(df.shape)
print(df.isna().sum())
print("duplicates:", df.duplicated().sum())

#daily return
df['lag_1'] = df.groupby('ticker')['close'].shift(1)
df['return'] = df['close']/df['lag_1'] - 1

#20 days momentum
df['mom20'] = df.groupby('ticker')['close'].pct_change(20)
df["mom60"] = (df.groupby("ticker")["close"].pct_change(60))

#20 rolling Volatiliy
df['vol20'] = df.groupby('ticker')['return'].transform(lambda x: x.rolling(20).std())
df["vol60"] = (df.groupby("ticker")["return"].transform(lambda x: x.rolling(60).std()))

#改成月频
#创建月份
df["month"] = (df["date"].dt.to_period("M"))
#找每只股票每个月最后一个交易日
monthly = (df.sort_values("date").groupby(["ticker", "month"]).tail(1))
print(monthly)

#Forward 1-Month Return   本月月末观察到的 momentum，能不能预测下个月收益
monthly['forward_return'] = monthly.groupby('ticker')['close'].shift(-1) / monthly['close'] - 1

#Momentum Ranking
monthly['momentum_rank'] = monthly.groupby('month')['mom60'].rank(pct=True)
#split into 5 groups
monthly["momentum_group"] = (monthly.groupby("month")["mom60"].transform(
               lambda x: pd.qcut(x,5, labels=False, duplicates="drop")) + 1
)



portfolio_returns = (
    monthly.groupby(["month", "momentum_group"])["forward_return"].mean().unstack()
)
print(portfolio_returns)

#long high momentum, short low momentum
long_short = (
    portfolio_returns[5]
    - portfolio_returns[1]
)

print(long_short)


avg_monthly_return = long_short.mean()
print('avg_monthly_return:',avg_monthly_return)

#Annualised
annualised_return = ( (1 + long_short).prod()  ** (12 / len(long_short))  - 1 )
print('annualised_return:',annualised_return)

#Annualised Volatility
monthly_vol = long_short.std()
annualised_vol = monthly_vol  * np.sqrt(12)
print('annualised_vol:',annualised_vol)

#sharpe ratio(assume risk free rate = 0)
sharpe = long_short.mean()/ long_short.std() * np.sqrt(12)
print('sharpe ratio:',sharpe)

#maximum drawdowm
wealth = (1 + long_short).cumprod()
peak = wealth.cummax()
drawdown = (wealth / peak - 1)
max_drawdown = drawdown.min()
print('max_drawdown:',max_drawdown)