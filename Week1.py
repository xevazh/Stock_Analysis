#WEEK 1 MINI PROJECT
import yfinance as yf
import pandas as pd
import numpy as np

ticker = "0700.HK"
start_date = "2024-01-16"
end_date = "2026-08-17"

df = yf.download(
    ticker,
    start=start_date,
    end=end_date,
    interval="1d",
    auto_adjust=False,
    progress=False
)

# 处理新版 yfinance 可能出现的多层列名
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)

df = df.reset_index()

# 保存 CSV
df.to_csv(
    "Tencent_0700_HK.csv",
    index=False,
    encoding="utf-8-sig"
)

print(df.head())
print("\n下载完成！")
print(f"共 {len(df)} 条数据")

# 1. 读取csv
df = pd.read_csv("Tencent_0700_HK.csv")
# 2. 把 Date 转成日期
df["Date"] = pd.to_datetime(df["Date"])
# 3. Date 设为 index
df = df.set_index("Date")
# 4. 取 Close 价格
prices = df["Close"]

#定义
def calculate_returns(prices):
    return prices.pct_change().dropna()

def annualised_return(returns):
    return (np.prod(1 + returns) ** (252 / len(returns))- 1)

def annualised_volatility(returns):
    return np.std(returns) * np.sqrt(252)

def sharpe_ratio(returns):
    return np.mean(returns)/ np.std(returns,ddof = 1 ) * np.sqrt(252)

def maximum_drawdown(returns):
    wealth = np.cumprod(1 + returns)
    running_max = np.maximum.accumulate(wealth)
    drawdown = wealth / running_max - 1
    return np.min(drawdown)

# =========================
# Main Program
# =========================
returns = calculate_returns(prices)
print(returns)

total_return = np.prod((1 + returns)) - 1
print("Total Return:",total_return)

ann_return = annualised_return(returns)
print("Annualised Return:", ann_return)

ann_vol = annualised_volatility(returns)
print("Annualised Volatility:", ann_vol)

sharpe = sharpe_ratio(returns)
print("Sharpe Ratio:",sharpe)

max_dd = maximum_drawdown(returns)
print("Maximum Drawdown:",max_dd)

best_date = returns.idxmax()
best_return = returns.max()
print("Best Day:", best_date, best_return)

worst_date = returns.idxmin()
worst_return = returns.min()
print("Worst Day:", worst_date, worst_return)

#PLOT
plt.figure()
plt.plot(prices)
plt.title("Price")
plt.xlabel("Date")
plt.ylabel("Price")
plt.show()



rolling_vol = (returns.rolling(20).std() * np.sqrt(252))
plt.figure()
plt.plot(rolling_vol)
plt.title("20-Day Rolling Volatility")
plt.xlabel("Date")
plt.ylabel("Annualised Volatility")
plt.show()