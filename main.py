import yfinance as yf
import numpy as np

ticker = yf.Ticker("KO")
historical_data = ticker.history(start = "2023-05-29", end = "2025-05-29")

# First step--final closing price
S0 = historical_data["Close"].iloc[-1]

# Calculate the daily returns as a decimal
returns = historical_data["Close"].pct_change().dropna()

# The mean of the returns for the average expected annual growth (drift)
mu = returns.mean() * 252

# The annualized volatility (standard deviation of daily percentage returns times the sqrt of the number of steps)
sigma = returns.std() * (252 ** 0.5)

# The size of each time step in years
dt = 1/252

paths = 30000
days = 252

# The random number array drawn fresh for each step and path(shock)
ran_num = np.random.default_rng(seed = 42)
Z_arr = ran_num.normal(loc = 0.0, scale = 1.0, size = (paths, days))

# Creates the second part of the formula for each entry
full_arr = np.exp((mu - (sigma ** 2) / 2) * dt + (sigma * np.sqrt(dt) * Z_arr))

# Creates the price paths by multiplying last day's price times next
full_arr = S0 * np.cumprod(full_arr, axis = 1)

final_pred = full_arr[:, -1]
print(f"The final prediction (mean) for the stock in a year (5/29/2026) is {round(np.mean(final_pred), 2)}")
print(f"The median price for the stock in a year (5/29/2026) is: {round(np.median(final_pred), 2)}")

for p in [5, 25, 50, 75, 95]:
    print(f"{p}th percentile: {np.percentile(final_pred, p):.2f}")