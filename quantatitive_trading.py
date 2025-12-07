import yfinance as yf
import pandas as pd
from datetime import datetime

# Load SPY ticker
ticker = yf.Ticker("SPY")

# List all available option expiration dates
expirations = ticker.options
print("Available expirations:", expirations)

# Select the nearest expiration (you can loop through more later)
expiry = expirations[0]

# Fetch option chain
chain = ticker.option_chain(expiry)
calls = chain.calls
puts = chain.puts

# Add metadata
today = datetime.now().strftime("%Y-%m-%d")
calls["type"] = "call"
puts["type"] = "put"

# Combine calls and puts
options = pd.concat([calls, puts], ignore_index=True)
options["retrieved_date"] = today
options["expiration"] = expiry

# Save to CSV
filename = f"spy_options_{today}.csv"
options.to_csv(filename, index=False)
print(f"Saved option chain to: {filename}")
