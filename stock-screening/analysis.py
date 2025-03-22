import yfinance as yf
import pandas as pd
import mplfinance as mpf
from datetime import datetime, timedelta
import numpy as np

# translated from pinescript
def find_pivot_points(data, lookback_left=5, lookback_right=5):
    """Find pivot high and low points in the data."""
    data = data.copy()
    
    # Initialize pivot columns
    data['pivot_low'] = False
    data['pivot_high'] = False
    
    # Find pivot lows
    for i in range(lookback_left, len(data) - lookback_right):
        if all(data['Low'].iloc[i] <= data['Low'].iloc[i-lookback_left:i]) and \
           all(data['Low'].iloc[i] <= data['Low'].iloc[i+1:i+lookback_right+1]):
            data.loc[data.index[i], 'pivot_low'] = True
    
    # Find pivot highs
    for i in range(lookback_left, len(data) - lookback_right):
        if all(data['High'].iloc[i] >= data['High'].iloc[i-lookback_left:i]) and \
           all(data['High'].iloc[i] >= data['High'].iloc[i+1:i+lookback_right+1]):
            data.loc[data.index[i], 'pivot_high'] = True
    
    return data

def find_previous_pivot(data, current_idx, lookback, pivot_type='low'):
    """Find the previous pivot point of specified type."""
    if pivot_type == 'low':
        pivot_col = 'pivot_low'
        price_col = 'Low'
    else:
        pivot_col = 'pivot_high'
        price_col = 'High'
    
    for i in range(current_idx - lookback, -1, -1):
        if data[pivot_col].iloc[i]:
            return i, data[price_col].iloc[i]
    return None, None

def fetch_stock_data(ticker, start_date, end_date):
    """Fetches daily stock data from Yahoo Finance with unadjusted prices."""
    stock = yf.download(ticker, start=start_date, end=end_date, auto_adjust=False)
    # Flatten MultiIndex columns for easier access
    stock.columns = [col[0] for col in stock.columns]
    return stock

def calculate_macd(data, short_window=12, long_window=26, signal_window=9):
    """Calculates MACD and Signal Line."""
    data = data.copy()
    
    # Calculate EMAs only after we have enough data points
    data['EMA_12'] = data['Close'].ewm(span=short_window, adjust=False, min_periods=short_window).mean()
    data['EMA_26'] = data['Close'].ewm(span=long_window, adjust=False, min_periods=long_window).mean()
    
    # Calculate MACD only when both EMAs are available
    data['MACD'] = data['EMA_12'] - data['EMA_26']
    
    # Calculate Signal Line only after we have enough MACD values
    data['Signal_Line'] = data['MACD'].ewm(span=signal_window, adjust=False, min_periods=signal_window).mean()
    
    # Drop rows where we don't have enough data
    data = data.dropna()
    
    return data[['Close', 'MACD', 'Signal_Line']]

def calculate_macd_divergences(data, lookback_left=5, lookback_right=5):
    """Calculate MACD divergences and return the dates of buy/sell signals."""
    data = data.copy()

    # Check if MACD column exists before proceeding
    if 'MACD' not in data.columns:
        raise ValueError("MACD column not found. Ensure MACD is calculated first.")

    # Find pivot points
    data = find_pivot_points(data, lookback_left, lookback_right)

    # Remove rows where MACD is NaN (due to missing EMA history)
    data = data.dropna(subset=['MACD'])

    # Lists to store the dates of divergences
    bull_div_dates = []
    bear_div_dates = []
    hidden_bull_div_dates = []
    hidden_bear_div_dates = []

    for i in range(lookback_left + lookback_right, len(data)):
        # Regular Bullish Divergence
        if data['pivot_low'].iloc[i]:
            prev_pivot_idx, prev_price = find_previous_pivot(data, i, lookback_left + lookback_right, 'low')
            if prev_pivot_idx is not None:
                price_ll = data['Low'].iloc[i] < prev_price
                osc_hl = data['MACD'].iloc[i] > data['MACD'].iloc[prev_pivot_idx]
                below_zero = data['MACD'].iloc[i] < 0
                if price_ll and osc_hl and below_zero:
                    bull_div_dates.append(data.index[i])  # Store the date of the divergence

        # Regular Bearish Divergence
        if data['pivot_high'].iloc[i]:
            prev_pivot_idx, prev_price = find_previous_pivot(data, i, lookback_left + lookback_right, 'high')
            if prev_pivot_idx is not None:
                price_hh = data['High'].iloc[i] > prev_price
                osc_lh = data['MACD'].iloc[i] < data['MACD'].iloc[prev_pivot_idx]
                above_zero = data['MACD'].iloc[i] > 0
                if price_hh and osc_lh and above_zero:
                    bear_div_dates.append(data.index[i])  # Store the date of the divergence

        # Hidden Bullish Divergence
        if data['pivot_low'].iloc[i]:
            prev_pivot_idx, prev_price = find_previous_pivot(data, i, lookback_left + lookback_right, 'low')
            if prev_pivot_idx is not None:
                price_hl = data['Low'].iloc[i] > prev_price
                osc_ll = data['MACD'].iloc[i] < data['MACD'].iloc[prev_pivot_idx]
                below_zero = data['MACD'].iloc[i] < 0
                if price_hl and osc_ll and below_zero:
                    hidden_bull_div_dates.append(data.index[i])  # Store the date of the divergence

        # Hidden Bearish Divergence
        if data['pivot_high'].iloc[i]:
            prev_pivot_idx, prev_price = find_previous_pivot(data, i, lookback_left + lookback_right, 'high')
            if prev_pivot_idx is not None:
                price_lh = data['High'].iloc[i] < prev_price
                osc_hh = data['MACD'].iloc[i] > data['MACD'].iloc[prev_pivot_idx]
                above_zero = data['MACD'].iloc[i] > 0
                if price_lh and osc_hh and above_zero:
                    hidden_bear_div_dates.append(data.index[i])  # Store the date of the divergence

    # Combine and sort all bullish divergence dates
    all_bull_div_dates = sorted(bull_div_dates + hidden_bull_div_dates)

    # Check if the latest bullish divergence is within the last 5 days
    currently_buyable = False
    if all_bull_div_dates:
        latest_divergence_date = all_bull_div_dates[-1]
        latest_stock_date = data.index[-1]
        if (latest_stock_date - latest_divergence_date).days <= 5:
            currently_buyable = True
    print(currently_buyable)
    return currently_buyable

def load_tickers_from_file(filename):
    """Reads a comma-separated list of tickers from a file and returns a list."""
    with open(filename, 'r') as file:
        content = file.read().strip()  # Read and remove leading/trailing spaces
        tickers = content.split(',')   # Split by comma
        tickers = [ticker.strip().upper() for ticker in tickers]  # Clean up spaces and ensure uppercase
    return tickers

ticker_list = load_tickers_from_file("tickers.txt")
print(ticker_list)
buyable_tickers = []


if __name__ == "__main__":
    
    # Calculate dates dynamically
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=1*365)).strftime("%Y-%m-%d")   
    for ticker in ticker_list:
        print(ticker)
    # Get the full stock data
        stock_data = fetch_stock_data(ticker, start_date, end_date)
    
    # Calculate MACD
        macd_data = calculate_macd(stock_data)
    
    # Merge MACD data with stock data
        plot_data = stock_data.copy()
        plot_data['MACD'] = macd_data['MACD']
        plot_data['Signal_Line'] = macd_data['Signal_Line']
    
    # Ensure 'MACD' is present before calling calculate_macd_divergences
        if 'MACD' in plot_data.columns:
        # Calculate divergences and output the dates
            buyableBool = calculate_macd_divergences(plot_data)
        else:
            print("Error: MACD column is missing from plot_data!")
        if buyableBool == True:
            buyable_tickers.append(ticker)
    print(buyable_tickers)