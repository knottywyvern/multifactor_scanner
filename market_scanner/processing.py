"""
Data processing functions for the Market Scanner.
"""
import pandas as pd
import numpy as np
import logging
import os

from .utils import display_log

def output_rejected_tickers(data_table, output_dir="output"):
    """
    Output rejected tickers to file.
    
    Args:
        data_table (pandas.DataFrame): Data table
        output_dir (str): Output directory
    """
    display_log("Outputting rejected tickers to rejected_tickers.txt.")
    
    try:
        # Check if required columns exist
        required_cols = ["Institutional Holders fct", "Turnover fct", "Revenue fct", "Security Type fct"]
        missing_cols = [col for col in required_cols if col not in data_table.columns]
        
        if missing_cols:
            display_log(f"Warning: Missing columns in data table: {missing_cols}")
            
            # If Security Type fct is missing, we can't filter by it
            if "Security Type fct" in missing_cols:
                display_log("Cannot identify rejected tickers without 'Security Type fct' column")
                # Just write the file with no tickers
                os.makedirs(output_dir, exist_ok=True)
                with open(os.path.join(output_dir, 'rejected_tickers.txt'), 'w') as f:
                    f.write("# No rejected tickers identified\n")
                return
        
        # Reject only stocks with missing Institutional Holders, Turnover, or Revenue
        filter_cols = [col for col in ["Institutional Holders fct", "Turnover fct", "Revenue fct"] 
                    if col in data_table.columns]
        
        if filter_cols:
            rejected_mask = (data_table["Security Type fct"] == "STOCK")
            for col in filter_cols:
                rejected_mask = rejected_mask & data_table[col].isna()
                
            rejected_tickers = data_table.loc[rejected_mask, "Ticker"] if "Ticker" in data_table.columns else []
        else:
            display_log("Warning: No filter columns available to identify rejected tickers")
            rejected_tickers = []
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Write to file
        with open(os.path.join(output_dir, 'rejected_tickers.txt'), 'w') as f:
            if len(rejected_tickers) == 0:
                f.write("# No rejected tickers identified\n")
            else:
                for ticker in rejected_tickers:
                    f.write("%s\n" % ticker)
                    
        display_log(f"Wrote {len(rejected_tickers)} rejected tickers to {os.path.join(output_dir, 'rejected_tickers.txt')}")
                
    except Exception as e:
        display_log(f"Error outputting rejected tickers: {e}")
        # Ensure the directory exists and write an empty file
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, 'rejected_tickers.txt'), 'w') as f:
            f.write("# Error occurred when identifying rejected tickers\n")

def winsorize_scores(scores):
    """
    Winsorize scores to [-3, 3] range.
    
    Args:
        scores (numpy.ndarray): Scores
        
    Returns:
        numpy.ndarray: Winsorized scores
    """
    return np.clip(scores, -3.0, 3.0)

def fix_sectors(data_table):
    """
    Filter data to proper sectors.
    
    Args:
        data_table (pandas.DataFrame): Data table
        
    Returns:
        pandas.DataFrame: Filtered data table
    """
    proper_sectors = ["Basic Materials", "Communication Services", "Consumer Cyclical", "Consumer Defensive",
                      "Energy", "Financial Services", "Healthcare", "Industrials", "Real Estate", "Technology",
                      "Utilities", "Other"]
    return data_table[data_table["Sector"].isin(proper_sectors)]

def scale_values(values, target_min, target_max):
    """
    Scale values to target range.
    
    Args:
        values (numpy.ndarray): Values to scale
        target_min (float): Target minimum
        target_max (float): Target maximum
        
    Returns:
        numpy.ndarray: Scaled values
    """
    min_value = values.min()
    max_value = values.max()
    if max_value == min_value:
        return np.full_like(values, (target_min + target_max) / 2)
    return (values - min_value) / (max_value - min_value) * (target_max - target_min) + target_min

def standardize_scores(data_table, factors, by_group=None):
    """
    Standardize scores for factors.
    
    Args:
        data_table (pandas.DataFrame): Data table
        factors (list): List of factors to standardize
        by_group (str): Column to group by
        
    Returns:
        pandas.DataFrame: Data table with standardized scores
    """
    display_log("Standardizing scores...")
    result = data_table.copy()
    
    for factor in factors:
        new_col = factor.replace('fct', 'Z-Scores').replace('raw', 'Z-Scores')
        
        if by_group and by_group in data_table.columns:
            # Standardize within each group
            groups = data_table[by_group].unique()
            for group in groups:
                mask = data_table[by_group] == group
                valid_data = data_table.loc[mask, factor].dropna()
                if not valid_data.empty:
                    mean = valid_data.mean()
                    std = valid_data.std()
                    if std > 0:  # Avoid division by zero
                        scores = (data_table.loc[mask, factor] - mean) / std
                        result.loc[mask, new_col] = winsorize_scores(scores)
        else:
            # Standardize across all data
            valid_data = data_table[factor].dropna()
            if not valid_data.empty:
                mean = valid_data.mean()
                std = valid_data.std()
                if std > 0:  # Avoid division by zero
                    scores = (data_table[factor] - mean) / std
                    result[new_col] = winsorize_scores(scores)
    
    return result

def standardize_simple_scores(data_table):
    """
    Standardize simple scores.
    
    Args:
        data_table (pandas.DataFrame): Data table
        
    Returns:
        pandas.DataFrame: Data table with standardized scores
    """
    display_log("Standardizing simple scores...")
    simple_factors = ["Institutional Holders fct", "Momentum fct", "1Month fct", "Size raw", "Return Volatility fct",
                      "Dollar Volume fct", "Turnover fct", "Trend Clarity fct"]
    
    # Copy the DataFrame to avoid modifying the original
    result = data_table.copy()
    
    for factor in simple_factors:
        if factor in ["Institutional Holders fct", "Turnover fct"]:
            # Standardize only for stocks
            mask = data_table["Security Type fct"] == "STOCK"
            valid_data = data_table.loc[mask, factor].dropna()
            if not valid_data.empty:
                mean = valid_data.mean()
                std = valid_data.std()
                result.loc[mask, factor[:-3] + "Z-Scores"] = winsorize_scores(
                    (data_table.loc[mask, factor] - mean) / std)
        else:
            valid_data = data_table[factor].dropna()
            if not valid_data.empty:
                mean = valid_data.mean()
                std = valid_data.std()
                result[factor[:-3] + "Z-Scores"] = winsorize_scores(
                    (data_table[factor] - mean) / std)
    
    return result

def scores_post_processing(data_table):
    """
    Post-process scores.
    
    Args:
        data_table (pandas.DataFrame): Data table
        
    Returns:
        pandas.DataFrame: Processed data table
    """
    result = data_table.copy()
    
    # Calculate trailing amount
    result["Trailing Amt"] = result["Stop Vol fct"] * 100
    
    # Calculate Max ROC factor and Z-Scores
    result["Max ROC fct"] = 1 / abs(result["Max ROC raw"].mean() - result["Max ROC raw"])
    result["Max ROC Z-Scores"] = scale_values(result["Max ROC fct"], -3, 3)
    
    # Invert Return Volatility Z-Scores
    result["Return Volatility Z-Scores"] *= -1
    
    # Calculate Illiquidity Z-Scores
    result["Illiquidity Z-Scores"] = result["Illiquidity fct"] * -1 * scale_values(result["Size raw"], -1, 1)
    result["Illiquidity Z-Scores"] = scale_values(result["Illiquidity Z-Scores"], -3, 3)
    
    # Calculate stop price and limit offset
    result["Stop Px"] = result["Price"] - (result["Stop Vol fct"] * result["Price"])
    result["Limit Offset"] = result["Stop Px"] * 0.005
    
    # Reorder columns
    important_columns = ["Ticker", "Currency", "PrimaryExchange", "Trailing Amt", "Stop Px", "Limit Offset"]
    return result[[col for col in result.columns if col not in important_columns] + important_columns]

def filter_invalid_data(data_table):
    """
    Filter out invalid data.
    
    Args:
        data_table (pandas.DataFrame): Data table
        
    Returns:
        pandas.DataFrame: Filtered data table
    """
    display_log("Filtering invalid data...")
    
    # Filter out stocks with missing required data, keep funds
    required_cols = ["Institutional Holders fct", "Turnover fct", "Revenue fct"]
    filtered_data = data_table[
        ~((data_table["Security Type fct"] == "STOCK") &
          (data_table[required_cols].isna().any(axis=1)))
    ]
    
    return filtered_data

def volatility_analysis_vectorized(price_data, is_israel_stock=False):
    """
    Vectorized implementation of volatility analysis.
    
    Args:
        price_data (pandas.DataFrame): Price data
        is_israel_stock (bool): Whether the stock is from Israel
        
    Returns:
        tuple: (volatility_momentum, volatility_stop)
    """
    # Convert dates if not already datetime
    if not pd.api.types.is_datetime64_any_dtype(price_data['Date']):
        price_data['Date'] = pd.to_datetime(price_data['Date'])
    
    # Extract weekly returns using vectorized operations
    weekday = 3 if is_israel_stock else 4
    weekly_mask = price_data['Date'].dt.weekday == weekday
    weekly_prices = price_data.loc[weekly_mask, 'Adjusted_close']
    weekly_returns = weekly_prices.pct_change().dropna()
    
    # Calculate 3-year volatility (assuming 156 weeks)
    volatility_3y = weekly_returns.tail(156).std() * np.sqrt(52)
    
    # Calculate 1-month volatility for stop loss
    last_month = price_data['Adjusted_close'].tail(21)
    volatility_stop = last_month.std() / last_month.iloc[-1] * 3
    
    return volatility_3y, volatility_stop

def calculate_momentum_vectorized(stock_prices, riskfree_rate):
    """
    Vectorized implementation of momentum calculation.
    
    Args:
        stock_prices (pandas.Series): Stock prices
        riskfree_rate (float): Risk-free rate
        
    Returns:
        tuple: (momentum_returns, momentum_1_month, max_roc, volatility_3y)
    """
    # Calculate returns for different periods
    if len(stock_prices) < 252:
        return None  # Not enough data
    
    # Calculate volatility
    returns = stock_prices.pct_change().dropna()
    volatility_3y = returns.std() * np.sqrt(252)
    
    # Get relevant price points
    price_1y_ago = stock_prices.iloc[-252]
    price_6m_ago = stock_prices.iloc[-126]
    price_1m_ago = stock_prices.iloc[-21]
    current_price = stock_prices.iloc[-1]
    
    # Calculate momentum metrics
    momentum_12_month = (((price_1m_ago / price_1y_ago) - 1) - riskfree_rate) / volatility_3y
    momentum_6_month = (((price_1m_ago / price_6m_ago) - 1) - riskfree_rate) / volatility_3y
    momentum_returns = (momentum_12_month + momentum_6_month) / 2
    momentum_1_month = ((current_price / price_1m_ago) - 1)
    
    # Calculate max average ROC for top 10 days in last month
    max_roc = returns.tail(21).nlargest(10).mean()
    
    return momentum_returns, momentum_1_month, max_roc, volatility_3y