"""
Factor analysis functions for the Market Scanner.
"""
import pandas as pd
import numpy as np
import math
import statsmodels.api as sm
import logging

from .data import TickerAnalysis
from .utils import display_log, load_json_data, extract_currency_code, map_currency_code

def date_string_to_datetime(dataframe):
    """
    Convert date strings to datetime objects.
    
    Args:
        dataframe (pandas.DataFrame): DataFrame with Date column
        
    Returns:
        pandas.DataFrame: DataFrame with converted Date column
    """
    if "Date" in dataframe.columns:
        dataframe["Date"] = pd.to_datetime(dataframe["Date"])
    return dataframe

def extract_weekly_returns(is_israel_stock, price_data):
    """
    Extract weekly returns from price data.
    
    Args:
        is_israel_stock (bool): Whether the stock is from Israel
        price_data (pandas.DataFrame): Price data
        
    Returns:
        pandas.Series: Weekly returns
    """
    weekday = 3 if is_israel_stock else 4
    weekly = price_data[price_data["Date"].dt.weekday == weekday]
    weekly_returns = (weekly["Adjusted_close"].shift(-1) / weekly["Adjusted_close"]) - 1
    return weekly_returns[:-1]

def calculate_volatility(returns, period=52):
    """
    Calculate volatility from returns.
    
    Args:
        returns (pandas.Series): Returns
        period (int): Annualization period
        
    Returns:
        float: Volatility
    """
    return np.std(returns) * math.sqrt(period)

def volatility_analysis(ticker, is_israel_stock, price_data):
    """
    Perform volatility factor analysis.
    
    Args:
        ticker (str): Ticker symbol
        is_israel_stock (bool): Whether the stock is from Israel
        price_data (pandas.DataFrame): Price data
        
    Returns:
        tuple: (volatility_momentum, volatility_stop) or "drop_ticker" on error
    """
    display_log(f"Performing {ticker} volatility factor analysis...")
    try:
        # Make sure price_data has sufficient length
        if len(price_data) < 21:
            display_log(f"Insufficient data for {ticker} volatility analysis (need at least 21 days)")
            return 0.2, 0.05  # Default values: 20% volatility, 5% stop
            
        # Convert Date to datetime if needed
        price_data = date_string_to_datetime(price_data)
        
        # Extract weekly returns
        try:
            weekly_returns = extract_weekly_returns(is_israel_stock, price_data)
            if len(weekly_returns) < 10:
                display_log(f"Insufficient weekly data for {ticker} volatility analysis")
                # Fall back to daily returns if weekly data is insufficient
                daily_returns = price_data["Adjusted_close"].pct_change().dropna()
                volatility_momentum = calculate_volatility(daily_returns, period=252) if len(daily_returns) > 20 else 0.2
            else:
                volatility_momentum = calculate_volatility(weekly_returns.tail(156))
        except Exception as e:
            display_log(f"Error calculating volatility momentum for {ticker}: {e}")
            volatility_momentum = 0.2  # Default 20% annual volatility
        
        # Calculate stop volatility
        try:
            # Get last month's prices
            last_month = price_data["Adjusted_close"].tail(21).to_list()
            if len(last_month) < 21:
                display_log(f"Insufficient recent data for {ticker} stop volatility")
                volatility_stop = 0.05  # Default 5% stop
            else:
                volatility_stop = np.std(last_month) / last_month[-1] * 3
                
                # Sanity check on volatility stop
                if volatility_stop > 0.25:  # Cap at 25%
                    display_log(f"Capping excessive stop volatility for {ticker}: {volatility_stop:.2%} -> 25%")
                    volatility_stop = 0.25
                elif volatility_stop < 0.01:  # Floor at 1%
                    display_log(f"Increasing too low stop volatility for {ticker}: {volatility_stop:.2%} -> 1%")
                    volatility_stop = 0.01
        except Exception as e:
            display_log(f"Error calculating volatility stop for {ticker}: {e}")
            volatility_stop = 0.05  # Default 5% stop
        
        return volatility_momentum, volatility_stop
    
    except Exception as e:
        display_log(f"Error in volatility analysis for {ticker}: {e}")
        return 0.2, 0.05  # Default values

def calculate_momentum(stock_prices, riskfree_rate, volatility_3y):
    """
    Calculate momentum factors.
    
    Args:
        stock_prices (list): List of stock prices
        riskfree_rate (float): Risk-free rate
        volatility_3y (float): 3-year volatility
        
    Returns:
        tuple: (momentum_returns, momentum_1_month) or "drop_ticker" on error
    """
    try:
        momentum_12_month = (((stock_prices[-21] / stock_prices[-252]) - 1) - riskfree_rate) / volatility_3y
        momentum_6_month = (((stock_prices[-21] / stock_prices[-126]) - 1) - riskfree_rate) / volatility_3y
        momentum_returns = (momentum_12_month + momentum_6_month) / 2
        momentum_1_month = ((stock_prices[-1] / stock_prices[-21]) - 1)
        return momentum_returns, momentum_1_month
    except Exception as e:
        display_log(f"Error calculating momentum: {e}")
        return "drop_ticker"

def momentum_analysis(ticker, stock_price_data, fund_data, riskfree_rate_table):
    """
    Perform momentum factor analysis.
    
    Args:
        ticker (str): Ticker symbol
        stock_price_data (pandas.DataFrame): Stock price data
        fund_data: Fundamental data
        riskfree_rate_table (pandas.DataFrame): Risk-free rate table
        
    Returns:
        list: [momentum_returns, volatility_3y, volatility_stop, momentum_1_month, max_roc]
              or ["drop_ticker"] * 5 on error
    """
    display_log(f"Performing {ticker} momentum factor analysis...")
    if stock_price_data is None:
        return ["drop_ticker"] * 5
    try:
        fx_code = map_currency_code(extract_currency_code(fund_data))
        riskfree_rate = riskfree_rate_table[riskfree_rate_table["Currency"] == fx_code]["Rate"].iloc[0]
        stock_prices_tail = stock_price_data.tail(882)
        stock_prices_tail = date_string_to_datetime(stock_prices_tail)
        if len(stock_prices_tail) <= 147:
            display_log(f"{ticker} does not have enough timeseries data.")
            return ["drop_ticker"] * 5
        is_israel_stock = fx_code == 'ILS'
        volatility_result = volatility_analysis(ticker, is_israel_stock, stock_prices_tail)
        if volatility_result == "drop_ticker":
            return ["drop_ticker"] * 5
        volatility_3y, volatility_stop = volatility_result
        stock_prices_list = stock_prices_tail["Adjusted_close"].tolist()
        while len(stock_prices_list) < 1260:
            stock_prices_list.insert(0, stock_prices_list[0])
        momentum_result = calculate_momentum(stock_prices_list, riskfree_rate, volatility_3y)
        if momentum_result == "drop_ticker":
            return ["drop_ticker"] * 5
        momentum_returns, momentum_1_month = momentum_result
        max_roc = stock_prices_tail["ROC"].tail(21).nlargest(10).mean()
        return [momentum_returns, volatility_3y, volatility_stop, momentum_1_month, max_roc]
    except Exception as e:
        display_log(f"Error in momentum analysis for {ticker}: {e}")
        return ["drop_ticker"] * 5

def trend_analysis(ticker, stock_price_data):
    """
    Perform trend analysis.
    
    Args:
        ticker (str): Ticker symbol
        stock_price_data (pandas.DataFrame): Stock price data
        
    Returns:
        float: R-squared or "drop_ticker" on error
    """
    display_log(f"Performing {ticker} trend analysis...")
    if stock_price_data is None:
        return "drop_ticker"
    try:
        stock_price_data = stock_price_data.tail(252)
        stock_price_data["time"] = np.arange(len(stock_price_data))
        X = sm.add_constant(stock_price_data["time"])
        model = sm.OLS(stock_price_data["Adjusted_close"], X).fit()
        return model.rsquared
    except Exception as e:
        display_log(f"Error in trend analysis for {ticker}: {e}")
        return "drop_ticker"

def adjust_forex(fx_code, forex_code_list, forex_rate_list):
    """
    Adjust for forex rates.
    
    Args:
        fx_code (str): Currency code
        forex_code_list (list): List of forex codes
        forex_rate_list (list): List of forex rates
        
    Returns:
        float: Adjusted forex rate
    """
    if fx_code not in forex_code_list:
        forex_code_list.append(fx_code)
        # This function needs to be updated to use the async API
        # For now, we'll just return 1.0 for simplicity
        forex_rate_list.append(1.0)
        return 1.0
    else:
        fx_rate = forex_rate_list[forex_code_list.index(fx_code)]
        return fx_rate

def size_analysis(ticker, fund_data, forex_code_list, forex_rate_list):
    """
    Perform size factor analysis.
    
    Args:
        ticker (str): Ticker symbol
        fund_data: Fundamental data
        forex_code_list (list): List of forex codes
        forex_rate_list (list): List of forex rates
        
    Returns:
        float: Size factor or "drop_ticker" on error
    """
    display_log(f"Performing {ticker} size factor analysis...")
    try:
        fund_data_dict = load_json_data(fund_data)
        general_data = fund_data_dict.get("General", {})
        fx_code = map_currency_code(extract_currency_code(fund_data))
        if general_data.get("Type") == "ETF":
            market_cap = fund_data_dict.get("ETF_Data", {}).get("TotalAssets", np.nan)
        elif general_data.get("Type") == "FUND":
            market_cap = fund_data_dict.get("MutualFund_Data", {}).get("Portfolio_Net_Assets", np.nan)
        else:
            market_cap = fund_data_dict.get("Highlights", {}).get("MarketCapitalization", np.nan)
        if pd.isna(market_cap) or market_cap == 0:
            return "drop_ticker"
        market_cap_base = float(market_cap) / adjust_forex(fx_code, forex_code_list, forex_rate_list)
        return 1 / math.log(market_cap_base)
    except Exception as e:
        display_log(f"Error in size analysis for {ticker}: {e}")
        return "drop_ticker"

def calculate_return_volatility(ticker, stock_price_data, fund_data, region_data):
    """
    Calculate return volatility factor.
    
    Args:
        ticker (str): Ticker symbol
        stock_price_data (pandas.DataFrame): Stock price data
        fund_data: Fundamental data
        region_data (pandas.DataFrame): Region data
        
    Returns:
        float: Return volatility beta or "drop_ticker" on error
    """
    display_log(f"Performing {ticker} return volatility factor analysis...")
    if stock_price_data is None:
        return "drop_ticker"
    try:
        # Get country name from fundamental data
        general_data = load_json_data(fund_data).get("General", {})
        country_name = general_data.get("CountryName")
        if country_name is None:
            raise ValueError("CountryName not found.")
        
        # Find matching country in region_data
        index_number = region_data.index[region_data["Country"] == country_name].tolist()
        if not index_number:
            # Try a fallback if exact match not found
            display_log(f"Country '{country_name}' not found in region_data. Using USA as fallback.")
            index_number = region_data.index[region_data["Country"] == "USA"].tolist()
            if not index_number:
                raise ValueError("Country not found in region_data and USA fallback not available.")
        
        # Get volatility data
        vol_table = region_data["Vol Table"][index_number[0]]
        vol_roc = vol_table["ROC"]
        vol_var = vol_table["Var"]
        
        # Get stock return data
        stock_roc = stock_price_data["ROC"].tail(252).dropna()
        
        # Clean volatility ROC data
        vol_roc_clean = vol_roc.dropna()
        
        # Handle dimension mismatch by truncating to the shorter length
        min_length = min(len(vol_roc_clean), len(stock_roc))
        if min_length < 20:
            display_log(f"Warning: Very few data points for volatility calculation ({min_length})")
            if min_length < 10:
                return 1.0  # Default beta of 1.0 if too few points
        
        # Use the latest data points from both series
        stock_roc_used = stock_roc.iloc[-min_length:].values
        vol_roc_used = vol_roc_clean.iloc[-min_length:].values
        
        # Calculate covariance and beta
        covariance = np.cov(stock_roc_used, vol_roc_used)[0, 1]
        
        # Handle vol_var potentially being a Series or scalar
        if isinstance(vol_var, (pd.Series, np.ndarray)):
            vol_var_value = vol_var.iloc[0] if len(vol_var) > 0 else np.var(vol_roc_used)
        else:
            vol_var_value = vol_var
            
        beta = covariance / vol_var_value
        
        # Handle beta potentially being a Series, array, or scalar
        if isinstance(beta, (pd.Series, list, np.ndarray)):
            if len(beta) > 0:
                return beta[-1]  # Return the last element
            else:
                return 1.0  # Default if empty
        else:
            return beta  # Return as scalar
            
    except Exception as e:
        display_log(f"Error in return volatility for {ticker}: {e}")
        return 1.0  # Return a default beta of 1.0 instead of dropping the ticker

def liquidity_analysis(ticker, stock_price_data, fund_data, forex_code_list, forex_rate_list):
    """
    Perform liquidity analysis.
    
    Args:
        ticker (str): Ticker symbol
        stock_price_data (pandas.DataFrame): Stock price data
        fund_data: Fundamental data
        forex_code_list (list): List of forex codes
        forex_rate_list (list): List of forex rates
        
    Returns:
        list: [dvol, turnover, ill, is_fund] or ["drop_ticker"] * 4 on error
    """
    display_log(f"Performing {ticker} liquidity analysis...")
    if stock_price_data is None:
        return ["drop_ticker"] * 4
    try:
        fx_code = map_currency_code(extract_currency_code(fund_data))
        stock_prices_tail = stock_price_data.tail(252)
        prices = stock_prices_tail["Adjusted_close"].tolist()
        volumes = stock_prices_tail["Volume"].tolist()
        returns = stock_prices_tail["ROC"].tolist()
        if len(prices) < 252 or len(volumes) < 252 or len(returns) < 252:
            raise ValueError("Insufficient data.")
        per_day_dvol = [close * vol for close, vol in zip(prices, volumes)]
        fx_rate = adjust_forex(fx_code, forex_code_list, forex_rate_list)
        dvol = sum(per_day_dvol[-42:-21]) / fx_rate / 1000000
        ill = (sum([roc / dvol for roc, dvol in zip(returns, per_day_dvol)]) / 252) * 1000000000
        general_data = load_json_data(fund_data).get("General", {})
        turnover = np.nan if general_data.get("Type") in ["ETF", "FUND"] else None
        if general_data.get("Type") == "Common Stock":
            shares_outstanding = load_json_data(fund_data).get("SharesStats", {}).get("SharesOutstanding")
            if shares_outstanding:
                turnover = sum(volumes) / len(volumes) / shares_outstanding
        is_fund = "FUND" if general_data.get("Type") in ["ETF", "FUND"] else "STOCK"
        return [dvol, turnover, ill, is_fund]
    except Exception as e:
        display_log(f"Error in liquidity analysis for {ticker}: {e}")
        return ["drop_ticker"] * 4

def get_general_data(ticker, fund_data):
    """
    Get general data for a ticker.
    
    Args:
        ticker (str): Ticker symbol
        fund_data: Fundamental data
        
    Returns:
        list: [name, country, sector, industry, exchange, currency, type]
    """
    display_log(f"Getting general {ticker} stock data.")
    fund_data_dict = load_json_data(fund_data)
    general_data = fund_data_dict.get("General", {})
    return [
        general_data.get("Name", np.nan),
        general_data.get("CountryName", np.nan),
        general_data.get("Sector", "Other"),
        general_data.get("Industry", "Other"),
        general_data.get("Exchange", "Other"),
        general_data.get("CurrencyCode", "USD"),
        general_data.get("Type", "Common Stock")  # Default to stock if Type is missing
    ]

def get_share_data(ticker, fund_data, security_type):
    """
    Get share data for a ticker.
    
    Args:
        ticker (str): Ticker symbol
        fund_data: Fundamental data
        security_type (str): Security type
        
    Returns:
        list: [institutional_holders, revenue]
    """
    display_log(f"Getting {ticker} share data.")
    fund_data_dict = load_json_data(fund_data)
    share_data = fund_data_dict.get("SharesStats", {})
    highlights_data = fund_data_dict.get("Highlights", {})
    institutional_holders = share_data.get("PercentInstitutions", np.nan) if security_type == "STOCK" else np.nan
    revenue = highlights_data.get("RevenueTTM", np.nan) if security_type == "STOCK" else np.nan
    return [institutional_holders, revenue]

def price_last_close(ticker, stock_price_data):
    """
    Get last closing price.
    
    Args:
        ticker (str): Ticker symbol
        stock_price_data (pandas.DataFrame): Stock price data
        
    Returns:
        float: Last closing price
    """
    display_log(f"Getting {ticker} EOD price.")
    try:
        if stock_price_data is not None and not stock_price_data.empty:
            return stock_price_data["Adjusted_close"].iloc[-1]
        else:
            return np.nan
    except Exception as e:
        display_log(f"Error getting price for {ticker}: {e}")
        return np.nan

def analyze_ticker(ticker, fund_data, eod_data, riskfree_rate_table, region_data):
    """
    Analyze a ticker and create a TickerAnalysis object.
    
    Args:
        ticker (str): Ticker symbol
        fund_data: Fundamental data
        eod_data (pandas.DataFrame): EOD price data
        riskfree_rate_table (pandas.DataFrame): Risk-free rate table
        region_data (pandas.DataFrame): Region data
        
    Returns:
        TickerAnalysis: Analysis object or None on error
    """
    try:
        # Get general data
        general_data = get_general_data(ticker, fund_data)
        name, country, sector, industry, exchange, currency, security_type = general_data
        
        # Create analysis object
        analysis = TickerAnalysis(
            ticker=ticker,
            name=name,
            country=country,
            sector=sector,
            industry=industry,
            exchange=exchange,
            currency=currency,
            security_type="FUND" if security_type in ["ETF", "FUND"] else "STOCK"
        )
        
        # Get price
        analysis.price = price_last_close(ticker, eod_data)
        
        # Get share data
        institutional_holders, revenue = get_share_data(ticker, fund_data, analysis.security_type)
        analysis.institutional_holders = institutional_holders
        analysis.revenue = revenue
        
        # Create forex lists for use in analyses
        forex_code_list = []
        forex_rate_list = []
        
        # Momentum analysis
        try:
            momentum_data = momentum_analysis(ticker, eod_data, fund_data, riskfree_rate_table)
            if "drop_ticker" not in momentum_data:
                analysis.momentum, analysis.volatility_12m, analysis.volatility_stop, analysis.momentum_1month, analysis.max_roc = momentum_data
            else:
                # Use defaults if momentum analysis fails
                logging.warning(f"Using default momentum values for {ticker}")
                analysis.momentum = 0.0
                analysis.volatility_12m = 0.2  # 20% annual volatility
                analysis.volatility_stop = 0.05  # 5% stop
                analysis.momentum_1month = 0.0
                analysis.max_roc = 0.01  # 1% max ROC
        except Exception as e:
            logging.error(f"Error in momentum analysis for {ticker}: {e}")
            analysis.momentum = 0.0
            analysis.volatility_12m = 0.2
            analysis.volatility_stop = 0.05
            analysis.momentum_1month = 0.0
            analysis.max_roc = 0.01
        
        # Size analysis
        try:
            size_data = size_analysis(ticker, fund_data, forex_code_list, forex_rate_list)
            if size_data != "drop_ticker":
                analysis.size = size_data
            else:
                # Default size based on security type
                if analysis.security_type == "FUND":
                    analysis.size = 0.05  # ETFs/funds usually have more AUM
                else:
                    analysis.size = 0.1  # Default for stocks
        except Exception as e:
            logging.error(f"Error in size analysis for {ticker}: {e}")
            analysis.size = 0.1
        
        # Return volatility analysis
        try:
            rvol_data = calculate_return_volatility(ticker, eod_data, fund_data, region_data)
            if rvol_data != "drop_ticker":
                analysis.return_volatility = rvol_data
            else:
                analysis.return_volatility = 1.0  # Default beta of 1.0
        except Exception as e:
            logging.error(f"Error in return volatility analysis for {ticker}: {e}")
            analysis.return_volatility = 1.0
        
        # Liquidity analysis
        try:
            liquidity_data = liquidity_analysis(ticker, eod_data, fund_data, forex_code_list, forex_rate_list)
            if "drop_ticker" not in liquidity_data:
                analysis.dollar_volume, analysis.turnover, analysis.illiquidity, security_type_check = liquidity_data
                
                # Ensure security_type is consistent
                if analysis.security_type != security_type_check:
                    display_log(f"Warning: Security type mismatch for {ticker}: {analysis.security_type} vs {security_type_check}")
                    analysis.security_type = security_type_check
            else:
                # Default liquidity values
                if analysis.security_type == "FUND":
                    analysis.dollar_volume = 1000000  # $1M daily volume
                    analysis.turnover = 0.01  # 1% turnover
                    analysis.illiquidity = 0.0001  # Low illiquidity for funds
                else:
                    analysis.dollar_volume = 500000  # $500k daily volume
                    analysis.turnover = 0.005  # 0.5% turnover
                    analysis.illiquidity = 0.001  # Higher illiquidity for stocks
        except Exception as e:
            logging.error(f"Error in liquidity analysis for {ticker}: {e}")
            analysis.dollar_volume = 500000
            analysis.turnover = 0.005
            analysis.illiquidity = 0.001
        
        # Trend analysis
        try:
            trend_data = trend_analysis(ticker, eod_data)
            if trend_data != "drop_ticker":
                analysis.trend_clarity = trend_data
            else:
                analysis.trend_clarity = 0.5  # Default R² of 0.5
        except Exception as e:
            logging.error(f"Error in trend analysis for {ticker}: {e}")
            analysis.trend_clarity = 0.5
        
        return analysis
    except Exception as e:
        display_log(f"Failed to analyze {ticker}: {e}")
        return None

def post_process_analysis(market_analysis):
    """
    Post-process market analysis.
    
    Args:
        market_analysis: MarketAnalysis object
        
    Returns:
        MarketAnalysis: Processed MarketAnalysis object
    """
    display_log("Post-processing market analysis...")
    
    # Filter analyses to valid sectors
    proper_sectors = ["Basic Materials", "Communication Services", "Consumer Cyclical", "Consumer Defensive",
                      "Energy", "Financial Services", "Healthcare", "Industrials", "Real Estate", "Technology",
                      "Utilities", "Other"]
    
    filtered_analysis = market_analysis.filter_analyses(
        lambda a: a.sector is None or a.sector in proper_sectors
    )
    
    # Standardize scores
    factors = [
        "institutional_holders", "momentum", "momentum_1month", 
        "size", "return_volatility", "dollar_volume", 
        "turnover", "trend_clarity"
    ]
    
    filtered_analysis.standardize_scores(factors, by_security_type=True)
    
    return filtered_analysis

def calculate_derived_values(market_analysis):
    """
    Calculate derived values for analyses.
    
    Args:
        market_analysis: MarketAnalysis object
        
    Returns:
        MarketAnalysis: MarketAnalysis with calculated derived values
    """
    display_log("Calculating derived values...")
    
    for analysis in market_analysis.analyses:
        # Calculate trailing amount
        if analysis.volatility_stop is not None:
            analysis.trailing_amount = analysis.volatility_stop * 100
        
        # Calculate stop price
        if analysis.price is not None and analysis.volatility_stop is not None:
            analysis.stop_price = analysis.price - (analysis.volatility_stop * analysis.price)
        
        # Calculate limit offset
        if analysis.stop_price is not None:
            analysis.limit_offset = analysis.stop_price * 0.005
        
        # Special processing for Max ROC
        if analysis.max_roc is not None:
            # We need to get the mean of max_roc across all analyses
            # This is a bit tricky since we need all analyses to calculate it
            # For now, we'll just add a placeholder
            analysis.z_scores["max_roc_z"] = 0
    
    # Calculate Max ROC Z-Scores
    max_roc_values = [a.max_roc for a in market_analysis.analyses if a.max_roc is not None]
    if max_roc_values:
        mean_max_roc = np.mean(max_roc_values)
        for analysis in market_analysis.analyses:
            if analysis.max_roc is not None:
                max_roc_factor = 1 / abs(mean_max_roc - analysis.max_roc)
                # Scale to -3 to 3 range
                max_value = max([1 / abs(mean_max_roc - v) for v in max_roc_values])
                min_value = min([1 / abs(mean_max_roc - v) for v in max_roc_values])
                scaled_value = -3 + 6 * (max_roc_factor - min_value) / (max_value - min_value)
                analysis.z_scores["max_roc_z"] = scaled_value
    
    # Adjust Return Volatility Z-Scores (invert)
    for analysis in market_analysis.analyses:
        if "return_volatility_z" in analysis.z_scores:
            analysis.z_scores["return_volatility_z"] *= -1
    
    # Calculate Illiquidity Z-Scores
    for analysis in market_analysis.analyses:
        if analysis.illiquidity is not None and analysis.size is not None:
            # Scale size raw to -1 to 1
            size_values = [a.size for a in market_analysis.analyses if a.size is not None]
            if size_values:
                min_size = min(size_values)
                max_size = max(size_values)
                scaled_size = -1 + 2 * (analysis.size - min_size) / (max_size - min_size)
                
                # Calculate illiquidity score
                illiquidity_score = analysis.illiquidity * -1 * scaled_size
                
                # Scale to -3 to 3
                illiquidity_values = [a.illiquidity * -1 * (-1 + 2 * (a.size - min_size) / (max_size - min_size)) 
                                    for a in market_analysis.analyses 
                                    if a.illiquidity is not None and a.size is not None]
                
                if illiquidity_values:
                    min_ill = min(illiquidity_values)
                    max_ill = max(illiquidity_values)
                    scaled_ill = -3 + 6 * (illiquidity_score - min_ill) / (max_ill - min_ill)
                    analysis.z_scores["illiquidity_z"] = scaled_ill
    
    return market_analysis