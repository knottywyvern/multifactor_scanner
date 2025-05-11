"""
API functions for the Market Scanner.
"""
import os
import logging
import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
import pandas as pd
import io
import asyncio
import aiohttp
import numpy as np

from .utils import display_log, load_json_data, load_tickers_from_file
from .rate_limiter import global_rate_limiter, rate_limited_api_call

def create_session():
    """
    Create a requests session with retry logic.
    
    Returns:
        requests.Session: Session with retry logic
    """
    session = requests.Session()
    retry = Retry(total=7, backoff_factor=1, status_forcelist=[502, 503, 504])
    adapter = HTTPAdapter(max_retries=retry)
    session.mount('https://', adapter)
    return session

def get_indices(config):
    """
    Get index tickers to analyze.
    
    Args:
        config (dict): Application configuration
        
    Returns:
        list: List of index tickers
    """
    index_file = config["paths"]["index_inclusion"]
    if os.path.exists(index_file):
        with open(index_file) as f:
            index_inclusion_list = [x.strip() for x in f.readlines()]
        display_log(f"Including index data of the following tickers: {index_inclusion_list}")
    else:
        default_index = config.get("default_index", "DIA.US")
        logging.warning(f"No index_inclusion.txt file found. Using {default_index} as the default index.")
        index_inclusion_list = [default_index]
    return index_inclusion_list

def canada_check(url, ticker):
    """
    Handle Canadian tickers special case because EOD is weird.
    
    Args:
        url (str): API URL
        ticker (str): Ticker symbol
        
    Returns:
        bytes: API response content
    """
    session = create_session()
    if ticker.endswith('.TO'):
        url_un = url.replace(ticker, ticker[:-3] + "-UN.TO")
        response = session.get(url_un).content
        if response.decode("utf-8") in ["Ticker Not Found.", "Symbol not found"]:
            response = session.get(url).content
    else:
        response = session.get(url).content
    return response

async def async_canada_check(session, url, ticker):
    """
    Async version of canada_check with rate limiting.
    
    Args:
        session (aiohttp.ClientSession): Async session
        url (str): API URL
        ticker (str): Ticker symbol
        
    Returns:
        tuple: (response content, ticker)
    """
    # Apply rate limiting
    await global_rate_limiter.acquire()
    try:
        if ticker.endswith('.TO'):
            url_un = url.replace(ticker, ticker[:-3] + "-UN.TO")
            async with session.get(url_un) as response:
                if response.status == 200:
                    content = await response.read()
                    text = content.decode("utf-8")
                    if text not in ["Ticker Not Found.", "Symbol not found"]:
                        return content, ticker
        
        async with session.get(url) as response:
            if response.status == 200:
                content = await response.read()
                return content, ticker
            else:
                logging.warning(f"Failed to get data for {ticker}: HTTP {response.status}")
                return None, ticker
    finally:
        global_rate_limiter.release()

def get_fundamental_data(ticker, token):
    """
    Fetch fundamental data for a ticker.
    
    Args:
        ticker (str): Ticker symbol
        token (str): API token
        
    Returns:
        bytes: Fundamental data
    """
    display_log(f"Getting {ticker} fundamental data.")
    url = f"https://eodhistoricaldata.com/api/fundamentals/{ticker}?api_token={token}"
    response = canada_check(url, ticker)
    if not response or response.decode("utf-8") in ["Ticker Not Found.", "Symbol not found"]:
        display_log(f"No fundamental data for {ticker}.")
        return None
    return response

async def get_fundamental_data_async(tickers, token):
    """
    Fetch fundamental data for multiple tickers concurrently with rate limiting.
    
    Args:
        tickers (list): List of ticker symbols
        token (str): API token
        
    Returns:
        dict: Dictionary mapping tickers to fundamental data
    """
    display_log(f"Getting fundamental data for {len(tickers)} tickers concurrently (rate limited to 750/min).")
    results = {}
    
    async with aiohttp.ClientSession() as session:
        tasks = []
        for ticker in tickers:
            url = f"https://eodhistoricaldata.com/api/fundamentals/{ticker}?api_token={token}"
            # We don't need to rate limit here since async_canada_check is already rate limited
            tasks.append(async_canada_check(session, url, ticker))
        
        for future in asyncio.as_completed(tasks):
            content, ticker = await future
            if content and content.decode("utf-8") not in ["Ticker Not Found.", "Symbol not found"]:
                results[ticker] = content
            else:
                display_log(f"No fundamental data for {ticker}.")
    
    display_log(f"Retrieved fundamental data for {len(results)} tickers.")
    return results

def get_timeseries_data(ticker, token, days_lookback=3650):
    """
    Fetch timeseries data for a ticker.
    
    Args:
        ticker (str): Ticker symbol
        token (str): API token
        days_lookback (int): Number of days to look back
        
    Returns:
        pandas.DataFrame: Timeseries data
    """
    display_log(f"Getting {ticker} timeseries data.")
    import datetime as dt
    current_date = dt.datetime.today()
    
    # Use provided days_lookback parameter (defaults to 10 years)
    timeseries_days_lookback = days_lookback
    timeseries_start_date = current_date - dt.timedelta(days=timeseries_days_lookback)
    grab_start_date = timeseries_start_date.strftime("%Y-%m-%d")
    
    url = f"https://eodhistoricaldata.com/api/eod/{ticker}?from={grab_start_date}&api_token={token}"
    response_content = canada_check(url, ticker)
    if not response_content:
        display_log(f"No timeseries data for {ticker}.")
        return None
    
    try:
        timeseries_df = pd.read_csv(io.StringIO(response_content.decode("utf-8")))
        timeseries_df["ROC"] = timeseries_df["Adjusted_close"].pct_change()
        return timeseries_df
    except Exception as e:
        display_log(f"Failed to process timeseries data for {ticker}: {e}")
        return None

async def get_timeseries_data_async(tickers, token, start_date):
    """
    Fetch timeseries data for multiple tickers concurrently with rate limiting.
    
    Args:
        tickers (list): List of ticker symbols
        token (str): API token
        start_date (str): Start date in YYYY-MM-DD format
        
    Returns:
        dict: Dictionary mapping tickers to timeseries DataFrames
    """
    display_log(f"Getting timeseries data for {len(tickers)} tickers concurrently (rate limited to 750/min).")
    results = {}
    
    async with aiohttp.ClientSession() as session:
        tasks = []
        for ticker in tickers:
            url = f"https://eodhistoricaldata.com/api/eod/{ticker}?from={start_date}&api_token={token}"
            # We don't need to rate limit here since async_canada_check is already rate limited
            tasks.append(async_canada_check(session, url, ticker))
        
        for future in asyncio.as_completed(tasks):
            content, ticker = await future
            if content:
                try:
                    timeseries_df = pd.read_csv(io.StringIO(content.decode("utf-8")))
                    timeseries_df["ROC"] = timeseries_df["Adjusted_close"].pct_change()
                    results[ticker] = timeseries_df
                except Exception as e:
                    display_log(f"Failed to process timeseries data for {ticker}: {e}")
            else:
                display_log(f"No timeseries data for {ticker}.")
    
    display_log(f"Retrieved timeseries data for {len(results)} tickers.")
    return results

@global_rate_limiter  # This uses the rate limiter as a decorator
async def fetch_website_data_async(url):
    """
    Fetch HTML table data from a website with rate limiting.
    
    Args:
        url (str): Website URL
        
    Returns:
        pandas.DataFrame: Table data
    """
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                response.raise_for_status()
                content = await response.read()
                return pd.read_html(content)[0]
    except aiohttp.ClientError as e:
        logging.error(f"Failed to fetch website data from {url}: {e}")
        raise ConnectionError(f"Network error: {e}")
    except ValueError as e:
        logging.error(f"No tables found at {url}: {e}")
        raise ValueError(f"Table parsing error: {e}")

def fetch_website_data(url):
    """
    Fetch HTML table data from a website with rate limiting.
    
    Args:
        url (str): Website URL
        
    Returns:
        pandas.DataFrame: Table data
    """
    try:
        response = requests.get(url)
        response.raise_for_status()
        return pd.read_html(response.content)[0]
    except requests.exceptions.RequestException as e:
        logging.error(f"Failed to fetch website data from {url}: {e}")
        raise ConnectionError(f"Network error: {e}")
    except ValueError as e:
        logging.error(f"No tables found at {url}: {e}")
        raise ValueError(f"Table parsing error: {e}")

def extract_riskfree_rates(table, benchmark):
    """
    Extract risk-free rates from table.
    
    Args:
        table (pandas.DataFrame): Table with rates
        benchmark (pandas.DataFrame): Benchmark data
        
    Returns:
        list: List of risk-free rates
    """
    table = table.replace({"United Kingdom": "UK", "United States": "USA"})
    rfr_list = []
    for country in benchmark["Country"]:
        try:
            rate_str = table.loc[table["Country▴"] == country, "Central Bank Rate"].values[0]
            rate = float(rate_str[:-2]) / 100
            rfr_list.append(rate)
        except IndexError:
            logging.warning(f"Rate not found for country: {country}")
            rfr_list.append(np.nan)
    return rfr_list

def get_riskfree_rate_from_website(region_data):
    """
    Get risk-free rates from website until I figure out a better way to get rates.
    
    Args:
        region_data (pandas.DataFrame): Region data
        
    Returns:
        pandas.DataFrame: Region data with rates
    """
    website_url = "http://www.worldgovernmentbonds.com/central-bank-rates/"
    try:
        table_dataframe = fetch_website_data(website_url)
        rfr_list = extract_riskfree_rates(table_dataframe, region_data)
        region_data = region_data.copy()
        region_data["Rate"] = rfr_list
        return region_data
    except (ValueError, ConnectionError) as e:
        logging.error(f"Error fetching risk-free rate data: {e}")
        return None

async def get_riskfree_rate(region_data):
    """
    Get risk-free rates. :LD
    
    Args:
        region_data (pandas.DataFrame): Region data
        
    Returns:
        pandas.DataFrame: Region data with rates
    """
    display_log("Getting risk-free rates.")
    
    try:
        # Import the EOD rate manager
        from .eod_rate_manager import EODRateManager
        
        # Load API key
        with open(config["paths"]["license_file"], "r") as license_file:
            api_key = license_file.read().strip()
        
        # Create rate manager
        rate_manager = EODRateManager(api_key, data_dir=config["paths"].get("data_dir", "data"))
        
        # Update region data with rates
        rate_dataframe = rate_manager.update_region_data(region_data)
        
        # Save updated region data
        rate_dataframe.to_csv(os.path.join(config["paths"].get("data_dir", "data"), "regions_data_with_rates.csv"), index=False)
        
        return rate_dataframe
        
    except Exception as e:
        logging.error(f"Error getting risk-free rates: {e}")
        
        # Try to load previously saved rates
        try:
            rates_file = os.path.join(config["paths"].get("data_dir", "data"), "regions_data_with_rates.csv")
            if os.path.exists(rates_file):
                logging.info("Loading risk-free rates from local file")
                rate_dataframe = pd.read_csv(rates_file)
                return rate_dataframe
        except Exception as e2:
            logging.error(f"Error loading local risk-free rates: {e2}")
        
        # If all else fails, return the original data with default rates
        logging.warning("Using default risk-free rates (3%)")
        rate_dataframe = region_data.copy()
        
        # If the Rate column doesn't exist, add it
        if 'Rate' not in rate_dataframe.columns:
            rate_dataframe['Rate'] = 0.03
        else:
            # Fill any missing rates with default
            rate_dataframe['Rate'] = rate_dataframe['Rate'].fillna(0.03)
        
        return rate_dataframe

async def fetch_tickers_from_market_index(market_index, token):
    """
    Fetch tickers from market index.
    
    Args:
        market_index (list): List of index tickers
        token (str): API token
        
    Returns:
        set: Set of tickers
    """
    ticker_list = set()
    fundamental_data = await get_fundamental_data_async(market_index, token)
    
    for ticker, response in fundamental_data.items():
        try:
            if response:
                table_dataframe = pd.read_json(response.decode('utf8'), orient="index")
                holdings = table_dataframe["Holdings"].iloc[2]
                ticker_list.update(pd.DataFrame.from_dict(holdings, orient="index").index.values)
        except Exception as e:
            display_log(f"Error processing {ticker}: {e}")
    
    return ticker_list

async def vix_table(region_data, token):
    """
    Create VIX tables for regional volatility.
    
    Args:
        region_data (pandas.DataFrame): Region data
        token (str): API token
        
    Returns:
        dict: Dictionary mapping index to timeseries data
    """
    vol_index_dict = {}
    unique_indices = set(region_data["Vol Index"])
    display_log(f"Getting VIX data for {len(unique_indices)} indices.")
    
    # Get timeseries data for each volatility index
    timeseries_data = await get_timeseries_data_async(
        list(unique_indices), 
        token, 
        (pd.Timestamp.now() - pd.Timedelta(days=252)).strftime("%Y-%m-%d")
    )
    
    for index, data in timeseries_data.items():
        if data is not None:
            data = data.tail(252)
            data["Var"] = np.nanvar(data["ROC"])
            vol_index_dict[index] = data
    
    return vol_index_dict