import datetime as dt
import logging
import os
import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
import pandas as pd
import io
import json
import numpy as np
import math
import ta as ta

# Elapsed time it took for the script to run.
script_start_time = dt.datetime.now()

# Hard coded parameters.
master_ticker_list = []
current_date = dt.datetime.today()

timeseries_days_lookback = 882 # 3.5 year look back. Length determined by amount of data needed to calculate momentum.
timeseries_start_date = current_date - dt.timedelta(days = timeseries_days_lookback)

# Risk-free rate variables.
forex_code_list = []
forex_rate_list = []

# Factor variables.
new_ticker_list = []
name_list = []
eod_list = []
country_list = []
sector_list = []
industry_list = []
institutional_holders_list = []
momentum_list = []
momentum_1mo_list = []
market_cycle_list = []
y3_vol_list = []
market_cap_list = []
rvol_list = []
max_roc_list = []
dvol_list = []
turn_list = []
ill_list = []
stop_vol_list = []
stop_px_list = []
trailing_amt_list = []
offset_list = []
isolated_ticker = []
primary_exchange_list = []
currency_code_list = []


# Check if we need to be using dev settings.
def is_dev_mode():
    if os.getcwd()[-3:] == "dev":
        logging.basicConfig(filename="scannerlog.txt", level=logging.DEBUG)
    else:
        logging.basicConfig(filename="scannerlog.txt", level=logging.INFO)


# Save and display log lines
def display_log(string):
    logging.info(string)
    print(string)
    

# Creating sessions to reduce API calls .   
def create_session():
    session = requests.Session()
    retry = Retry(total=7, backoff_factor=1, status_forcelist=[502, 503, 504])
    adapter = HTTPAdapter(max_retries=retry)
    session.mount('https://', adapter)
    return session


# In a dev environment it is recommended that FEZ.US and DIA.US ETFs be used for scrapping index data.
# ETFs used in place of index data should be saved in a text file named "index_inclusion.txt".
# If none is found, DIA.US will be used.
# One ETF per line.
# Every ticker needs a country code, e.g. SPY.US.
def get_indices():
    index_file = "parameters/index_inclusion.txt"
    if os.path.exists(index_file):
        with open(index_file) as f:
            index_inclusion_list = [x.strip() for x in f.readlines()]
        display_log(f"Including index data of the following tickers: {index_inclusion_list}")
    else:
        default_index = "DIA.US"
        logging.warning(f"No index_inclusion.txt file found. Using {default_index} as the default index.")
        index_inclusion_list = [default_index]
    return index_inclusion_list


def canada_check(url, ticker):
    session = create_session()
    if ticker.endswith('.TO'):
        url_un = url.replace(ticker, ticker[:-3] + "-UN.TO")
        response = session.get(url_un).content
        if response.decode("utf-8") == "Ticker Not Found." or "Symbol not found":
            response = session.get(url).content
            return response
        else:
            return response
    else:
        response = session.get(url).content
        return response


# Returns a json object filled with fundamental data of the security ticker. Do keep in mind that
# ETFs and individual securities contain different data when called.
def get_fundamental_data(ticker, token):
    display_log(f"Getting {ticker} fundamental data.")
    url = f"https://eodhistoricaldata.com/api/fundamentals/{ticker}?api_token={token}"
    return canada_check(url, ticker)


# For calculating yield factor.
def get_dividend_data(ticker, token):
    display_log(f"Getting {ticker} dividend data.")
    url = f"https://eodhistoricaldata.com/api/div/{ticker}?fmt=json&api_token={token}"
    return canada_check(url, ticker)


# For calculating momentum and volatility.
def get_timeseries_data(ticker, token):
    display_log(f"Getting {ticker} timeseries data.")
    grab_start_date = timeseries_start_date.strftime("%Y-%m-%d")
    url = f"https://eodhistoricaldata.com/api/eod/{ticker}?from={grab_start_date}&api_token={token}"
    timeseries_df = pd.read_csv(io.StringIO(canada_check(url, ticker).decode("utf-8")))
    try:
        timeseries_df["ROC"] = timeseries_df["Adjusted_close"].pct_change()
    except Exception as e:
        display_log(f"Error occured while processing {ticker}: {e}")
        timeseries_df = None
    return timeseries_df


def save_riskfree_rate_to_csv(dataframe):
    dataframe.to_csv("regions_data.csv", index = False)


def get_riskfree_rate(region_data):
    logging.info("Getting risk-free rates.")
    rate_dataframe = get_riskfree_rate_from_website(region_data)
    if rate_dataframe is not None:
        save_riskfree_rate_to_csv(rate_dataframe)
    else:
        rate_dataframe = region_data
    return rate_dataframe


def fetch_website_data(url):
    response = requests.get(url)
    if response.status_code == 200:
        return pd.read_html(response.content)[0]
    else:
        raise ConnectionError("Failed to fetch website data")


def extract_riskfree_rates(table, benchmark):
    rfr_list = []

    for country in benchmark["Country"]:
        row = table.loc[table["Country▴"] == country]["Central Bank Rate"]
        rate = row.astype("str").tolist()[0]
        rate = float(rate[:-2]) / 100
        rfr_list.append(rate)

    return rfr_list


def get_riskfree_rate_from_website(region_data):
    try:
        website_url = "http://www.worldgovernmentbonds.com/central-bank-rates/"
        table_dataframe = fetch_website_data(website_url)

        table_dataframe = table_dataframe.replace("United Kingdom", "UK")
        table_dataframe = table_dataframe.replace("United States", "USA")

        benchmark_dataframe = region_data

        rfr_list = extract_riskfree_rates(table_dataframe, benchmark_dataframe)
        
        benchmark_dataframe["Rate"] = rfr_list
        
        return benchmark_dataframe
    except (ValueError, ConnectionError) as e:
        print(f"Error fetching risk-free rate data: {e}")
        return None
    

def get_ticker_list(marketIndex, token):
    ticker_list = []

    inclusion_file = "parameters/ticker_inclusion.txt"
    exclusion_file = "parameters/ticker_exclusion.txt"

    # Add tickers from market index
    for ticker in marketIndex:
        try:
            response = get_fundamental_data(ticker, token)
            if response:
                table_dataframe = pd.read_json(response.decode('utf8'), orient = "index")
                ticker_list += list(pd.DataFrame.from_dict(table_dataframe["Holdings"].iloc[2], orient = "index").index.values)
        except Exception as e:
            display_log(f"Error occured while processing {ticker}: {e}")

    # Add tickers that the users want that are not part of the index that is being referenced.
    if os.path.exists(inclusion_file):
        display_log("Including specific tickers;")
        with open(inclusion_file) as f:
            ticker_inclusion_list = [x.strip() for x in f.readlines() if x.strip() != ""]
            ticker_list += ticker_inclusion_list
            display_log(ticker_inclusion_list)

    # Tickers like ERIC B.ST are not returning the correct ticker data.
    ticker_list = list(set(ticker_list))
    ticker_list = [ticker.replace(' ', '-') for ticker in ticker_list]

    # Tickers like FAE/D.MC? Why does our vendor list tickers like this?
    ticker_list = [ticker.replace('/', '-') for ticker in ticker_list]

    # Remove tickers that the users do not want to analyze.
    if os.path.exists(exclusion_file):
        display_log("Excluding specific tickers;")
        with open(exclusion_file) as f:
            ticker_exclusion_list = [x.strip() for x in f.readlines() if x.strip() != ""]
            for ticker in ticker_exclusion_list:
                if ticker in ticker_list:
                    ticker_list.remove(ticker)
            display_log(ticker_exclusion_list)

    ticker_ex_exchange = [ticker.split(".", 1)[0] for ticker in ticker_list]

    return ticker_list, ticker_ex_exchange


def price_last_close(ticker, stock_price_data):
    display_log(f"Getting {ticker} EOD price.")
    try:
        if not stock_price_data.empty:
            return stock_price_data["Adjusted_close"].iloc[-1]
    except Exception as e:
        display_log(f"o price data found for {ticker}?")
        return 0


# Getting non-numerical data of the ticker.
def get_general_data(ticker, fund_data):
    display_log(f"Getting general {ticker} stock data.")

    try:
        general_data = json.loads(fund_data.decode("utf-8")).get("General", {})
        stock_name = general_data["Name"]
        country_name = general_data["CountryName"]
        sector_name = general_data.get("Sector", "Other")
        industry_name = general_data.get("Industry", "Other")
        primary_exchange = general_data.get("Exchange", "Other")
        if primary_exchange == "NASDAQ":
            primary_exchange = "ISLAND"
        elif primary_exchange == "NYSE ARCA":
            primary_exchange = "ARCA"
        currency_code = general_data["CurrencyCode"]
    except (KeyError, json.decoder.JSONDecodeError) as e:
        logging.debug(f"Error: {e}")
        stock_name = "drop_ticker"
        country_name = "drop_ticker"
        sector_name = "Other"
        industry_name = "Other"
        primary_exchange = "drop_ticker"
        currency_code = "drop_ticker"

    try:
        share_data = json.loads(fund_data.decode("utf-8")).get("SharesStats", {})
        institutional_holders = share_data["PercentInstitutions"]
    except (KeyError, json.decoder.JSONDecodeError) as e:
        institutional_holders = 0

    return [stock_name, country_name, sector_name, industry_name, institutional_holders, primary_exchange, currency_code]


def extract_currency_code(fund_data):
    return json.loads(fund_data.decode("utf-8")).get("General", {}).get("CurrencyCode", "USD")


# GBP is coded as GBX and ILS is coded as ILA for some reason.
def map_currency_code(code):
    return "GBP" if code == "GBX" else "ILS" if code == "ILA" else code


def date_string_to_datetime(dataframe):
    if "Date" in dataframe.columns:
        dataframe["Date"] = pd.to_datetime(dataframe["Date"])
    else:
        dataframe["Adjusted_close"] = [0]
    return dataframe


def extract_weekly_returns(is_israel_stock, price_data):
    weekly = price_data[price_data["Date"].dt.weekday == 3] if is_israel_stock else price_data[price_data["Date"].dt.weekday == 4]
    weekly_returns = (weekly["Adjusted_close"].shift(-1) / weekly["Adjusted_close"]) - 1
    return weekly_returns[:-1]


def calculate_volatility(returns, period=52):
    return np.std(returns) * math.sqrt(period)


def volatility_analysis(ticker, is_israel_stock, price_data):
    display_log(f"Performing {ticker} volatility factor analysis...")

    weekly_returns = extract_weekly_returns(is_israel_stock, price_data)

    volatility_momentum = calculate_volatility(weekly_returns.tail(156))

    month_prior = price_data["Adjusted_close"][:-21].tail(21).to_list()
    volatility_stop = np.std(month_prior) / month_prior[-1] * 3

    return volatility_momentum, volatility_stop


def calculate_momentum(stock_prices, riskfree_rate, volatility_3y):
    try:
        # Calculate 12-month momentum
        momentum_12_month = (((stock_prices[-21] / stock_prices[-273]) - 1) - riskfree_rate) / volatility_3y
        
        # Calculate 6-month momentum
        momentum_6_month = (((stock_prices[-21] / stock_prices[-147]) - 1) - riskfree_rate) / volatility_3y
        
        # Calculate average momentum returns
        momentum_returns = (momentum_12_month + momentum_6_month) / 2
        
        # Calculate 1-month momentum
        momentum_1_month = ((stock_prices[-1] / stock_prices[-21]) - 1)
        
        return momentum_returns, momentum_1_month
    except Exception as e:
        display_log(f"Error calculating momentum: {e}")
        raise


def momentum_analysis(ticker, stock_price_data, fund_data, riskfree_rate_table):
    display_log(f"Performing {ticker} momentum factor analysis...")
    subfactors = 6
    drop_ticker = "drop_ticker"
    try:
        fx_code = map_currency_code(extract_currency_code(fund_data))

        # Get risk-free rate and rate change for the currency
        riskfree_rate = riskfree_rate_table[riskfree_rate_table["Currency"] == fx_code]["Rate"].iloc[0]
        
        stock_prices_tail = stock_price_data.tail(882)

        # String objects need to be converted into datetime objects.
        stock_prices_tail = date_string_to_datetime(stock_prices_tail)

        if len(stock_prices_tail) <= 147:
            display_log(f"{ticker} does not have enough timeseries data for momentum analysis.")
            return [drop_ticker for subfactor in range(subfactors)]
        
        is_israel_stock = fx_code == 'ILS'

        volatility_data = volatility_analysis(ticker, is_israel_stock, stock_prices_tail)
        volatility_3y, volatility_stop = volatility_data
        
        stock_prices_list = stock_prices_tail["Adjusted_close"].tolist()
        while len(stock_prices_list) < 1260:
            stock_prices_list.insert(0, stock_prices_list[0])
        
        momentum_returns, momentum_1_month = calculate_momentum(stock_prices_list, riskfree_rate, volatility_3y)
        top_roc = stock_prices_tail["ROC"].tail(21).nlargest(5)
        max_roc = top_roc.mean()
        if momentum_returns > 0 and momentum_1_month > 0:
            mom_cycle = 1
        elif momentum_returns < 0 and momentum_1_month > 0:
            mom_cycle = 0.5
        else:
            mom_cycle = 0
        
        return [momentum_returns, volatility_3y, volatility_stop, momentum_1_month, max_roc, mom_cycle]
    
    except Exception as e:
        display_log(f"Error analyzing momentum for {ticker}: {e}")
        return [drop_ticker for subfactor in range(subfactors)]


def adjust_forex(fx_code):
    # Get the exchange rate for the currency if it's not already in the list.
    if fx_code not in forex_code_list:
        forex_code_list.append(fx_code)
        fx_rate = get_timeseries_data(fx_code + ".FOREX", token)["Adjusted_close"].tail(1)
        forex_rate_list.append(fx_rate)
    else:
        fx_rate = forex_rate_list[forex_code_list.index(fx_code)]

    return fx_rate


def size_analysis(ticker, fund_data):
    display_log(f"Performing {ticker} size factor analysis...")

    try:
        fund_data_dict = json.loads(fund_data)
        
        # Market capitalization normalized for currency rates.
        general_data = json.loads(fund_data).get("General", {})
        fx_code = map_currency_code(extract_currency_code(fund_data))
    
        if general_data["Type"] == "ETF":
            etf_data = json.loads(fund_data).get("ETF_Data", {})
            market_cap = etf_data.get("TotalAssets", {})
        elif general_data["Type"] == "FUND":
            mf_data = json.loads(fund_data).get("MutualFund_Data", {})
            market_cap = mf_data.get("Portfolio_Net_Assets", {})
        elif general_data["Type"] == "Common Stock":
            market_cap = fund_data_dict.get("Highlights", {}).get("MarketCapitalization")

    except Exception as e:
        display_log(f"Failed to solve market cap")
        return "drop_ticker"
    
    # Calculate the market capitalization in the base currency.
    if market_cap == None or market_cap == 0:
        return "drop_ticker"
    else:
        market_cap_base_currency = float(market_cap) / adjust_forex(fx_code)
        market_cap_log_normalized = 1 / math.log(market_cap_base_currency)
    
    return market_cap_log_normalized


def vix_table(region_data):
    vol_index_dict = dict.fromkeys(set(region_data["Vol Index"].tolist()))
    for index in vol_index_dict:
        vol_index_dict[index] = get_timeseries_data(index, token).tail(252)
        vol_index_dict[index]["Var"] = np.nanvar(vol_index_dict[index]["ROC"])
    
    return vol_index_dict


def compile_to_region_data(region_data, vix_tables):
    vol_index_table = []
    for index in region_data["Vol Index"]:
        vol_index_table.append(vix_tables[index])

    return vol_index_table

# We should try and see if we can seperate taking in region data since we will need
# the function for calculating market beta anyways and writing a second function
# with essentially 90% of the same lines is pretty redundant.
def calculate_return_volatility(ticker, stock_price_data, fund_data, region_data):
    display_log(f'Performing {ticker} return volatility factor analysis...')

    try:
        general_data = json.loads(fund_data.decode("utf-8")).get("General", {})
        country_name = general_data.get("CountryName")

        if country_name is None:
            raise ValueError("CountryName not found in fund_data.")
        index_number = region_data.index[region_data["Country"] == country_name].tolist()

        if not index_number:
            raise ValueError("Country not found in region_data.")
    
        vol_roc = region_data["Vol Table"][index_number[0]]["ROC"]
        vol_var = region_data["Vol Table"][index_number[0]]["Var"]

        stock_prices_tail = stock_price_data.tail(252)
        stock_roc = stock_prices_tail["ROC"].tail(252)
        stock_roc_clean = stock_roc.dropna()
        vol_roc_clean = vol_roc.dropna()
        covariance = np.cov(stock_roc_clean, vol_roc_clean)[0, 1]
        beta = covariance / vol_var
        
    except Exception as e:
        display_log(f"Error analyzing return volatility for {ticker}: {e}")
        return "drop_ticker"

    return beta.tolist()[-1]


def liquidity_analysis(ticker, stock_price_data, fund_data):
    display_log(f"Performing {ticker} liquidity analysis...")
    per_day_dvol = []
    general_data = json.loads(fund_data).get("General", {})

    try:
        fx_code = map_currency_code(extract_currency_code(fund_data))
        fx_code = "ILS" if fx_code == "ILA" else fx_code

        stock_prices_tail = stock_price_data.tail(252)
        prices = stock_prices_tail["Adjusted_close"].tolist()
        volume = stock_prices_tail["Volume"].tolist()
        returns = stock_prices_tail["ROC"].tolist()

        per_day_dvol = [close * vol for close, vol in zip(prices, volume)]
        per_day_ret_ratio = [roc / dvol for roc, dvol in zip(returns, per_day_dvol)]
        # Divided sum by a million to make numbers more readable for debugging.
        dvol = (sum(per_day_dvol[-42:-21]) / adjust_forex(fx_code) / 1000000).tolist()[0]
        ill = (sum(per_day_ret_ratio) / 252) * 1000000000
    
    except Exception as e:
        display_log(f"Error analyzing dollor volume for {ticker}: {e}")
        dvol = "drop_ticker"
        ill = "drop_ticker"

    try:
        if general_data["Type"] == "ETF" or general_data["Type"] == "FUND":
            turnover = 1
        elif general_data["Type"] == "Common Stock":
            shares_outstanding = json.loads(fund_data).get("SharesStats", {})["SharesOutstanding"]

            turnover = sum(volume) / len(volume) / shares_outstanding  
        
    except Exception as e:
        display_log(f"Error analyzing turnover for {ticker}: {e}")
        turnover = "drop_ticker"

    return [dvol, turnover, ill]


# Run all our factor analysis at once.
def factor_analysis():
    for ticker in master_ticker_list:
        display_log(f"Performing factor analysis on {ticker}.")
        fund_data = get_fundamental_data(ticker, token)
        eod_data = get_timeseries_data(ticker, token)
        
        # Fix for REITs and Trust units in Canada is causing bad data to be gathered
        try:
            ticker_code = json.loads(fund_data.decode("utf-8"))["General"]["Code"]
        except:
            pass
        if ticker_code not in ticker and ".TO" in ticker:
            master_ticker_list[master_ticker_list.index(ticker)] = ticker_code + ".TO"

        general_data_var = get_general_data(ticker, fund_data)
        momentum_data_var = momentum_analysis(ticker, eod_data, fund_data, riskfree_rate_table)
        size_data_var = size_analysis(ticker, fund_data)
        rvol_data_var = calculate_return_volatility(ticker, eod_data, fund_data, region_data)
        liquidity_var = liquidity_analysis(ticker, eod_data, fund_data)
                
        # General Data
        name_list.append(general_data_var[0])
        eod_list.append(price_last_close(ticker, eod_data))
        country_list.append(general_data_var[1])
        sector_list.append(general_data_var[2])
        industry_list.append(general_data_var[3])
        institutional_holders_list.append(general_data_var[4])
        primary_exchange_list.append(general_data_var[5])
        currency_code_list.append(general_data_var[6])
        
        # Momentum
        momentum_list.append(momentum_data_var[0])
        momentum_1mo_list.append(momentum_data_var[3])
        max_roc_list.append(momentum_data_var[4])
        market_cycle_list.append(momentum_data_var[5])

        # Size
        market_cap_list.append(size_data_var)
        
        # Volatility
        y3_vol_list.append(momentum_data_var[1])
        stop_vol_list.append(momentum_data_var[2])
        rvol_list.append(rvol_data_var)

        # Liquidity
        dvol_list.append(liquidity_var[0])
        turn_list.append(liquidity_var[1])
        ill_list.append(liquidity_var[2])

    
# Compile data for reading
def compile_data():
    display_log("Compiling data...")
    
    data_table = pd.DataFrame()
    data_table["Ticker"] = master_ticker_list
    data_table["Ticker IBKR"] = isolated_ticker
    data_table["Name"] = name_list
    data_table["Price"] = eod_list
    data_table["Country Name"] = country_list
    data_table["Sector"] = sector_list
    data_table["Industry"] = industry_list

    data_table["Institutional Holders fct"] = institutional_holders_list
    data_table["Momentum fct"] = momentum_list
    data_table["1Month fct"] = momentum_1mo_list
    data_table["Max ROC raw"] = max_roc_list
    data_table["Cycle fct"] = market_cycle_list

    data_table["12M Vol fct"] = y3_vol_list
    data_table["Stop Vol fct"] = stop_vol_list

    data_table["Size raw"] = market_cap_list
    numeric_size = pd.to_numeric(data_table["Size raw"], errors="coerce")
    numeric_sum = numeric_size.sum()
    data_table["Size fct"] = numeric_size / numeric_sum
    data_table["Return Volatility fct"] = rvol_list

    data_table["Dollar Volume fct"] = dvol_list
    data_table["Turnover fct"] = turn_list
    data_table["Illiquidity fct"] = ill_list

    data_table["PrimaryExchange"] = primary_exchange_list
    data_table["Currency"] = currency_code_list
    
    return data_table


# We need a list of rejected tickers for full analysis and checking of false negatives.
def output_rejected_tickers():
    display_log("Outputting rejected tickers. These can be found in rejected_tickers.txt.")
    
    rejected_data_table = fundamental_data_table[(fundamental_data_table.iloc[:, 1:].isin(["drop_ticker"]).any(axis = 1))]
    rejected_tickers = rejected_data_table["Ticker"]
    with open('output/rejected_tickers.txt', 'w') as f:
        for ticker in list(rejected_tickers):
            f.write("%s\n" % ticker)
            

def winsorize_scores(scores):
    upper_bound = 3.0
    lower_bound = -3.0
    
    winsorized_scores = [score if lower_bound <= score <= upper_bound else upper_bound if score > upper_bound else lower_bound for score in scores]
    
    return winsorized_scores


# There are several sectors that are not labled correctly from the data source.
def fix_sectors(data_table):
    proper_sectors = ["Basic Materials",
                      "Communication Services",
                      "Consumer Cyclical",
                      "Consumer Defensive",
                      "Energy",
                      "Financial Services",
                      "Healthcare",
                      "Industrials",
                      "Real Estate",
                      "Technology",
                      "Utilities",
                      "Other"]
    new_data_table = data_table[data_table["Sector"].isin(proper_sectors)]
    
    return new_data_table


def scale_values(values, target_min, target_max):
    min_value = min(values)
    max_value = max(values)

    scaled_values = [(value - min_value) / (max_value - min_value) * (target_max - target_min) + target_min
                     for value in values]

    return scaled_values


# These factors do not need to be normalized, simply winsorized.
def standardize_simple_scores(data_table):
    display_log("Standardizing simple scores...")
    simple_factors = ["Institutional Holders fct",
                        "Momentum fct", 
                        "1Month fct", 
                        "Size fct",
                        "Return Volatility fct",
                        "Dollar Volume fct",
                        "Turnover fct"]

    scores = [(data_table[factor].astype(float) - np.nanmean(data_table[factor].astype(float))) / np.nanstd(data_table[factor].astype(float)) for factor in simple_factors]
    winsorized_scores = [winsorize_scores(score) for score in scores]
    
    for i, factor in enumerate(simple_factors):
        display_log(f"Calculating {factor} Z-Scores...")
        data_table[factor[:-3] + "Z-Scores"] = winsorized_scores[i]

    return data_table
    

def scores_post_processing(data_table):
    data_table["Trailing Amt"] = data_table["Stop Vol fct"] * 100    
    data_table["1Month Reversal fct"] = data_table["1Month fct"] * scale_values(data_table["Turnover Z-Scores"], -1, 1)
    data_table["1Month Reversal Z-Scores"] = scale_values(data_table["1Month Reversal fct"], -3, 3)
    data_table["Max ROC fct"] = [1 / abs(value - data_table["Max ROC raw"].mean().tolist()) for value in data_table["Max ROC raw"]]
    data_table["Max ROC Z-Scores"] = scale_values(data_table["Max ROC fct"], -3, 3)
    data_table["Return Volatility Z-Scores"] = data_table["Return Volatility Z-Scores"] * -1
    data_table["Turnover Z-Scores"] = data_table["Turnover Z-Scores"] * -1
    data_table["Illiquidity Z-Scores"] = data_table["Illiquidity fct"] * -1 * scale_values(data_table["Size fct"], -1, 1)
    data_table["Illiquidity Z-Scores"] = scale_values(data_table["Illiquidity Z-Scores"], -3, 3)

    data_table["Stop Px"] = data_table["Price"] - (data_table["Stop Vol fct"] * data_table["Price"])
    data_table["Limit Offset"] = (data_table["Price"] - (data_table["Stop Vol fct"] * data_table["Price"])) * 0.005

    cols = data_table.columns.tolist()
    important_columns = ["Ticker IBKR", "Currency", "PrimaryExchange", "Trailing Amt", "Stop Px", "Limit Offset"]
    cols = [column_name for column_name in cols if column_name not in important_columns]
    cols.extend(important_columns)
    data_table = data_table[cols]

    return data_table


dev_mode = is_dev_mode()

# API key for authentication
licenseFile = open("licenseFile.key", "r")
api_key = licenseFile.read()
region_data = pd.read_csv("regions_data.csv")
token = api_key

display_log("Grabbing indicies data...")
master_ticker_list, isolated_ticker = get_ticker_list(get_indices(), token)

display_log("Grabbing riskfree rates...")
riskfree_rate_table = get_riskfree_rate(region_data)

display_log("Grabbing VIX tables...")
region_data["Vol Table"] = compile_to_region_data(region_data, vix_table(region_data))

factor_analysis()
fundamental_data_table = compile_data()
output_rejected_tickers()
fundamental_data_table.to_csv("output/" + current_date.strftime("%Y%m%d") + "dump.csv", index = False, header = True)
#fundamental_data_table = pd.read_csv('20221107dump.csv')
fundamental_data_table = fix_sectors(fundamental_data_table)

# THE SQUIGGLY IS NOT A TYPO
fundamental_data_table = fundamental_data_table[~fundamental_data_table.isin(["drop_ticker"]).any(axis=1)]
fundamental_data_table = fundamental_data_table.reset_index(drop=True)

# These are ordered this way specifically to make it easier to read the output spreadsheet.
fundamental_data_table = standardize_simple_scores(fundamental_data_table)
fundamental_data_table = scores_post_processing(fundamental_data_table)

fundamental_data_table.to_csv("output/" + current_date.strftime("%Y%m%d") + ".csv", index = False, header = True)
fundamental_data_table.to_csv("output/output.csv", index = False, header = True)

display_log("[" + str(dt.datetime.now()) + "] " + "Script duration " + str((dt.datetime.now() - script_start_time)))