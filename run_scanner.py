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
y3_vol_list = []
stop_vol_list = []
stop_px_list = []
trailing_amt_list = []
offset_list = []
rsi_list = []
macd_line_list = []
macd_signal_list = []
macd_spread_list = []
isolated_ticker = []
sma200_list = []


# Check if we need to be using dev settings.
def is_dev_mode():
    if os.getcwd()[-3:] == "dev":
        logging.basicConfig(filename="log.txt", level=logging.DEBUG)
    else:
        logging.basicConfig(filename="log.txt", level=logging.INFO)


# Save and display log lines
def display_log(string):
    logging.info(string)
    print(string)
    

# Creating sessions to reduce API calls .   
def create_session():
    session = requests.Session()
    retry = Retry(total=3, backoff_factor=1, status_forcelist=[502, 503, 504])
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
    return pd.read_csv(io.StringIO(canada_check(url, ticker).decode("utf-8")))
    

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
    if not stock_price_data.empty:
        return stock_price_data["Adjusted_close"].iloc[-1]
    
    logging.debug("No price data found?")
    return 0


# Getting non-numerical data of the ticker.
def get_general_data(ticker, fund_data):
    display_log(f"Getting general {ticker} stock data.")

    try:
        general_data = json.loads(fund_data.decode("utf-8")).get("General", {})
        share_data = json.loads(fund_data.decode("utf-8")).get("SharesStats", {})
        stock_name = general_data["Name"]
        country_name = general_data["CountryName"]
        sector_name = general_data.get("Sector", "Other")
        industry_name = general_data.get("Industry", "Other")
        institutional_holders = share_data["PercentInstitutions"]
    except (KeyError, json.decoder.JSONDecodeError) as e:
        logging.debug(f"Error: {e}")
        stock_name = "drop_ticker"
        country_name = "drop_ticker"
        sector_name = "Other"
        industry_name = "Other"
        institutional_holders = 0

    return [stock_name, country_name, sector_name, industry_name, institutional_holders]


def extract_currency_code(fund_data):
    return json.loads(fund_data.decode("utf-8")).get("General", {}).get("CurrencyCode", "USD")


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
    display_log(f"Performing {ticker} volatility factor analysis.")

    weekly_returns = extract_weekly_returns(is_israel_stock, price_data)

    volatility_momentum = calculate_volatility(weekly_returns.tail(156))

    month_prior = price_data["Adjusted_close"][:-21].tail(21).to_list()
    volatility_stop = np.std(month_prior) / month_prior[-1] * 3

    return volatility_momentum, volatility_stop


def momentum_analysis(ticker, stock_price_data, fund_data):
    display_log(f"Performing {ticker} momentum factor analysis.")

    fx_code = map_currency_code(extract_currency_code(fund_data))

    stock_price_data_tail = stock_price_data.tail(882)

    # String objects need to be converted into datetime objects.
    stock_price_data_tail = date_string_to_datetime(stock_price_data_tail)

    # Tickers listed for less than 1 year do not have enough data for momentum trading.
    if len(stock_price_data_tail) <= 147:
        display_log(f"{ticker} does not have enough timeseries data for momentum analysis. It has been marked to drop.")
        return ["drop_ticker", "drop_ticker", "drop_ticker", "drop_ticker", "drop_ticker", "drop_ticker", "drop_ticker", "drop_ticker"]

    # Check if the stock is traded in Israel. Israel trades on different schedules.
    is_israel_stock = fx_code == 'ILS'

    # ILS is coded as ILA for some reason.
    fx_code = "ILS" if fx_code == "ILA" else fx_code

    try:
        volatility_data = volatility_analysis(ticker, is_israel_stock, stock_price_data_tail)
        volatility_3y, volatility_stop = volatility_data

        stock_price_data_list = stock_price_data_tail["Adjusted_close"].tolist()

        while len(stock_price_data_list) < 1260:
            stock_price_data_list.insert(0, stock_price_data_list[0])

        momentum_12_month = ((stock_price_data_list[-21] / stock_price_data_list[-273]) - 1) / volatility_3y
        momentum_6_month = ((stock_price_data_list[-21] / stock_price_data_list[-147]) - 1) / volatility_3y
        momentum_returns = (momentum_12_month + momentum_6_month)/2

        rsi = ta.momentum.RSIIndicator(stock_price_data_tail["Adjusted_close"]).rsi().tail(1).to_list()[-1]
        macd = ta.trend.MACD(stock_price_data_tail["Adjusted_close"])

        macd_line = macd.macd().tail(1).to_list()[-1] / stock_price_data_list[-1]
        macd_signal = macd.macd_signal().tail(1).to_list()[-1]  / stock_price_data_list[-1]
        macd_spread = macd.macd_diff().tail(1).to_list()[-1]  / stock_price_data_list[-1]
        sma200 = ta.trend.SMAIndicator(stock_price_data_tail["Adjusted_close"], 200, fillna = True).sma_indicator().tail(1).to_list()[-1]

    except Exception as e:
        logging.error(f"Error analyzing momentum for {ticker}: {e}")
        return ["drop_ticker", "drop_ticker", "drop_ticker", "drop_ticker", "drop_ticker", "drop_ticker", "drop_ticker", "drop_ticker"]

    return [momentum_returns, volatility_3y, volatility_stop, rsi, macd_line, macd_signal, macd_spread, sma200]


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
        momentum_data_var = momentum_analysis(ticker, eod_data, fund_data)
        
        # General Data
        name_list.append(general_data_var[0])
        eod_list.append(price_last_close(ticker, eod_data))
        country_list.append(general_data_var[1])
        sector_list.append(general_data_var[2])
        industry_list.append(general_data_var[3])
        institutional_holders_list.append(general_data_var[4])
        
        # Momentum
        momentum_list.append(momentum_data_var[0])
        
        # Volatility
        y3_vol_list.append(momentum_data_var[1])
        stop_vol_list.append(momentum_data_var[2])

        # Trading Indicators
        rsi_list.append(momentum_data_var[3])
        macd_line_list.append(momentum_data_var[4])
        macd_signal_list.append(momentum_data_var[5])
        macd_spread_list.append(momentum_data_var[6])
        sma200_list.append(momentum_data_var[7])
        
    
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
    data_table["Insitutional Holders"] = institutional_holders_list

    data_table["Momentum fct"] = momentum_list

    data_table["12M Vol fct"] = y3_vol_list
    data_table["Stop Vol fct"] = stop_vol_list

    data_table["RSI fct"] = rsi_list
    data_table["MACD Line"] = macd_line_list
    data_table["MACD Signal"] = macd_signal_list
    data_table["MACD Hist fct"] = macd_spread_list
    data_table["SMA 200"] = sma200_list
    
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


# These factors do not need to be normalized, simply winsorized.
def standardize_simple_scores(data_table):
    display_log("Standardizing simple scores...")
    simple_factors = ["Momentum fct", "RSI fct", "MACD Hist fct"]

    scores = [(data_table[factor].astype(float) - np.nanmean(data_table[factor].astype(float))) / np.nanstd(data_table[factor].astype(float)) for factor in simple_factors]
    winsorized_scores = [winsorize_scores(score) for score in scores]
    
    for i, factor in enumerate(simple_factors):
        data_table[factor[:-3] + "Z-Scores"] = winsorized_scores[i]

    return data_table
    

def multifactor_scores(data_table):
    data_table["Reversal Score"] = (data_table["RSI Z-Scores"] + data_table["MACD Hist Z-Scores"]) * 0.05
    data_table["Trailing Amt"] = data_table["Stop Vol fct"] * 100
    data_table["Stop Px"] = data_table["Price"] - (data_table["Stop Vol fct"] * data_table["Price"])
    data_table["Limit Offset"] = (data_table["Price"] - (data_table["Stop Vol fct"] * data_table["Price"])) * 0.005

    cols = data_table.columns.tolist()
    important_columns = ["Ticker IBKR", "Trailing Amt", "Stop Px", "Limit Offset"]
    cols = [column_name for column_name in cols if column_name not in important_columns]
    cols.extend(important_columns)
    data_table = data_table[cols]

    return data_table


dev_mode = is_dev_mode()

# API key for authentication
licenseFile = open("licenseFile.key", "r")
api_key = licenseFile.read()
token = api_key

display_log("Grabbing indicies data.")

master_ticker_list, isolated_ticker = get_ticker_list(get_indices(), token)

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
fundamental_data_table = multifactor_scores(fundamental_data_table)

fundamental_data_table.to_csv("output/" + current_date.strftime("%Y%m%d") + ".csv", index = False, header = True)

display_log("[" + str(dt.datetime.now()) + "] " + "Script duration " + str((dt.datetime.now() - script_start_time)))