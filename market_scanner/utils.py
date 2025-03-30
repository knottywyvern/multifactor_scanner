"""
Utility functions for the Market Scanner.
"""
import os
import logging
import json
import pandas as pd
import numpy as np
import yaml
from logging.handlers import RotatingFileHandler

def setup_logging(config):
    """
    Set up logging with proper configuration.
    
    Args:
        config (dict): Application configuration
        
    Returns:
        logging.Logger: Configured logger
    """
    # Determine log level based on environment
    log_level = logging.DEBUG if config.get('dev_mode', False) else logging.INFO
    
    # Ensure log directory exists
    log_dir = os.path.dirname(config['paths']['log_file'])
    os.makedirs(log_dir, exist_ok=True)
    
    # Configure root logger
    logger = logging.getLogger()
    logger.setLevel(log_level)
    
    # Remove any existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Create a rotating file handler
    file_handler = RotatingFileHandler(
        filename=config['paths']['log_file'],
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5
    )
    file_handler.setLevel(log_level)
    
    # Create a console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    
    # Create a formatter and set it for both handlers
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Add the handlers to the root logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

def load_config(config_path="config.yaml"):
    """
    Load configuration from YAML file.
    
    Args:
        config_path (str): Path to the configuration file
        
    Returns:
        dict: Configuration dictionary
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    # Set default values for paths
    if "paths" not in config:
        config["paths"] = {}
    
    if "log_file" not in config["paths"]:
        config["paths"]["log_file"] = "logs/scannerlog.txt"
    
    return config

def display_log(string):
    """
    Log and print a message.
    
    Args:
        string (str): Message to log and print
    """
    logging.info(string)
    print(string)

def load_json_data(data):
    """
    Safely load JSON data.
    
    Args:
        data: Data to parse as JSON
        
    Returns:
        dict: Parsed JSON data or empty dict on error
    """
    if data is None:
        return {}
    try:
        return json.loads(data.decode("utf-8"))
    except (json.JSONDecodeError, AttributeError) as e:
        display_log(f"Error decoding JSON data: {e}")
        return {}

def map_primary_exchange(exchange):
    """
    Map exchange name to standardized format.
    
    Args:
        exchange (str): Exchange name
        
    Returns:
        str: Standardized exchange name
    """
    exchange_mapping = {"NYSE ARCA": "ARCA", "NYSE MKT": "AMEX"}
    return exchange_mapping.get(exchange, exchange)

def map_currency_code(code):
    """
    Map currency code to standardized format.
    
    Args:
        code (str): Currency code
        
    Returns:
        str: Standardized currency code
    """
    currency_mapping = {"GBX": "GBP", "ILA": "ILS"}
    return currency_mapping.get(code, code)

def extract_currency_code(fund_data):
    """
    Extract currency code from fundamental data.
    
    Args:
        fund_data: Fundamental data
        
    Returns:
        str: Currency code
    """
    fund_data_dict = load_json_data(fund_data)
    return fund_data_dict.get("General", {}).get("CurrencyCode", "USD")

def get_ticker_list(market_index, token, config):
    """
    Get list of tickers to analyze.
    
    Args:
        market_index (list): List of index tickers
        token (str): API token
        config (dict): Application configuration
        
    Returns:
        tuple: (list of tickers, list of tickers without exchange)
    """
    # This function will be implemented in main.py and replaced with proper imports
    # Only the signature is kept here as a reference
    pass

def load_tickers_from_file(file_path, log_message):
    """
    Load tickers from a file.
    
    Args:
        file_path (str): Path to the file
        log_message (str): Message to log when loading
        
    Returns:
        set: Set of tickers
    """
    tickers = set()
    if os.path.exists(file_path):
        display_log(log_message)
        with open(file_path) as f:
            tickers = {x.strip() for x in f.readlines() if x.strip()}
            display_log(list(tickers))
    return tickers

def normalize_ticker_format(ticker_list):
    """
    Normalize ticker format.
    
    Args:
        ticker_list (set/list): List of tickers
        
    Returns:
        set: Set of normalized tickers
    """
    return {ticker.replace(' ', '-').replace('/', '-') for ticker in ticker_list}

def get_ticker_code(fund_data, ticker):
    """
    Get correct ticker code.
    
    Args:
        fund_data: Fundamental data
        ticker (str): Original ticker
        
    Returns:
        str: Corrected ticker code
    """
    try:
        ticker_code = load_json_data(fund_data)["General"]["Code"]
        if ticker_code not in ticker and ".TO" in ticker:
            return ticker_code + ".TO"
    except Exception:
        pass
    return ticker