#!/usr/bin/env python3
"""
Market Scanner - Main script for fetching and analyzing market data.
"""
import asyncio
import datetime as dt
import logging
import os
import pandas as pd
import sys
import traceback

from .utils import (
    setup_logging, load_config, load_tickers_from_file, 
    normalize_ticker_format, get_ticker_code
)
from .api import (
    get_indices, get_fundamental_data_async, get_timeseries_data_async,
    get_riskfree_rate, vix_table, fetch_tickers_from_market_index
)
from .data import MarketAnalysis
from .analysis import (
    analyze_ticker, post_process_analysis, calculate_derived_values
)
from .processing import (
    output_rejected_tickers, filter_invalid_data, scores_post_processing
)

async def build_ticker_list(config, index_tickers, token):
    """
    Build the list of tickers to analyze.
    
    Args:
        config (dict): Application configuration
        index_tickers (list): List of index tickers
        token (str): API token
        
    Returns:
        list: List of tickers to analyze
    """
    # Get tickers from market indices
    tickers = set()
    tickers.update(await fetch_tickers_from_market_index(index_tickers, token))
    
    # Add tickers from inclusion file
    if os.path.exists(config["paths"].get("ticker_inclusion", "")):
        tickers.update(load_tickers_from_file(
            config["paths"]["ticker_inclusion"], 
            "Including specific tickers;"
        ))
    
    # Normalize ticker format
    tickers = normalize_ticker_format(tickers)
    
    # Remove tickers from exclusion file
    if os.path.exists(config["paths"].get("ticker_exclusion", "")):
        exclusions = load_tickers_from_file(
            config["paths"]["ticker_exclusion"], 
            "Excluding specific tickers;"
        )
        tickers.difference_update(exclusions)
    
    return list(tickers)

async def main():
    """Main entry point for the market scanner."""
    # Start timing the script
    script_start_time = dt.datetime.now()
    current_date = dt.datetime.today()
    
    try:
        # Load configuration
        config = load_config()
        
        # Setup logging
        logger = setup_logging(config)
        logger.info(f"Starting market scanner at {script_start_time}")
        
        # Calculate timeseries start date
        timeseries_days_lookback = config.get("timeseries_days_lookback", 882)
        timeseries_start_date = current_date - dt.timedelta(days=timeseries_days_lookback)
        logger.info(f"Looking back to {timeseries_start_date.strftime('%Y-%m-%d')}")
        
        # API key for authentication
        with open(config["paths"]["license_file"], "r") as license_file:
            api_key = license_file.read().strip()
            
        # Ensure output directory exists
        output_dir = config["paths"].get("output_dir", "output")
        os.makedirs(output_dir, exist_ok=True)
        
        # Load region data
        region_data = pd.read_csv(config["paths"]["regions_data"])
        logger.info(f"Loaded region data with {len(region_data)} regions")
        
        # Get index tickers
        logger.info("Getting index tickers...")
        index_tickers = get_indices(config)
        
        # Get risk-free rates
        logger.info("Getting risk-free rates...")
        riskfree_rate_table = await get_riskfree_rate(region_data)
        
        # Get VIX tables for regional volatility
        logger.info("Getting VIX tables...")
        vix_tables = await vix_table(region_data, api_key)
        region_data["Vol Table"] = [vix_tables.get(index, pd.DataFrame()) for index in region_data["Vol Index"]]
        
        # Get ticker list from indices and inclusion/exclusion files
        logger.info("Building ticker list...")
        ticker_list = await build_ticker_list(config, index_tickers, api_key)
        logger.info(f"Analyzing {len(ticker_list)} tickers")
        
        # Fetch data for all tickers concurrently
        logger.info("Fetching fundamental data...")
        fundamental_data = await get_fundamental_data_async(ticker_list, api_key)
        
        logger.info("Fetching timeseries data...")
        timeseries_data = await get_timeseries_data_async(
            ticker_list, 
            api_key,
            timeseries_start_date.strftime("%Y-%m-%d")
        )

        # Log information about index tickers
        for index_ticker in index_tickers:
            if index_ticker in fundamental_data:
                try:
                    fund_data_dict = load_json_data(fundamental_data[index_ticker])
                    holdings_count = 0
                    ticker_examples = []
                    
                    if "Holdings" in fund_data_dict:
                        holdings = fund_data_dict["Holdings"]
                        if isinstance(holdings, dict) and len(holdings) >= 3:
                            holdings_data = holdings.get(2, {})
                            if holdings_data:
                                holdings_count = len(holdings_data)
                                ticker_examples = list(holdings_data.keys())[:5]  # First 5 tickers
                    
                    logging.info(f"Index {index_ticker} information:")
                    logging.info(f"  - Holdings count: {holdings_count}")
                    if ticker_examples:
                        logging.info(f"  - Example tickers: {', '.join(ticker_examples)}")
                    else:
                        logging.warning(f"  - No holdings found for {index_ticker}")
                except Exception as e:
                    logging.error(f"Error processing index {index_ticker}: {e}")
            else:
                logging.error(f"Failed to retrieve fundamental data for index {index_ticker}")
        
        # Create market analysis object
        market_analysis = MarketAnalysis()
        
        # Analyze each ticker
        for ticker in ticker_list:
            try:
                if ticker in fundamental_data and ticker in timeseries_data:
                    # Get correct ticker code (handle Canadian tickers)
                    ticker_code = get_ticker_code(fundamental_data[ticker], ticker)
                    
                    # Analyze the ticker
                    analysis = analyze_ticker(
                        ticker_code, 
                        fundamental_data[ticker], 
                        timeseries_data[ticker], 
                        riskfree_rate_table, 
                        region_data
                    )
                    
                    if analysis:
                        market_analysis.add_analysis(analysis)
                else:
                    logger.warning(f"Skipping {ticker}: Missing fundamental or timeseries data")
            except Exception as e:
                logger.error(f"Error analyzing {ticker}: {e}")
        
        # Post-process analyses
        logger.info(f"Post-processing {len(market_analysis.analyses)} ticker analyses")
        market_analysis = post_process_analysis(market_analysis)
        
        # Calculate derived values
        market_analysis = calculate_derived_values(market_analysis)
        
        # Convert to DataFrame
        result_df = market_analysis.to_dataframe()
        
        # Save main output
        output_file = os.path.join(output_dir, f"{current_date.strftime('%Y%m%d')}.csv")
        result_df.to_csv(output_file, index=False)
        logger.info(f"Saved results to {output_file}")
        
        # Save a copy as output.csv
        current_output = os.path.join(output_dir, "output.csv")
        result_df.to_csv(current_output, index=False)
        
        # Check if required columns exist
        required_columns = ["Ticker", "Security Type fct"]
        for column in required_columns:
            if column not in result_df.columns:
                logging.warning(f"Missing required column: {column}")
                if column == "Security Type fct":
                    # Add a default Security Type column
                    logging.info("Adding default 'Security Type fct' column")
                    result_df["Security Type fct"] = "STOCK"        
        
        # Output rejected tickers
        output_rejected_tickers(result_df, output_dir)
        
        # Log execution time
        script_duration = dt.datetime.now() - script_start_time
        logger.info(f"Script completed in {script_duration}")
        
    except Exception as e:
        logging.error(f"Unhandled exception in main: {e}")
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        logging.info("Script interrupted by user")
        sys.exit(130)
    except Exception as e:
        logging.exception(f"Unhandled exception: {e}")
        sys.exit(1)