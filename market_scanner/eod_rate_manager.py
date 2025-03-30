"""
EOD Historical Data based risk-free rate manager with rate limiting.
"""
import os
import pandas as pd
import requests
import logging
import datetime as dt
import asyncio
import aiohttp
from typing import Dict, Optional, List

from .rate_limiter import global_rate_limiter, rate_limited_api_call

class EODRateManager:
    """Class to manage risk-free rate data using EOD Historical Data API."""
    
    def __init__(self, api_key, data_dir="data"):
        """
        Initialize the rate manager.
        
        Args:
            api_key: EOD Historical Data API key
            data_dir: Directory for storing data files
        """
        self.api_key = api_key
        self.data_dir = data_dir
        self.logger = logging.getLogger(__name__)
        self.rates_file = os.path.join(data_dir, "rates_cache.csv")
        
        # Create data directory if it doesn't exist
        os.makedirs(data_dir, exist_ok=True)
        
        # Map of countries to bond symbols, currencies, and fallback symbols
        self.country_bonds = {
            "USA": {"symbol": "US10Y.INDX", "currency": "USD", "fallback": None},
            "UK": {"symbol": "GB10Y.INDX", "currency": "GBP", "fallback": None},
            "Germany": {"symbol": "DE10Y.INDX", "currency": "EUR", "fallback": None},
            "Japan": {"symbol": "JP10Y.INDX", "currency": "JPY", "fallback": None},
            "Canada": {"symbol": "CA10Y.INDX", "currency": "CAD", "fallback": None},
            "Australia": {"symbol": "AU10Y.INDX", "currency": "AUD", "fallback": None},
            "Switzerland": {"symbol": "CH10Y.INDX", "currency": "CHF", "fallback": None},
            "France": {"symbol": "FR10Y.INDX", "currency": "EUR", "fallback": "DE10Y.INDX"},
            "Italy": {"symbol": "IT10Y.INDX", "currency": "EUR", "fallback": "DE10Y.INDX"},
            "Spain": {"symbol": "ES10Y.INDX", "currency": "EUR", "fallback": "DE10Y.INDX"},
            "Netherlands": {"symbol": "NL10Y.INDX", "currency": "EUR", "fallback": "DE10Y.INDX"},
            "Belgium": {"symbol": "BE10Y.INDX", "currency": "EUR", "fallback": "DE10Y.INDX"},
            "Austria": {"symbol": "AT10Y.INDX", "currency": "EUR", "fallback": "DE10Y.INDX"},
            "Finland": {"symbol": "FI10Y.INDX", "currency": "EUR", "fallback": "DE10Y.INDX"},
            "Portugal": {"symbol": "PT10Y.INDX", "currency": "EUR", "fallback": "DE10Y.INDX"},
            "Greece": {"symbol": "GR10Y.INDX", "currency": "EUR", "fallback": "DE10Y.INDX"},
            "Ireland": {"symbol": "IE10Y.INDX", "currency": "EUR", "fallback": "DE10Y.INDX"},
            "New Zealand": {"symbol": "NZ10Y.INDX", "currency": "NZD", "fallback": "AU10Y.INDX"},
            "Sweden": {"symbol": "SE10Y.INDX", "currency": "SEK", "fallback": "DE10Y.INDX"},
            "Norway": {"symbol": "NO10Y.INDX", "currency": "NOK", "fallback": "DE10Y.INDX"},
            "Denmark": {"symbol": "DK10Y.INDX", "currency": "DKK", "fallback": "DE10Y.INDX"},
            "Hong Kong": {"symbol": "HK10Y.INDX", "currency": "HKD", "fallback": "US10Y.INDX"},
            "China": {"symbol": "CN10Y.INDX", "currency": "CNY", "fallback": "US10Y.INDX"},
            "South Korea": {"symbol": "KR10Y.INDX", "currency": "KRW", "fallback": "JP10Y.INDX"},
            "India": {"symbol": "IN10Y.INDX", "currency": "INR", "fallback": "US10Y.INDX"},
            "Israel": {"symbol": "IL10Y.INDX", "currency": "ILS", "fallback": "US10Y.INDX"},
            "Brazil": {"symbol": "BR10Y.INDX", "currency": "BRL", "fallback": "US10Y.INDX"},
            "Mexico": {"symbol": "MX10Y.INDX", "currency": "MXN", "fallback": "US10Y.INDX"}
        }
        
        # Alternate data sources for specific countries
        self.alternate_sources = {
            "Hong Kong": {"symbol": "HKGG3.FXBOND", "description": "Hong Kong 3-Year Government Bond"}
        }
    
    async def get_rates_async(self, force_refresh=False) -> pd.DataFrame:
        """
        Get risk-free rates for all countries asynchronously, with rate limiting.
        
        Args:
            force_refresh: Whether to force refresh rates from API
            
        Returns:
            DataFrame with risk-free rates
        """
        # Check for cached rates
        if not force_refresh and os.path.exists(self.rates_file):
            try:
                rates_df = pd.read_csv(self.rates_file)
                last_updated = pd.to_datetime(rates_df['Last_Updated'].iloc[0])
                
                # If rates are less than 7 days old, use cached data
                if (dt.datetime.now() - last_updated).days < 7:
                    self.logger.info("Using cached rates (less than 7 days old)")
                    return rates_df
            except Exception as e:
                self.logger.warning(f"Error reading cached rates: {e}")
        
        # Get fresh rates from EOD Historical Data
        return await self._fetch_rates_from_eod_async()
    
    def get_rates(self, force_refresh=False) -> pd.DataFrame:
        """
        Synchronous wrapper for get_rates_async.
        
        Args:
            force_refresh: Whether to force refresh rates from API
            
        Returns:
            DataFrame with risk-free rates
        """
        # Use asyncio to run the async method
        try:
            import asyncio
            loop = asyncio.get_event_loop()
            return loop.run_until_complete(self.get_rates_async(force_refresh))
        except Exception as e:
            self.logger.error(f"Error in get_rates: {e}")
            return self._get_default_rates()
    
    async def update_region_data_async(self, region_df) -> pd.DataFrame:
        """
        Update region data with latest risk-free rates, asynchronously.
        
        Args:
            region_df: DataFrame with region data
            
        Returns:
            Updated DataFrame with risk-free rates
        """
        # Get latest rates
        rates_df = await self.get_rates_async()
        
        # Create a copy of the input DataFrame
        result = region_df.copy()
        
        # Update rates in the DataFrame
        for index, row in result.iterrows():
            country = row['Country']
            if country in rates_df['Country'].values:
                rate = rates_df.loc[rates_df['Country'] == country, 'Rate'].iloc[0]
                result.at[index, 'Rate'] = rate
            else:
                self.logger.warning(f"No rate found for {country}, using default")
                result.at[index, 'Rate'] = 0.03  # Default to 3%
        
        return result
    
    def update_region_data(self, region_df) -> pd.DataFrame:
        """
        Synchronous wrapper for update_region_data_async.
        
        Args:
            region_df: DataFrame with region data
            
        Returns:
            Updated DataFrame with risk-free rates
        """
        try:
            import asyncio
            loop = asyncio.get_event_loop()
            return loop.run_until_complete(self.update_region_data_async(region_df))
        except Exception as e:
            self.logger.error(f"Error in update_region_data: {e}")
            
            # If error, just return the original with default rates
            result = region_df.copy()
            if 'Rate' not in result.columns:
                result['Rate'] = 0.03
            return result
    
    async def _fetch_rates_from_eod_async(self) -> pd.DataFrame:
        """
        Fetch risk-free rates from EOD Historical Data API asynchronously.
        Uses rate limiting to avoid hitting API limits.
        
        Returns:
            DataFrame with risk-free rates
        """
        self.logger.info("Fetching rates from EOD Historical Data API (rate limited)...")
        rates_data = []
        rate_cache = {}  # Store rates for fallbacks
        
        async with aiohttp.ClientSession() as session:
            # First pass: try to get rates for all countries
            tasks = []
            for country, info in self.country_bonds.items():
                # Check if we should use an alternate source
                if country in self.alternate_sources:
                    alt_info = self.alternate_sources[country]
                    symbol = alt_info['symbol']
                    description = alt_info['description']
                    self.logger.info(f"Using alternate source for {country}: {description}")
                else:
                    symbol = info['symbol']
                
                url = f"https://eodhistoricaldata.com/api/eod/{symbol}?api_token={self.api_key}&limit=1&fmt=json"
                tasks.append(self._fetch_rate_with_limit(session, url, country, info, symbol))
            
            # Wait for all tasks to complete
            results = await asyncio.gather(*tasks)
            
            # Collect results
            for result in results:
                if result is not None:
                    country, rate, symbol = result
                    rates_data.append({
                        "Country": country,
                        "Currency": self.country_bonds[country]["currency"],
                        "Rate": rate,
                        "Source": symbol,
                        "Last_Updated": dt.datetime.now().strftime("%Y-%m-%d")
                    })
                    # Store in cache for fallbacks
                    rate_cache[symbol] = rate
        
        # Second pass: use fallbacks for countries without data
        countries_with_data = {item["Country"] for item in rates_data}
        
        for country, info in self.country_bonds.items():
            if country not in countries_with_data and info["fallback"] is not None:
                fallback_symbol = info["fallback"]
                
                if fallback_symbol in rate_cache:
                    self.logger.info(f"Using fallback {fallback_symbol} for {country}")
                    rates_data.append({
                        "Country": country,
                        "Currency": info["currency"],
                        "Rate": rate_cache[fallback_symbol],
                        "Source": f"Fallback: {fallback_symbol}",
                        "Last_Updated": dt.datetime.now().strftime("%Y-%m-%d")
                    })
        
        # Create DataFrame
        if rates_data:
            rates_df = pd.DataFrame(rates_data)
            
            # Save to cache
            rates_df.to_csv(self.rates_file, index=False)
            
            return rates_df
        else:
            self.logger.error("Failed to fetch any rates from EOD Historical Data")
            return self._get_default_rates()
    
    async def _fetch_rate_with_limit(self, session, url, country, info, symbol):
        """
        Fetch a single rate with rate limiting.
        
        Args:
            session: aiohttp session
            url: API URL
            country: Country name
            info: Country info dictionary
            symbol: Symbol to fetch
            
        Returns:
            Tuple of (country, rate, symbol) or None if error
        """
        # Apply rate limiting
        await global_rate_limiter.acquire()
        try:
            async with session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    if data and len(data) > 0:
                        rate = float(data[0]['close']) / 100  # Convert percentage to decimal
                        return country, rate, symbol
                    else:
                        self.logger.warning(f"No data returned for {symbol}")
                else:
                    self.logger.warning(f"Error fetching data for {symbol}: {response.status}")
            return None
        except Exception as e:
            self.logger.error(f"Error processing {symbol}: {e}")
            return None
        finally:
            global_rate_limiter.release()
    
    def _get_default_rates(self) -> pd.DataFrame:
        """
        Create default risk-free rates as a last resort.
        
        Returns:
            DataFrame with default rates
        """
        self.logger.warning("Creating default rates")
        default_data = []
        
        for country, info in self.country_bonds.items():
            # Use country-specific default rates
            default_rate = 0.03  # Default 3%
            
            # Adjust for specific countries
            if country == "Japan":
                default_rate = 0.005  # 0.5%
            elif country == "Switzerland":
                default_rate = 0.01  # 1%
            elif country in ["Brazil", "Mexico", "India"]:
                default_rate = 0.07  # 7%
            elif country == "China":
                default_rate = 0.025  # 2.5%
            elif country == "Hong Kong":
                default_rate = 0.03  # 3% (pegged to USD)
            
            default_data.append({
                "Country": country,
                "Currency": info['currency'],
                "Rate": default_rate,
                "Source": "Default",
                "Last_Updated": dt.datetime.now().strftime("%Y-%m-%d")
            })
        
        default_df = pd.DataFrame(default_data)
        
        # Save to cache
        try:
            default_df.to_csv(self.rates_file, index=False)
        except Exception as e:
            self.logger.error(f"Error saving default rates: {e}")
        
        return default_df