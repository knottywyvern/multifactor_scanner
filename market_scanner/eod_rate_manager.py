"""
EOD Historical Data based risk-free rate manager.
"""
import os
import pandas as pd
import requests
import logging
import datetime as dt
from typing import Dict, Optional

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
    
    def get_rates(self, force_refresh=False) -> pd.DataFrame:
        """
        Get risk-free rates for all countries.
        
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
        return self._fetch_rates_from_eod()
    
    def update_region_data(self, region_df) -> pd.DataFrame:
        """
        Update region data with latest risk-free rates.
        
        Args:
            region_df: DataFrame with region data
            
        Returns:
            Updated DataFrame with risk-free rates
        """
        # Get latest rates
        rates_df = self.get_rates()
        
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
    
    def _fetch_rates_from_eod(self) -> pd.DataFrame:
        """
        Fetch risk-free rates from EOD Historical Data API.
        
        Returns:
            DataFrame with risk-free rates
        """
        rates_data = []
        rate_cache = {}  # Store rates for fallbacks
        
        # First pass: try to get rates for all countries
        for country, info in self.country_bonds.items():
            symbol = info['symbol']
            currency = info['currency']
            
            try:
                # Check if we should use an alternate source
                if country in self.alternate_sources:
                    alt_info = self.alternate_sources[country]
                    url = f"https://eodhistoricaldata.com/api/eod/{alt_info['symbol']}?api_token={self.api_key}&limit=1&fmt=json"
                    self.logger.info(f"Using alternate source for {country}: {alt_info['description']}")
                else:
                    url = f"https://eodhistoricaldata.com/api/eod/{symbol}?api_token={self.api_key}&limit=1&fmt=json"
                
                response = requests.get(url)
                
                if response.status_code == 200:
                    data = response.json()
                    if data and len(data) > 0:
                        rate = float(data[0]['close']) / 100  # Convert percentage to decimal
                        
                        # Store in cache for fallbacks
                        rate_cache[symbol] = rate
                        
                        rates_data.append({
                            "Country": country,
                            "Currency": currency,
                            "Rate": rate,
                            "Source": symbol,
                            "Last_Updated": dt.datetime.now().strftime("%Y-%m-%d")
                        })
                    else:
                        self.logger.warning(f"No data returned for {symbol}")
                else:
                    self.logger.warning(f"Error fetching data for {symbol}: {response.status_code}")
            except Exception as e:
                self.logger.error(f"Error processing {symbol}: {e}")
        
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
            
            # If we have cached data, use it as fallback
            if os.path.exists(self.rates_file):
                self.logger.info("Using cached rates as fallback")
                return pd.read_csv(self.rates_file)
            
            # Last resort: create default data
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
            default_df.to_csv(self.rates_file, index=False)
            
            return default_df

# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Load API key
    with open("license.txt", "r") as f:
        api_key = f.read().strip()
    
    # Create rate manager
    rate_manager = EODRateManager(api_key)
    
    # Get rates
    rates = rate_manager.get_rates(force_refresh=True)
    print(rates)