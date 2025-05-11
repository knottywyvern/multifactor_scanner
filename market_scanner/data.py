"""
Data models for the Market Scanner.
"""
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

from .utils import display_log, extract_currency_code, map_currency_code

class TickerData:
    """
    Class to store data for multiple tickers.
    
    This is the legacy class from the original implementation.
    """
    def __init__(self):
        self.data = {
            "Ticker": [], "Name": [], "Price": [], "Country Name": [], "Sector": [], "Industry": [],
            "Institutional Holders fct": [], "Momentum fct": [], "1Month fct": [], "Max ROC raw": [],
            "12M Vol fct": [], "Stop Vol fct": [], "Size raw": [], "Return Volatility fct": [],
            "Dollar Volume fct": [], "Turnover fct": [], "Illiquidity fct": [], "Trend Clarity fct": [],
            "PrimaryExchange": [], "Currency": [], "Security Type fct": [], "Revenue fct": []
        }
        self.forex_code_list = []
        self.forex_rate_list = []

    def append_data(self, ticker, fund_data, eod_data, riskfree_rate_table, region_data):
        """
        Append data for a ticker.
        
        Args:
            ticker (str): Ticker symbol
            fund_data: Fundamental data
            eod_data (pandas.DataFrame): EOD timeseries data
            riskfree_rate_table (pandas.DataFrame): Risk-free rate table
            region_data (pandas.DataFrame): Region data
        """
        if fund_data is None or eod_data is None:
            display_log(f"Skipping {ticker} due to missing data.")
            return
        
        # This function will be modified to use the new analysis functions
        display_log(f"Data for {ticker} has been added to the database.")

    def to_dataframe(self):
        """
        Convert stored data to DataFrame.
        
        Returns:
            pandas.DataFrame: DataFrame with all ticker data
        """
        return pd.DataFrame(self.data)


@dataclass
class TickerAnalysis:
    """
    Class to hold analysis results for a single ticker.
    
    This is the new implementation that will replace the dictionary-based approach.
    """
    ticker: str
    name: Optional[str] = None
    price: Optional[float] = None
    country: Optional[str] = None
    sector: Optional[str] = None
    industry: Optional[str] = None
    exchange: Optional[str] = None
    currency: str = "USD"
    security_type: str = "STOCK"
    
    # Factor values
    institutional_holders: Optional[float] = None
    momentum: Optional[float] = None
    momentum_1month: Optional[float] = None
    max_roc: Optional[float] = None
    volatility_12m: Optional[float] = None
    volatility_stop: Optional[float] = None
    size: Optional[float] = None
    return_volatility: Optional[float] = None
    dollar_volume: Optional[float] = None
    turnover: Optional[float] = None
    illiquidity: Optional[float] = None
    trend_clarity: Optional[float] = None
    revenue: Optional[float] = None
    
    # Derived values
    # trailing_amount: Optional[float] = None
    # stop_price: Optional[float] = None
    # limit_offset: Optional[float] = None
    
    # New financial metrics (last 10 years)
    cagr: Optional[float] = None
    longest_drawdown: Optional[float] = None # Duration in days/periods
    max_drawdown: Optional[float] = None
    calmar_ratio: Optional[float] = None
    std_dev: Optional[float] = None
    sharpe_ratio: Optional[float] = None
    sortino_ratio: Optional[float] = None
    sum_three_largest_drawdown: Optional[float] = None # Sum of durations of 3 largest drawdown periods
    actual_timeseries_start_date: Optional[str] = None # Actual start date of data used for 10yr metrics
    
    # Standardized scores
    z_scores: Dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the analysis to a dictionary.
        
        Returns:
            dict: Dictionary with analysis data
        """
        result = {
            "Ticker": self.ticker,
            "Name": self.name,
            "Price": self.price,
            "Country Name": self.country,
            "Sector": self.sector,
            "Industry": self.industry,
            "PrimaryExchange": self.exchange,
            "Currency": self.currency,
            "Security Type fct": self.security_type,
            
            "Institutional Holders fct": self.institutional_holders,
            "Momentum fct": self.momentum,
            "1Month fct": self.momentum_1month,
            "Max ROC raw": self.max_roc,
            "12M Vol fct": self.volatility_12m,
            "Stop Vol fct": self.volatility_stop,
            "Size raw": self.size,
            "Return Volatility fct": self.return_volatility,
            "Dollar Volume fct": self.dollar_volume,
            "Turnover fct": self.turnover,
            "Illiquidity fct": self.illiquidity,
            "Trend Clarity fct": self.trend_clarity,
            "Revenue fct": self.revenue,
            
            # New financial metrics
            "CAGR 10yr fct": self.cagr,
            "Longest Drawdown fct": self.longest_drawdown,
            "Sum 3 Largest Drawdown fct": self.sum_three_largest_drawdown,
            "Max Drawdown 10yr fct": self.max_drawdown,
            "Calmar Ratio 10yr fct": self.calmar_ratio,
            "Std Dev 10yr fct": self.std_dev,
            "Sharpe Ratio 10yr fct": self.sharpe_ratio,
            "Sortino Ratio 10yr fct": self.sortino_ratio,
            "Actual Timeseries Start Date": self.actual_timeseries_start_date,
        }
        
        # Add derived values if available
        # if self.trailing_amount is not None:
        #     result["Trailing Amt"] = self.trailing_amount
        # if self.stop_price is not None:
        #     result["Stop Px"] = self.stop_price
        # if self.limit_offset is not None:
        #     result["Limit Offset"] = self.limit_offset
            
        # Add z-scores
        for key, value in self.z_scores.items():
            result[key] = value
            
        return result


class MarketAnalysis:
    """
    Class to manage analysis for multiple tickers.
    """
    def __init__(self):
        self.analyses: List[TickerAnalysis] = []
        self.forex_rates: Dict[str, float] = {}
        
    def add_analysis(self, analysis: TickerAnalysis):
        """
        Add a ticker analysis to the collection.
        
        Args:
            analysis (TickerAnalysis): Analysis to add
        """
        self.analyses.append(analysis)
        
    def add_forex_rate(self, currency_code: str, rate: float):
        """
        Add or update a forex rate.
        
        Args:
            currency_code (str): Currency code
            rate (float): Exchange rate
        """
        self.forex_rates[currency_code] = rate
        
    def get_forex_rate(self, currency_code: str) -> Optional[float]:
        """
        Get forex rate for a currency code.
        
        Args:
            currency_code (str): Currency code
            
        Returns:
            float: Exchange rate
        """
        return self.forex_rates.get(currency_code)
    
    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert all analyses to a pandas DataFrame.
        
        Returns:
            pandas.DataFrame: DataFrame with all analyses
        """
        if not self.analyses:
            return pd.DataFrame()
            
        # Convert each analysis to a dictionary
        data_dicts = [analysis.to_dict() for analysis in self.analyses]
        
        # Create DataFrame
        return pd.DataFrame(data_dicts)
    
    def filter_analyses(self, condition_func):
        """
        Create a new MarketAnalysis with filtered analyses.
        
        Args:
            condition_func (callable): Function that takes a TickerAnalysis 
                                      and returns True to keep it
            
        Returns:
            MarketAnalysis: New MarketAnalysis with filtered analyses
        """
        result = MarketAnalysis()
        result.forex_rates = self.forex_rates.copy()
        result.analyses = [a for a in self.analyses if condition_func(a)]
        return result
    
    def standardize_scores(self, factors: List[str], by_security_type: bool = True):
        """
        Calculate standardized scores for specified factors.
        
        Args:
            factors (list): List of factor attributes to standardize
            by_security_type (bool): Whether to standardize within each security type
        """
        for factor in factors:
            # For certain factors, standardize only for stocks
            if factor in ["institutional_holders", "turnover", "revenue"] and by_security_type:
                self._standardize_by_group(factor, lambda a: a.security_type)
            else:
                self._standardize_factor(factor)
    
    def _standardize_factor(self, factor_name: str):
        """
        Standardize a single factor across all analyses.
        
        Args:
            factor_name (str): Name of the factor to standardize
        """
        # Get valid values
        values = [getattr(a, factor_name) for a in self.analyses 
                 if getattr(a, factor_name) is not None]
        
        if not values:
            return
            
        # Calculate mean and std
        mean = np.mean(values)
        std = np.std(values)
        
        if std == 0:
            return
            
        # Calculate z-scores and add to each analysis
        for analysis in self.analyses:
            value = getattr(analysis, factor_name)
            if value is not None:
                z_score = (value - mean) / std
                z_score = np.clip(z_score, -3.0, 3.0)  # Winsorize
                analysis.z_scores[f"{factor_name}_z"] = z_score
    
    def _standardize_by_group(self, factor_name: str, group_func):
        """
        Standardize a factor within groups.
        
        Args:
            factor_name (str): Name of the factor to standardize
            group_func (callable): Function that takes an analysis and returns its group
        """
        # Group analyses
        groups = {}
        for analysis in self.analyses:
            group = group_func(analysis)
            if group not in groups:
                groups[group] = []
            groups[group].append(analysis)
            
        # Standardize within each group
        for group, group_analyses in groups.items():
            # Get valid values for this group
            values = [getattr(a, factor_name) for a in group_analyses 
                     if getattr(a, factor_name) is not None]
            
            if not values:
                continue
                
            # Calculate mean and std for this group
            mean = np.mean(values)
            std = np.std(values)
            
            if std == 0:
                continue
                
            # Calculate z-scores and add to each analysis in this group
            for analysis in group_analyses:
                value = getattr(analysis, factor_name)
                if value is not None:
                    z_score = (value - mean) / std
                    z_score = np.clip(z_score, -3.0, 3.0)  # Winsorize
                    analysis.z_scores[f"{factor_name}_z"] = z_score