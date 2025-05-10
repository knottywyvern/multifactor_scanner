# Market Scanner

A comprehensive tool for analyzing financial market data and performing factor-based analysis on stocks and funds.

## Features

- Fetch fundamental and price data for stocks and ETFs
- Calculate various factors for analysis:
  - Momentum
  - Volatility
  - Size
  - Return volatility
  - Liquidity
  - Trend clarity
- Calculate key performance metrics (10-year period):
  - CAGR (Compound Annual Growth Rate)
  - Maximum Drawdown
  - Longest Drawdown
  - Calmar Ratio
  - Standard Deviation (annualized)
  - Sharpe Ratio
  - Sortino Ratio
- Generate standardized scores
- Export results to CSV

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/market_scanner.git
   cd market_scanner
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Create a `license.txt` file with your EOD Historical Data API key:
   ```
   echo "your_api_key_here" > license.txt
   ```

4. Configure the scanner by editing `config.yaml`

## Usage

1. Create the required input files:
   - `data/index_inclusion.txt`: List of indices to include
   - `data/ticker_inclusion.txt`: Additional tickers to include
   - `data/ticker_exclusion.txt`: Tickers to exclude
   - `data/regions_data.csv`: Region data for volatility analysis

2. Run the scanner:
   ```
   python -m market_scanner.main
   ```

3. Results will be saved to:
   - `output/YYYYMMDD.csv`: Date-stamped results
   - `output/output.csv`: Latest results
   - `output/rejected_tickers.txt`: Tickers that were rejected during analysis

## Configuration

Edit `config.yaml` to customize the scanner:

- `dev_mode`: Enable more verbose logging
- `timeseries_days_lookback`: Number of days of historical data to fetch
- `default_index`: Default index to use if none specified
- `paths`: File paths for inputs and outputs

## Project Structure

```
market_scanner/
├── config.yaml               # Configuration file
├── requirements.txt          # Python dependencies
├── README.md                 # This file
├── market_scanner/           # Main package
│   ├── __init__.py           # Package initialization
│   ├── api.py                # API-related functions
│   ├── data.py               # Data models
│   ├── analysis.py           # Factor analysis functions
│   ├── processing.py         # Data processing functions
│   ├── utils.py              # Utility functions
│   └── main.py               # Main script
├── data/                     # Data directory
│   ├── index_inclusion.txt   # Indices to include
│   ├── ticker_inclusion.txt  # Additional tickers to include
│   ├── ticker_exclusion.txt  # Tickers to exclude
│   └── regions_data.csv      # Region data
├── logs/                     # Log directory
│   └── scannerlog.txt        # Log file
└── output/                   # Output directory
    ├── YYYYMMDD.csv          # Date-stamped results
    ├── output.csv            # Latest results
    └── rejected_tickers.txt  # Rejected tickers
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.