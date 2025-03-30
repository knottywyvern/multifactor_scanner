"""
Create a default regions_data.csv file if it doesn't exist.
"""
import os
import pandas as pd
import logging

def create_default_regions_data(output_path="data/regions_data.csv"):
    """
    Create a default regions_data.csv file.
    
    Args:
        output_path: Path to save the file
        
    Returns:
        bool: True if file was created, False otherwise
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Check if file already exists
    if os.path.exists(output_path):
        logging.info(f"Regions data file already exists at {output_path}")
        return False
        
    # Create default data
    default_data = [
        # Major markets
        {"Country": "USA", "Ticker": "US", "Currency": "USD", "Rate": 0.03, "Vol Index": "VIX.INDX"},
        {"Country": "UK", "Ticker": "GB", "Currency": "GBP", "Rate": 0.03, "Vol Index": "VFTSE.INDX"},
        {"Country": "Germany", "Ticker": "DE", "Currency": "EUR", "Rate": 0.03, "Vol Index": "V1X.INDX"},
        {"Country": "France", "Ticker": "FR", "Currency": "EUR", "Rate": 0.03, "Vol Index": "V1X.INDX"},
        {"Country": "Japan", "Ticker": "JP", "Currency": "JPY", "Rate": 0.01, "Vol Index": "VXJ.INDX"},
        {"Country": "Canada", "Ticker": "CA", "Currency": "CAD", "Rate": 0.03, "Vol Index": "VIXC.INDX"},
        {"Country": "Australia", "Ticker": "AU", "Currency": "AUD", "Rate": 0.03, "Vol Index": "XVI.INDX"},
        
        # European markets
        {"Country": "Switzerland", "Ticker": "CH", "Currency": "CHF", "Rate": 0.01, "Vol Index": "V3X.INDX"},
        {"Country": "Italy", "Ticker": "IT", "Currency": "EUR", "Rate": 0.04, "Vol Index": "V1X.INDX"},
        {"Country": "Spain", "Ticker": "ES", "Currency": "EUR", "Rate": 0.04, "Vol Index": "V1X.INDX"},
        {"Country": "Netherlands", "Ticker": "NL", "Currency": "EUR", "Rate": 0.03, "Vol Index": "V1X.INDX"},
        {"Country": "Sweden", "Ticker": "SE", "Currency": "SEK", "Rate": 0.03, "Vol Index": "V1X.INDX"},
        {"Country": "Norway", "Ticker": "NO", "Currency": "NOK", "Rate": 0.03, "Vol Index": "V1X.INDX"},
        {"Country": "Denmark", "Ticker": "DK", "Currency": "DKK", "Rate": 0.03, "Vol Index": "V1X.INDX"},
        
        # Asian markets
        {"Country": "China", "Ticker": "CN", "Currency": "CNY", "Rate": 0.03, "Vol Index": "VXFXI.INDX"},
        {"Country": "Hong Kong", "Ticker": "HK", "Currency": "HKD", "Rate": 0.03, "Vol Index": "VHSI.INDX"},
        {"Country": "South Korea", "Ticker": "KR", "Currency": "KRW", "Rate": 0.03, "Vol Index": "VKOSPI.INDX"},
        {"Country": "India", "Ticker": "IN", "Currency": "INR", "Rate": 0.06, "Vol Index": "VIX.INDX"},
        
        # Other markets
        {"Country": "Israel", "Ticker": "IL", "Currency": "ILS", "Rate": 0.03, "Vol Index": "VIX.INDX"},
        {"Country": "Brazil", "Ticker": "BR", "Currency": "BRL", "Rate": 0.09, "Vol Index": "VXEWZ.INDX"},
        {"Country": "Mexico", "Ticker": "MX", "Currency": "MXN", "Rate": 0.08, "Vol Index": "VIX.INDX"}
    ]
    
    # Create DataFrame and save
    df = pd.DataFrame(default_data)
    df.to_csv(output_path, index=False)
    
    logging.info(f"Created default regions data file at {output_path}")
    return True

# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    create_default_regions_data()