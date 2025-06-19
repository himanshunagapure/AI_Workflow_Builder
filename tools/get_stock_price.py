# get_stock_price.py
import yfinance as yf

def get_stock_price(ticker: str) -> float | None:
    """
    Fetch the current stock price for a given ticker symbol.
    
    Args:
        ticker (str): The stock ticker symbol (e.g., 'AAPL', 'GOOGL', 'MSFT')
        
    Returns:
        float | None: The current stock price if successful, None if there's an error
        
    Example:
        >>> price = get_stock_price("AAPL")
        >>> print(price)
        150.25
    """
    try:
        stock = yf.Ticker(ticker)
        price = stock.info['regularMarketPrice']
        return price
    except Exception as e:
        print(f"Error: {e}")
        return None