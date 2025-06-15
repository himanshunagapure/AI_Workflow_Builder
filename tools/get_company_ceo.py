import yfinance as yf

def get_company_ceo(symbol: str) -> str:
    """
    Get the CEO information for a given company.
    
    Args:
        symbol (str): The stock ticker symbol (e.g., 'AAPL', 'GOOGL', 'MSFT')
        
    Returns:
        str: A formatted string containing the CEO's name, or an error message if unsuccessful
        
    Example:
        >>> get_company_ceo("AAPL")
        'The CEO of AAPL is Tim Cook.'
    """
    symbol = symbol.strip().upper()
    try:
        stock = yf.Ticker(symbol)
        ceo = stock.info.get("companyOfficers", [{}])[0].get("name", "Unknown")
        return f"The CEO of {symbol} is {ceo}."
    except Exception as e:
        return f"ERROR: Failed to get CEO for {symbol}: {e}"
    
#print(get_company_ceo("AAPL"))