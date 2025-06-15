import requests
from typing import Dict, Optional
import os

def get_weather(city: str) -> Dict[str, str]:
    """
    Get current weather information for a specified city using OpenWeatherMap API.
    
    Args:
        city (str): Name of the city to get weather for
        
    Returns:
        Dict[str, str]: Dictionary containing weather information:
            - temperature: current temperature in Celsius
            - description: weather description
            - humidity: humidity percentage
            - wind_speed: wind speed in m/s
            
    Example:
        >>> weather = get_weather("London")
        >>> print(weather)
        {'temperature': '15.2', 'description': 'scattered clouds', 'humidity': '76', 'wind_speed': '4.12'}
        
    Note:
        Requires OPENWEATHER_API_KEY environment variable to be set
    """
    api_key = os.getenv('OPENWEATHER_API_KEY')
    if not api_key:
        raise ValueError("OPENWEATHER_API_KEY environment variable not set")
    
    base_url = "http://api.openweathermap.org/data/2.5/weather"
    params = {
        'q': city,
        'appid': api_key,
        'units': 'metric'
    }
    
    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        data = response.json()
        
        return {
            'temperature': str(data['main']['temp']),
            'description': data['weather'][0]['description'],
            'humidity': str(data['main']['humidity']),
            'wind_speed': str(data['wind']['speed'])
        }
    except requests.RequestException as e:
        raise ValueError(f"Error fetching weather data: {str(e)}") 