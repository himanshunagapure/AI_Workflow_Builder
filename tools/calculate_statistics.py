from typing import List, Union, Dict
import statistics

def calculate_statistics(values: List[Union[int, float]]) -> Dict[str, float]:
    """
    Calculate basic statistical measures for a list of numerical values.
    
    Args:
        values (List[Union[int, float]]): List of numerical values to analyze
        
    Returns:
        Dict[str, float]: Dictionary containing statistical measures:
            - mean: arithmetic mean
            - median: middle value
            - mode: most common value
            - std_dev: standard deviation
            - variance: statistical variance
            
    Example:
        >>> values = [1, 2, 2, 3, 4, 5]
        >>> stats = calculate_statistics(values)
        >>> print(stats)
        {'mean': 2.8333333333333335, 'median': 2.5, 'mode': 2, 'std_dev': 1.329160519, 'variance': 1.766666667}
    """
    if not values:
        raise ValueError("Input list cannot be empty")
    
    try:
        return {
            'mean': statistics.mean(values),
            'median': statistics.median(values),
            'mode': statistics.mode(values),
            'std_dev': statistics.stdev(values),
            'variance': statistics.variance(values)
        }
    except statistics.StatisticsError as e:
        raise ValueError(f"Error calculating statistics: {str(e)}") 