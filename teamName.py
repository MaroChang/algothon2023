import numpy as np

nInst = 50
currentPos = np.zeros(nInst)
lookback = 20  # Period for the moving average and Fourier Transform

def fourier_smooth(prices, num_coeffs=5):
    """Smooth a price series using Fourier Transform."""
    # Compute the Fourier Transform
    coeffs = np.fft.fft(prices)
    
    # Set all but the first few coefficients to zero
    coeffs[num_coeffs:-num_coeffs] = 0
    
    # Compute the Inverse Fourier Transform
    smoothed_prices = np.fft.ifft(coeffs)
    
    return smoothed_prices.real  # Return the real part (imaginary part should be negligible)

def trend_following_strategy(prices):
    smoothed_prices = fourier_smooth(prices, lookback)
    
    # Check the last price against the smoothed series
    if prices[-1] > smoothed_prices[-1]:  # Price above smoothed series, bullish trend
        return 1
    elif prices[-1] < smoothed_prices[-1]:  # Price below smoothed series, bearish trend
        return -1
    return 0  # No clear trend

def getMyPosition(prcSoFar):
    global currentPos
    (nins, nt) = prcSoFar.shape
    if nt < lookback:  # Not enough data points to compute the trend
        return np.zeros(nins)
    
    rpos = np.zeros(nins)
    for inst in range(nins):
        price_history = prcSoFar[inst, :]
        
        decision = trend_following_strategy(price_history)
        
        # Adjust the position size if needed
        rpos[inst] = decision * 200  # Adjust this to control the amount invested/shorted
    
    currentPos = np.array([int(x) for x in currentPos + rpos])
    return currentPos
