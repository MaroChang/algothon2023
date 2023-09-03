import numpy as np
from sklearn.linear_model import Lasso

nInst = 50
currentPos = np.zeros(nInst)
lookback = 20

def fourier_smooth(prices, num_coeffs=5):
    """Smooth a price series using Fourier Transform."""
    coeffs = np.fft.fft(prices)
    coeffs[num_coeffs:-num_coeffs] = 0
    smoothed_prices = np.fft.ifft(coeffs)
    return smoothed_prices.real

def simple_moving_average(prices, window_size):
    """Compute Simple Moving Average."""
    return np.convolve(prices, np.ones(window_size)/window_size, mode='valid')

def create_features(price_history):
    """Creates features and target variable from a price history."""
    smoothed_prices = fourier_smooth(price_history)
    
    X = []
    y = []
    sma_5 = simple_moving_average(smoothed_prices, 5)
    sma_10 = simple_moving_average(smoothed_prices, 10)
    
    max_lookback = max([lookback, len(sma_5), len(sma_10)])
    
    for i in range(len(price_history) - max_lookback):
        combined_features = np.concatenate((
            price_history[i:i+lookback],
            sma_5[i:i+lookback-(5-1)],  # Adjust index due to SMA window
            sma_10[i:i+lookback-(10-1)]  # Adjust index due to SMA window
        ))
        X.append(combined_features)
        y.append(price_history[i+max_lookback])
    
    return np.array(X), np.array(y)

def lasso_regression_strategy(price_history, alpha=0.1):
    """Predicts the next price using Lasso Regression and returns trading decision."""
    X, y = create_features(price_history)
    
    if len(X) == 0:
        return 0
    
    model = Lasso(alpha=alpha, max_iter=10000).fit(X, y)
    smoothed_prices = fourier_smooth(price_history)
    sma_5_last = simple_moving_average(smoothed_prices, 5)[-lookback+(5-1):]
    sma_10_last = simple_moving_average(smoothed_prices, 10)[-lookback+(10-1):]
    combined_features = np.concatenate((price_history[-lookback:], sma_5_last, sma_10_last))
    next_price_pred = model.predict([combined_features])[0]
    
    if next_price_pred > price_history[-1]:
        return 1
    elif next_price_pred < price_history[-1]:
        return -1
    return 0

def getMyPosition(prcSoFar):
    global currentPos
    (nins, nt) = prcSoFar.shape
    
    # Adjust the minimum required data points considering the longest SMA
    if nt < lookback + 10:  
        return np.zeros(nins)
    
    rpos = np.zeros(nins)
    for inst in range(nins):
        price_history = prcSoFar[inst, :]
        decision = lasso_regression_strategy(price_history)
        rpos[inst] = decision * 200
    
    currentPos = np.array([int(x) for x in currentPos + rpos])
    return currentPos


