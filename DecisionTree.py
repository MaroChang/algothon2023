import numpy as np
from sklearn.tree import DecisionTreeClassifier

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

def generate_features_labels(prices):
    X, y = [], []
    for i in range(len(prices) - lookback - 1):
        X.append(prices[i:i+lookback])
        y.append(1 if prices[i+lookback] > prices[i+lookback-1] else -1)
    return np.array(X), np.array(y)

def decision_tree_strategy(prices):
    smoothed_prices = fourier_smooth(prices)
    X, y = generate_features_labels(smoothed_prices)
    
    if len(X) == 0:
        return 0

    model = DecisionTreeClassifier()
    model.fit(X, y)
    prediction = model.predict([smoothed_prices[-lookback:]])
    return prediction[0]

def getMyPosition(prcSoFar):
    global currentPos
    (nins, nt) = prcSoFar.shape
    
    if nt < lookback + 2:  # Need at least lookback + 2 days for a meaningful prediction
        return np.zeros(nins)
    
    rpos = np.zeros(nins)
    for inst in range(nins):
        price_history = prcSoFar[inst, :]
        
        decision = decision_tree_strategy(price_history)
        
        rpos[inst] = decision * 200  # Adjust this to control the amount invested/shorted
    
    currentPos = np.array([int(x) for x in currentPos + rpos])
    return currentPos
