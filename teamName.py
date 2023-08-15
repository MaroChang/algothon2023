import numpy as np

nInst = 50
currentPos = np.zeros(nInst)

def zscore(series):
    return (series - series.mean()) / np.std(series)

def pairs_trading_strategy(prices_A, prices_B, window=50, z_threshold=1.0):
    # Calculate spread and its z-score
    spread = prices_A - prices_B
    z_scores = zscore(spread[-window:])
    current_z_score = z_scores[-1]

    # Trading logic
    if current_z_score > z_threshold:
        return -1, 1  # short A, long B
    elif current_z_score < -z_threshold:
        return 1, -1  # long A, short B
    else:
        return 0, 0  # No position

def getMyPosition(prcSoFar):
    global currentPos
    (nins, nt) = prcSoFar.shape

    if nt < 51:  # Need at least 51 data points to compute 50-day mean reversion
        return np.zeros(nins)
    
    # Assume asset 0 and asset 1 are our correlated pair (this is just an example)
    position_A, position_B = pairs_trading_strategy(prcSoFar[0], prcSoFar[1])
    currentPos[0] = position_A
    currentPos[1] = position_B

    # Rest of the assets remain neutral
    for inst in range(2, nins):
        currentPos[inst] = 0
        
    return currentPos