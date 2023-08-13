#!/usr/bin/env python
                        ###############
                        #Apply mean reversion strategy
# import numpy as np

# nInst = 50
# currentPos = np.zeros(nInst)

# def mean_reversion_strategy(price_history, window=50, z_threshold=1.0):
#     mean = price_history[-window:].mean()
#     std = price_history[-window:].std()
#     z_score = (price_history[-1] - mean) / std

#     if z_score > z_threshold:
#         return -1  # Short position
#     elif z_score < -z_threshold:
#         return 1   # Long position
#     else:
#         return 0   # No position
    

# def getMyPosition(prcSoFar):
#     global currentPos
#     (nins, nt) = prcSoFar.shape

#     if nt < 51:  # Need at least 51 data points to compute 50-day mean reversion
#         return np.zeros(nins)
    
#     for inst in range(nins):
#         price_history = prcSoFar[inst, :]
#         currentPos[inst] = mean_reversion_strategy(price_history)
        
#     return currentPos


                        
import numpy as np
from sklearn.ensemble import RandomForestRegressor

nInst = 50
currentPos = np.zeros(nInst)
lookback = 50

def generate_features(price_history):
    returns = np.diff(price_history)
    lagged_returns = np.roll(returns, 1)
    lagged_returns[0] = 0
    return np.column_stack((lagged_returns, returns))

def mean_reversion_strategy(price_history, window=lookback, z_threshold=1.0):
    mean = price_history[-window:].mean()
    std = price_history[-window:].std()
    z_score = (price_history[-1] - mean) / std

    if z_score > z_threshold:
        return -1  # Short position
    elif z_score < -z_threshold:
        return 1   # Long position
    else:
        return 0   # No position

def getMyPosition(prcSoFar):
    global currentPos
    (nins, nt) = prcSoFar.shape

    if nt < 51:  # Need at least 51 data points to compute 50-day mean reversion and generate features
        return np.zeros(nins)
    
    for inst in range(nins):
        price_history = prcSoFar[inst, :]
        decision = mean_reversion_strategy(price_history)
        
        if decision == 0:
            X = generate_features(price_history)
            y = np.roll(price_history, -1)[:-1]  # shifted price history as target
            
            model = RandomForestRegressor(n_estimators=100)
            model.fit(X[:-1], y[:-1])  # Train using all data points except the last one
            
            forecast = model.predict([X[-1]])
            if forecast > price_history[-1]:  # If forecasted price is higher than the last observed price
                currentPos[inst] = 1
            elif forecast < price_history[-1]:  # If forecasted price is lower than the last observed price
                currentPos[inst] = -1
            else:
                currentPos[inst] = 0
        else:
            currentPos[inst] = decision
        
    return currentPos
