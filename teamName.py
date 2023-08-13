#!/usr/bin/env python
                        ###############
                        #Apply mean reversion strategy
import numpy as np

nInst = 50
currentPos = np.zeros(nInst)

def mean_reversion_strategy(price_history, window=50, z_threshold=1.0):
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

    if nt < 51:  # Need at least 51 data points to compute 50-day mean reversion
        return np.zeros(nins)
    
    for inst in range(nins):
        price_history = prcSoFar[inst, :]
        currentPos[inst] = mean_reversion_strategy(price_history)
        
    return currentPos


                        
