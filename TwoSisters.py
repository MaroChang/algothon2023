# import numpy as np
# import lightgbm as lgb

# nInst = 50
# currentPos = np.zeros(nInst)
# lookback = 20  # Period for moving average and Fourier Transform

# def fourier_smooth(prices, num_coeffs=5):
#     """Smooth a price series using Fourier Transform."""
#     coeffs = np.fft.fft(prices)
#     coeffs[num_coeffs:-num_coeffs] = 0
#     smoothed_prices = np.fft.ifft(coeffs)
#     return smoothed_prices.real

# def trend_following_strategy(prices):
#     smoothed_prices = fourier_smooth(prices, lookback)
    
#     # Check the last price against the smoothed series
#     if prices[-1] > smoothed_prices[-1]:  # Price above smoothed series, bullish trend
#         return 1
#     elif prices[-1] < smoothed_prices[-1]:  # Price below smoothed series, bearish trend
#         return -1
#     return 0  # No clear trend

# def create_dataset(dataset, lookback=1):
#     X, Y = [], []
#     for i in range(len(dataset)-lookback-1):
#         a = dataset[i:(i+lookback)]
#         X.append(a)
#         Y.append(dataset[i + lookback])
#     return np.array(X), np.array(Y)

# def getMyPosition(prcSoFar):
#     global currentPos
#     (nins, nt) = prcSoFar.shape
#     if nt < lookback: 
#         return np.zeros(nins)
    
#     rpos = np.zeros(nins)
#     for inst in range(nins):
#         price_history = prcSoFar[inst, :]
        
#         # Prepare data for LightGBM
#         X, y = create_dataset(price_history, lookback)
#         X = np.array(X).reshape(-1, lookback)
        
#         if len(X) > 0:
#             train_data = lgb.Dataset(X[:-1], label=y[:-1])
#             param = {'objective': 'regression', 'boosting_type': 'gbdt', 'metric': 'l2', 'num_leaves': 31, 'learning_rate': 0.05, 'feature_fraction': 0.9}
#             num_round = 100
#             bst = lgb.train(param, train_data, num_round)

#             next_day_price_prediction = bst.predict(np.array([X[-1]]))

#             lgb_decision = 1 if next_day_price_prediction > price_history[-1] else -1 if next_day_price_prediction < price_history[-1] else 0
#             fourier_decision = trend_following_strategy(price_history)

#             # Check if both the decisions match or not
#             if lgb_decision == fourier_decision:
#                 decision = lgb_decision
#             else:
#                 decision = 0

#             rpos[inst] = decision * 200
#         else:
#             rpos[inst] = 0

#     currentPos = np.array([int(x) for x in currentPos + rpos])
#     return currentPos

import numpy as np
import xgboost as xgb

nInst = 50
currentPos = np.zeros(nInst)
lookback = 20  # Period for moving average and Fourier Transform

def fourier_smooth(prices, num_coeffs=5):
    """Smooth a price series using Fourier Transform."""
    coeffs = np.fft.fft(prices)
    coeffs[num_coeffs:-num_coeffs] = 0
    smoothed_prices = np.fft.ifft(coeffs)
    return smoothed_prices.real

def trend_following_strategy(prices):
    smoothed_prices = fourier_smooth(prices, lookback)
    
    # Check the last price against the smoothed series
    if prices[-1] > smoothed_prices[-1]:  # Price above smoothed series, bullish trend
        return 1
    elif prices[-1] < smoothed_prices[-1]:  # Price below smoothed series, bearish trend
        return -1
    return 0  # No clear trend

def create_dataset(dataset, lookback=1):
    X, Y = [], []
    for i in range(len(dataset)-lookback-1):
        a = dataset[i:(i+lookback)]
        X.append(a)
        Y.append(dataset[i + lookback])
    return np.array(X), np.array(Y)

def getMyPosition(prcSoFar):
    global currentPos
    (nins, nt) = prcSoFar.shape
    if nt < lookback: 
        return np.zeros(nins)
    
    rpos = np.zeros(nins)
    for inst in range(nins):
        price_history = prcSoFar[inst, :]
        
        # Prepare data for XGBoost
        X, y = create_dataset(price_history, lookback)
        X = np.array(X).reshape(-1, lookback)
        
        if len(X) > 0:
            model = xgb.XGBRegressor(objective="reg:squarederror", n_estimators=100)
            model.fit(X[:-1], y[:-1])

            next_day_price_prediction = model.predict(np.array([X[-1]]))

            xgb_decision = 1 if next_day_price_prediction > price_history[-1] else -1 if next_day_price_prediction < price_history[-1] else 0
            fourier_decision = trend_following_strategy(price_history)

            # Check if both the decisions match or not
            if xgb_decision == fourier_decision:
                decision = xgb_decision
            else:
                decision = 0

            rpos[inst] = decision * 200
        else:
            rpos[inst] = 0

    currentPos = np.array([int(x) for x in currentPos + rpos])
    return currentPos
