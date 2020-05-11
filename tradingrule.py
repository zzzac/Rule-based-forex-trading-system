import numpy as np
import pandas as pd
from ta import *

def Rule1(param, OHLC):
    # Rule 1: Simple Moving Average Crossover
    # Input: Close prices, MA periods 1 and 2
    # Return: training periods accumulated returns
    ma1, ma2 = param
    open, high, low, close = OHLC
    logr = np.log(close/close.shift(1))
    s1 = close.rolling(ma1).mean()
    s2 = close.rolling(ma2).mean()
    signal = 2*(s1<s2).shift(1)-1
    port_logr = signal*logr
    return (abs(port_logr.sum()), signal)

def Rule2(param, OHLC):
    # Rule 2: EMA and close
    # Input: Close prices, EMA periods 1 and 2
    # Return: training periods accumulated returns
    ema1, ma2 = param
    open, high, low, close = OHLC
    logr = np.log(close/close.shift(1))
    s1 = ema(close, ema1)
    s2 = close.rolling(ma2).mean()
    signal = 2*(s1<s2).shift(1)-1
    port_logr = signal*logr
    return (abs(port_logr.sum()), signal)

def Rule3(param, OHLC):
    # Rule 3: EMA and EMA
    # Input: Close prices, EMA periods 1 and 2
    # Return: training periods accumulated returns
    ema1, ema2 = param
    open, high, low, close = OHLC
    logr = np.log(close/close.shift(1))
    s1 = ema(close, ema1)
    s2 = ema(close, ema2)
    signal = 2*(s1<s2).shift(1)-1
    port_logr = signal*logr
    return (abs(port_logr.sum()), signal)

def Rule4(param, OHLC):
    # Rule 4: DEMA and MA
    dema1, ma2 = param
    open, high, low, close = OHLC
    logr = np.log(close/close.shift(1))
    s1 = DEMA(close, dema1)
    s2 = close.rolling(ma2).mean()
    signal = 2*(s1<s2).shift(1)-1
    port_logr = signal*logr
    return (abs(port_logr.sum()), signal)

def Rule5(param, OHLC):
    # Rule 5: DEMA and DEMA
    dema1, dema2 = param
    open, high, low, close = OHLC
    logr = np.log(close/close.shift(1))
    s1 = DEMA(close, dema1)
    s2 = DEMA(close, dema2)
    signal = 2*(s1<s2).shift(1)-1
    port_logr = signal*logr
    return (abs(port_logr.sum()), signal)

def Rule6(param, OHLC):
    # Rule 6: TEMA and ma crossovers
    tema1, ma2 = param
    open, high, low, close = OHLC
    logr = np.log(close/close.shift(1))
    s1 = TEMA(close, tema1)
    s2 = close.rolling(ma2).mean()
    signal = 2*(s1<s2).shift(1)-1
    port_logr = signal*logr
    return (abs(port_logr.sum()), signal)

def Rule7(param, OHLC):
    stoch1, stochma2 = param
    open, high, low, close = OHLC
    logr = np.log(close/close.shift(1))
    s1 = stoch(high, low, close, stoch1)
    s2 = s1.rolling(stochma2, min_periods=0).mean()
    signal = 2*(s1<s2).shift(1)-1
    port_logr = signal*logr
    return (abs(port_logr.sum()), signal)

def Rule8(param, OHLC):
    vortex1, vortex2 = param
    open, high, low, close = OHLC
    logr = np.log(close/close.shift(1))
    s1 = vortex_indicator_pos(high, low, close, vortex1)
    s2 = vortex_indicator_neg(high, low, close, vortex2)
    signal = 2*(s1<s2).shift(1)-1
    port_logr = signal*logr
    return (abs(port_logr.sum()), signal)

def Rule9(param, OHLC):
    p1, p2 = param
    open, high, low, close = OHLC
    logr = np.log(close/close.shift(1))
    s1 = ichimoku_a(high, low, n1=p1, n2=round((p1+p2)/2))
    s2 = ichimoku_b(high, low, n2=round((p1+p2)/2), n3=p2)
    s3 = close
    signal = (-1*((s3>s1) & (s3>s2))+1*((s3<s2) & (s3<s1))).shift(1)
    port_logr = signal*logr
    return (abs(port_logr.sum()), signal)

# Type 2 Rules:
# > RSI
# > CCI *High must be greater than low

def Rule10(param, OHLC):
    rsi1, c2 = param
    open, high, low, close = OHLC
    logr = np.log(close/close.shift(1))
    s1 = rsi(close, rsi1)
    s2 = c2
    signal = 2*(s1<s2).shift(1)-1
    port_logr = signal*logr
    return (abs(port_logr.sum()), signal)

def Rule11(param, OHLC):
    cci1, c2 = param
    open, high, low, close = OHLC
    logr = np.log(close/close.shift(1))
    s1 = cci(high, low, close, cci1)
    s2 = c2
    signal = 2*(s1<s2).shift(1)-1
    port_logr = signal*logr
    return (abs(port_logr.sum()), signal)


# Type 3 Rules:
# > RSI
# > CCI
# ** High must be greater than low

def Rule12(param, OHLC):
    rsi1, hl, ll = param
    open, high, low, close = OHLC
    logr = np.log(close/close.shift(1))
    s1 = rsi(close, rsi1)
    signal = (-1*(s1>hl)+1*(s1<ll)).shift(1)
    port_logr = signal*logr
    return (abs(port_logr.sum()), signal)

def Rule13(param, OHLC):
    cci1, hl, ll = param
    open, high, low, close = OHLC
    logr = np.log(close/close.shift(1))
    s1 = cci(high, low, close, cci1)
    signal = (-1*(s1>hl)+1*(s1<ll)).shift(1)
    port_logr = signal*logr
    return (abs(port_logr.sum()), signal)


# Type 4 Rules:
# > Bollinger-bands high, low
# > keltner_channel
# > donchian_channel
# > ichimoko a and b

def Rule14(period, OHLC):
    open, high, low, close = OHLC
    logr = np.log(close/close.shift(1))
    s1 = keltner_channel_hband(high, low, close, n=period)
    s2 = keltner_channel_lband(high, low, close, n=period)
    s3 = close
    signal = (-1*(s3>s1)+1*(s3<s2)).shift(1)
    port_logr = signal*logr
    return (abs(port_logr.sum()), signal)

def Rule15(period, OHLC):
    open, high, low, close = OHLC
    logr = np.log(close/close.shift(1))
    s1 = donchian_channel_hband(close, n=period)
    s2 = donchian_channel_hband(close, n=period)
    s3 = close
    signal = (-1*(s3>s1)+1*(s3<s2)).shift(1)
    port_logr = signal*logr
    return (abs(port_logr.sum()), signal)

def Rule16(period, OHLC):
    open, high, low, close = OHLC
    logr = np.log(close/close.shift(1))
    s1 = bollinger_hband(close, n=period)
    s2 = bollinger_lband(close, n=period)
    s3 = close
    signal = (-1*(s3>s1)+1*(s3<s2)).shift(1)
    port_logr = signal*logr
    return (abs(port_logr.sum()), signal)


def trainTradingRuleFeatures(df):
    '''
    input:  df, a dataframe contains OHLC columns
    output: Rule_params, the parameters for 16 trading rules
    '''
    
    OHLC = [df.Open, df.High, df.Low, df.Close]
    periods = [1, 3, 5, 7, 11, 15, 19, 23, 27, 35, 41, 50, 61]
    
    type1 = [Rule1, Rule2, Rule3, Rule4, Rule5, Rule6, Rule7, Rule8, Rule9]
    type1_param = []
    type1_score = []
    for rule in type1:
        best = -1
        for i in range(len(periods)):
            for j in range(i, len(periods)):
                param = (periods[i], periods[j])
                score = rule(param, OHLC)[0]
                if score>best:
                    best = score
                    best_param = (periods[i], periods[j])
        type1_param.append(best_param)
        type1_score.append(best)

    rsi_limits = list(range(0,101,5))
    cci_limits = list(range(-120, 121, 20))
    limits = [rsi_limits, cci_limits]

    type2 = [Rule10, Rule11]
    type2_param = []
    type2_score = []

    for i in range(len(type2)):
        rule = type2[i]
        params = limits[i]
        best = -1
        for period in periods:
            for p in params:
                param = (period, p)
                score = rule(param, OHLC)[0]
                if score > best:
                    best = score
                    best_param = (period, p)
        type2_param.append(best_param)
        type2_score.append(best)

    type3 = [Rule12, Rule13]
    type3_param = []
    type3_score = []

    for i in range(len(type3)):
        rule = type3[i]
        params = limits[i]
        n = len(params)
        best = -1
        for period in periods:
            for lb in range(n-1):
                for ub in range(lb+1, n):
                    param = (period, params[ub], params[lb])
                    score = rule(param, OHLC)[0]
                    if score>best:
                        best = score
                        best_param = (period, params[ub], params[lb])
        type3_param.append(best_param)
        type3_score.append(best)


    type4 = [Rule14, Rule15, Rule16]
    type4_param = []
    type4_score = []

    for rule in type4:
        best = -1
        for i in periods:
            score = rule(i, OHLC)[0]
            if score>best:
                best = score
                best_param = i
        type4_param.append(best_param)
        type4_score.append(best)

    All_Rules = type1+type2+type3+type4
    Rule_params = type1_param+type2_param+type3_param+type4_param
    Rule_scores = type1_score+type2_score+type3_score+type4_score

    for i in range(len(All_Rules)):
        print('Training Rule{} score is: {:.3f}'.format(i+1, Rule_scores[i]))
        
    return Rule_params

def getTradingRuleFeatures(df, Rule_params):
    '''
    input: df, a dataframe contains OHLC columns
           Rule_params, the parameters for 16 trading rules
    output: trading_rule_df, a new dataframe contains the trading rule features only.
    '''
    OHLC = [df.Open, df.High, df.Low, df.Close]
    logr = np.log(df.Close/df.Close.shift(1))
    
    All_Rules = [Rule1, Rule2, Rule3, Rule4, Rule5, Rule6, Rule7, Rule8, Rule9, Rule10, Rule11, \
                 Rule12, Rule13, Rule14, Rule15, Rule16]
    
    trading_rule_df = pd.DataFrame({'logr': logr})
    for i in range(len(All_Rules)):
        trading_rule_df['Rule'+str(i+1)] = All_Rules[i](Rule_params[i], OHLC)[1]
    trading_rule_df.dropna(inplace = True)
    return trading_rule_df