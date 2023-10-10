import pandas as pd
import numpy as np
from pandas_datareader import data as pdr
import yfinance as yf
import datetime as dt
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import PercentFormatter
import math
from itertools import combinations
import json
yf.pdr_override()
np.random.seed(0)
pd.set_option('display.max_columns', 500)
# Import Data
def get_data(stocks, start, end):
    stockData = pdr.get_data_yahoo(stocks, start, end)
    stockData = stockData['Adj Close']
    pctChange = stockData.pct_change()
    meanReturns = pctChange.mean()
    covMatrix = pctChange.cov()
    return meanReturns, covMatrix, stockData
stockList = ['AAPL', 'MSFT', 'AMZN', 'GOOG', 'META']
stocks = [stock for stock in stockList]
endDate = dt.datetime.now()
startDate = endDate - dt.timedelta(days=300)  # 300 is important for covar
meanReturns, covMatrix, stockData = get_data(stocks, startDate, endDate)
# Define porfolio weights
weights = np.random.dirichlet(np.ones(len(meanReturns)))
returns = stockData.pct_change()
'''
Monte Carlo Method Simulation 
Number of simulations = 100
Time range = 100
'''
nSims = 100
T = 100
# Mean Returns Matrix
meanM = np.full(shape=(T, len(weights)), fill_value=meanReturns)
meanM = meanM.T
portfolio_sims = np.full(shape=(T, nSims), fill_value=0.0)
initialPortfolioValue = 10000

for sim in range(nSims):
    # MC Loops
    Z = np.random.normal(size=(T, len(weights)))
    L = np.linalg.cholesky(covMatrix)
    dailyReturns = meanM + np.inner(L, Z)
    portfolio_sims[:, sim] = np.cumprod(np.inner(weights, dailyReturns.T) + 1) * initialPortfolioValue

# Get max and min values for the portfolio at the end of
# the next 100 days should be valued at max and shouldn't drop below
# GA will try to minimize the difference between the
# max and min values
# Shapley value will look at which stock contributed the most
# Ideally want difference to be as small as possible

# max_sim = np.max(portfolio_sims[-1, :])
# min_sim = np.min(portfolio_sims[-1, :])
# difference = abs(max_sim - min_sim)
# print('Max Value = $', round(max_sim, 2))
# print('Min Value = $', round(min_sim, 2))
# print('Difference = $', round(difference, 2))
def plotPrices(stockPrices):
    plt.figure(figsize=(15, 10))
    plt.plot(stockPrices)
    plt.ylabel('Stock Price ($)')
    plt.xlabel('Days')
    plt.xticks(rotation=45)
    plt.title('Adjusted Close Price History')
    plt.legend(stockPrices.columns, loc='upper left')
    plt.show()
    return

plotPrices(stockData)
''''
To determine worst-case scenario for portfolio
Need to figure out VAR and cVAR
tells us how much we might lose in the case
the market or the portfolio 
tanks or dropped to a certain level
'''
def mcVAR(returns, alpha=5):
    '''
    Monte Carlo Value at Risk
    :param returns: Portfolio returns at n-Day Level
    :param alpha: Confidence level
    :return: output: percentile on return distribution to
                     a given confidence level alpha
    '''
    if isinstance(returns, pd.Series):
        return np.percentile(returns, alpha)
    else:
        raise TypeError('Expected returns to be pandas.Series')

def mcCVAR(returns, alpha=5):
    '''
    Monte Carlo Conditional Value at Risk
    Expected shortfall past a given confidence level
    :param returns: Portfolio returns at n-Day Level
    :param alpha: Confidence level
    :return: CVAR or Expected Shortfall to given alpha
    '''
    if isinstance(returns, pd.Series):
        belowVAR = returns <= mcVAR(returns, alpha=alpha)
        return returns[belowVAR].mean()
    else:
        raise TypeError('Expected returns to be pandas.Series')


# Since define value for actual portfolio need
# to take away initial value
# could also do on percentage change
# Var = limit
# CVAR = expected shortfall to that percentile

def plotDistribution(portSims, initialPortfolioValue, alpha, verbose=False):
    portfolioResults = pd.Series(portSims[-1, :])
    var = initialPortfolioValue - mcVAR(portfolioResults, alpha=alpha)
    cvar = initialPortfolioValue - mcCVAR(portfolioResults, alpha=alpha)

    print(f'Value at Risk: ${format(round(var, 2))}')
    print(f'Conditional Value at Risk: ${format(round(cvar, 2))}')

    plt.figure(figsize=(10, 8))
    plt.plot(portSims)
    plt.axhline(y=initialPortfolioValue - var, color='r', linestyle='-', label='VaR')
    plt.axhline(y=initialPortfolioValue - cvar, color='g', linestyle='-', label='CVaR')
    plt.ylabel('Portfolio Value ($)')
    plt.xlabel('Days')
    plt.title(f'MC Simulation of Portfolio\n VAR= \${round(var, 2)} and CVAR = \${round(cvar, 2)}')
    if verbose:
        plt.show()

    Q1 = np.quantile(portfolioResults, 0.25)
    Q3 = np.quantile(portfolioResults, 0.75)
    IQR = Q3 - Q1
    cube = np.cbrt(len(portfolioResults))
    binwidth = math.ceil(2 * IQR / cube)
    fig, ax1 = plt.subplots(figsize=(10, 8))
    sns.histplot(portfolioResults, kde=True, ax=ax1, stat='probability', color="blue",
                 label="Probabilities", binwidth=binwidth, alpha=0.25)

    ax2 = ax1.twinx()
    sns.kdeplot(portfolioResults, label='KDE Density', color="lightblue", lw=3, ax=ax2)
    ax2.set_ylim(0, ax1.get_ylim()[1] / binwidth)  # similar limits on the y-axis to align the plots
    ax2.yaxis.set_major_formatter(PercentFormatter(1 / binwidth))  # show axis such that 1/binwidth corresponds to 100%
    ax2.set_ylabel(f'Probability for a bin width of {binwidth}')

    ax2_x = ax2.lines[-1].get_xdata()
    ax2_y = ax2.lines[-1].get_ydata()

    ax2.fill_between(ax2_x, 0, ax2_y, where=ax2_x <= (initialPortfolioValue - cvar),
                     color='r', alpha=0.4, interpolate=True)
    ax2.fill_between(ax2_x, 0, ax2_y, where=ax2_x >= (initialPortfolioValue - cvar),
                     color='orange', alpha=0.4, interpolate=True)
    ax2.fill_between(ax2_x, ax2_y, where=ax2_x >= (initialPortfolioValue - var),
                     color='lightblue', alpha=0.7, interpolate=True)
    ax2.vlines(initialPortfolioValue, 0, np.interp(initialPortfolioValue, ax2_x, ax2_y),
               color='black', linestyle='--', label=f'Initial Value = ${initialPortfolioValue}', lw=2)
    ax2.vlines((initialPortfolioValue - var), 0, np.interp((initialPortfolioValue - var), ax2_x, ax2_y),
               color='orange', linestyle='--', label=f'IV - VaR = ${round(initialPortfolioValue - var, 2)}', lw=2)
    ax2.vlines((initialPortfolioValue - cvar), 0, np.interp((initialPortfolioValue - cvar), ax2_x, ax2_y),
               color='red', linestyle='--', label=f'IV - CVaR = ${round(initialPortfolioValue - cvar, 2)}', lw=2)

    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')
    plt.title(f'Portfolio Value Distribution\n VaR = \${round(var, 2)}, CVaR = \${round(cvar, 2)}, alpha = {alpha}')
    plt.xlabel('Portfolio Value ($)')
    if verbose:
        plt.show()
    return

plotDistribution(portfolio_sims, initialPortfolioValue, alpha=5, verbose=True)

def mcSim(stockData, weights, initPV, nSims, TRange):
    pctChange = stockData.pct_change()
    meanReturns = pctChange.mean()
    covMatrix = pctChange.cov()
    # Mean Returns Matrix
    meanM = np.full(shape=(TRange, len(weights)), fill_value=meanReturns)
    meanM = meanM.T
    portSims = np.full(shape=(TRange, nSims), fill_value=0.0)
    for sim in range(nSims):
        # MC Loops
        Z = np.random.normal(size=(TRange, len(weights)))
        L = np.linalg.cholesky(covMatrix)
        dailyReturns = meanM + np.inner(L, Z)
        portSims[:, sim] = np.cumprod(np.inner(weights, dailyReturns.T) + 1) * initPV
    return portSims

res = mcSim(stockData, weights, 10000, 100, 100)


def portfolio_metrics(weights, returns, index='Trial'):
    cov = returns.cov()

    rp = (returns.mean() * 252) @ weights

    port_var = weights @ (cov * 252) @ weights
    rf = 0.02 # risk-free rate
    sharpe = (rp - rf) / np.sqrt(port_var)
    df = pd.DataFrame({"Expected Return": rp,
                       "Portfolio Variance": port_var,
                       'Portfolio Std': np.sqrt(port_var),
                       'Sharpe Ratio': sharpe}, index=[index])
    return df

def mcWeights(stockList, nSims, returns):
    portfolios = pd.DataFrame(
        columns=[*stockList, "Expected Return", "Portfolio Variance", "Portfolio Std", "Sharpe Ratio"])
    for i in range(nSims):
        weights = np.random.random(len(stockList))
        weights /= np.sum(weights)
        portfolios.loc[i, stockList] = weights
        metrics = portfolio_metrics(weights, returns)
        portfolios["Expected Return"][i] = metrics["Expected Return"][0]
        portfolios["Portfolio Variance"][i] = metrics["Portfolio Variance"][0]
        portfolios["Portfolio Std"][i] = metrics["Portfolio Std"][0]
        portfolios["Sharpe Ratio"][i] = metrics["Sharpe Ratio"][0]
    return portfolios

stockList = ['AAPL', 'MSFT', 'AMZN', 'GOOG', 'META']
stocks = [stock for stock in stockList]
endDate = dt.datetime.now()
startDate = endDate - dt.timedelta(days=300)  # 300 is important for covar
# startDate = '2017-01-01'
# endDate = '2021-01-01'
meanReturns, covMatrix, stockData = get_data(stocks, startDate, endDate)
# Define porfolio weights
weights = np.random.dirichlet(np.ones(len(meanReturns)))
returns = stockData.pct_change()
'''
Monte Carlo Method Simulation 
Number of simulations = 100
Time range = 100
'''
nSims = 100
T = 100
# Mean Returns Matrix
meanM = np.full(shape=(T, len(weights)), fill_value=meanReturns)
meanM = meanM.T
portfolio_sims = np.full(shape=(T, nSims), fill_value=0.0)
initialPortfolioValue = 10000
port_res = mcWeights(stocks, 1000, returns)
print(port_res.head())
print(port_res[port_res['Sharpe Ratio'] == port_res['Sharpe Ratio'].max()])

def withinRisk(weights, returns):
    # Variance = sigma ** 2
    risk = {}
    assert len(weights) == len(returns.columns)
    for i in range(len(returns.columns)):
        sigma2 = np.var(returns.iloc[:][returns.columns[i]])* weights[i]**2
        risk[returns.columns[i]] = sigma2
    return risk

riskW = withinRisk(weights, returns)

# Systematic Risk
def betweenRisk(weights, returns):
    combList = []
    wcVar = {}
    for n in range(len(returns.columns)+ 1):
        combList += combinations(returns.columns, n)
    for comb in combList:
        if len(comb) >= 1:
            r = []
            ravg = []
            weightsList = []
            for i in range(len(comb)):
                r.append(np.array(returns[comb[i]]))
                ravgV = np.nanmean(r[i])
                ravg.append(ravgV)
            for stock in comb:
                weightsList.append(weights[returns.columns.get_loc(stock)])
            diffList = []
            for i in range(len(comb)):
                diff = returns[comb[i]] - ravg[i]
                diffList.append(diff)
            diffDF = pd.DataFrame(diffList).T
            multDF = diffDF.prod(1, skipna=False)
            weightsDF = pd.DataFrame(weightsList).T
            sum = np.nansum(multDF)
            covar = sum / (len(multDF) - 1)
            wcovar = covar * weightsDF.prod(1).values[0]
            str = ''
            for i in comb:
                str += f'{i}/'
            str = str[:-1]
            wcVar[f'wCov_{str}'] = wcovar
        else:
            wcVar['wCov_'] = 0

    allAssets = returns.columns.to_list()
    shapleyValues = {}
    for i in allAssets:
        shapley = 0
        res1 = {key: value for key, value in wcVar.items() if i in key}
        res2 = {key: value for key, value in wcVar.items() if i not in key}
        for s1, s2 in zip(res1.items(), res2.items()):
            text = s2[0][6:]
            if len(text) > 0:
                text = text.split('/')
                combS = len(text)
            else:
                combS = 0
            scaling = (math.comb(len(allAssets)-1, combS)**-1)
            shapley += scaling * (s1[1] - s2[1])
        shapleyValues[i] = shapley / len(allAssets)

    return wcVar, shapleyValues
btwCVAR, shapVals = betweenRisk(weights, returns)


print('Risk within\n', json.dumps(riskW, indent=4))
print('Weighted Between Covariance\n', json.dumps(btwCVAR, indent= 4))
print('Shapley Values\n', json.dumps(shapVals, indent=4))
print('Sum of Shapley Values\n', sum(shapVals.values()))
print('Between Risk Total\n', btwCVAR['wCov_AAPL/AMZN/GOOG/META/MSFT'])