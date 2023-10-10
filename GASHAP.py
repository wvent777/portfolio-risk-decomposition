import pandas as pd
import numpy as np
from pandas_datareader import data as pdr
import yfinance as yf
import datetime as dt
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import PercentFormatter
import math
import random
from itertools import combinations
from tabulate import tabulate as tb
import matplotlib.colors as colors
from matplotlib import cm
import matplotlib as mpl
import json
import heapq
import random
mpl.warnings.filterwarnings("ignore")
yf.pdr_override()
np.random.seed(0)
mpl.rcParams.update(mpl.rcParamsDefault)

# Import Data
def get_data(stocks, start, end):
    stockData = pdr.get_data_yahoo(stocks, start, end)
    stockData = stockData['Adj Close']
    pctChange = stockData.pct_change()
    meanReturns = pctChange.mean()
    covMatrix = pctChange.cov()
    return meanReturns, covMatrix, stockData, pctChange


def plotPrices(stockPrices, verbose=False):
    plt.figure(figsize=(10, 8))
    plt.plot(stockPrices)
    plt.ylabel('Stock Price ($)', fontsize=14)
    plt.xlabel('Days', fontsize=14)
    plt.xticks(rotation=45)
    plt.title('Adjusted Close Price History', fontsize=16)
    plt.legend(stockPrices.columns, loc='upper left')
    plt.tight_layout()
    if verbose:
        plt.show()
    plt.close()
    return


def mcSim(meanReturns, covMatrix, weights, initPV, nSims, TRange):
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


def mcVAR(simReturns: pd.Series, alpha: float) -> float:
    """
    Monte Carlo Value at Risk
    :param simReturns: Portfolio returns at n-Day Level
    :param alpha: Confidence level
    :return: output: percentile on return distribution to
                     a given confidence level alpha
    """
    if isinstance(simReturns, pd.Series):
        return np.percentile(simReturns, alpha)
    else:
        raise TypeError('Expected returns to be pandas.Series')


def mcCVAR(simReturns: pd.Series, alpha: float) -> float:
    """
    Monte Carlo Conditional Value at Risk
    Expected shortfall past a given confidence level
    :param simReturns: Portfolio returns at n-Day Level
    :param alpha: Confidence level
    :return: CVAR or Expected Shortfall to given alpha
    """
    if isinstance(simReturns, pd.Series):
        belowVAR = simReturns <= mcVAR(simReturns, alpha=alpha)
        return simReturns[belowVAR].mean()
    else:
        raise TypeError('Expected returns to be pandas.Series')


def plotDistribution(portSims, initPV, alpha, verbose=False):
    if isinstance(portSims, pd.Series):
        pass
    elif isinstance(portSims, list):
        portSims = pd.Series(portSims)
    else:
        portSims = pd.Series(portSims[-1, :])
    var = initPV - mcVAR(portSims, alpha=alpha)
    cvar = initPV - mcCVAR(portSims, alpha=alpha)
    median_ret = np.percentile(portSims, 50)
    var2 = mcVAR(portSims, alpha=alpha)
    cvar2 =mcCVAR(portSims, alpha=alpha)

    holding_period = len(portSims)/365
    e = (1/holding_period)
    arr =  (((median_ret)/initPV) ** e)-1
    print("arr", arr)
    if verbose:
        print(f'Value at Risk: ${format(round(var2, 2))}')
        print(f'Conditional Value at Risk: ${format(round(cvar2, 2))}')

    plt.figure(figsize=(10, 8))
    plt.plot(portSims)
    plt.axhline(y=initPV - var, color="r", linestyle='-', label='VaR')
    plt.axhline(y=initPV - cvar, color='g', linestyle='-', label='CVaR')
    plt.ylabel('Portfolio Value ($)')
    plt.xlabel('Days')
    plt.title(f'MC Simulation of Portfolio\n VAR= \${round(var, 2)} and CVAR = \${round(cvar, 2)}')
    plt.tight_layout()
    if verbose: plt.show()
    else: plt.close()
    Q1 = np.quantile(portSims, 0.25)
    Q3 = np.quantile(portSims, 0.75)
    IQR = Q3 - Q1
    cube = np.cbrt(len(portSims))
    binwidth = math.ceil(2 * IQR / cube)
    fig, ax1 = plt.subplots(figsize=(10, 8))
    sns.histplot(portSims, kde=True, ax=ax1, stat='probability', color="blue",
                 label="Probabilities", binwidth=binwidth, alpha=0.25)

    ax2 = ax1.twinx()
    sns.kdeplot(portSims, label='KDE Density', color="lightblue", lw=3, ax=ax2)
    ax2.set_ylim(0, ax1.get_ylim()[1] / binwidth)  # similar limits on the y-axis to align the plots
    ax2.yaxis.set_major_formatter(
        PercentFormatter(1 / binwidth))  # show axis such that 1/binwidth corresponds to 100%
    ax2.set_ylabel(f'Probability for a bin width of {binwidth}')
    ax2.set_xlim(0, ax1.get_xlim()[1])  # align the x-axis limits of both plots
    ax2_x = ax2.lines[-1].get_xdata()
    ax2_y = ax2.lines[-1].get_ydata()
    med_prob = np.interp(median_ret, ax2_x, ax2_y)
    ax2.hlines(med_prob,0, median_ret,color='magenta', linestyle='--', lw=1)
    ax2.vlines(median_ret, 0, np.interp(median_ret, ax2_x, ax2_y),
                color='magenta', linestyle='--', label=f'Median = ${round(median_ret, 2)}', lw=2)

    ax2.fill_between(ax2_x, 0, ax2_y, where=ax2_x <= cvar2,
                     color='r', alpha=0.4, interpolate=True)
    ax2.fill_between(ax2_x, 0, ax2_y, where=ax2_x >= cvar2,
                     color='orange', alpha=0.4, interpolate=True)
    ax2.fill_between(ax2_x, ax2_y, where=ax2_x >= var2,
                     color='lightblue', alpha=0.7, interpolate=True)
    ax2.vlines(initPV, 0, np.interp(initPV, ax2_x, ax2_y),
               color='black', linestyle='--', label=f'Initial Value = ${initPV}', lw=2)
    ax2.vlines((initPV - var), 0, np.interp((initPV - var), ax2_x, ax2_y),
               color='orange', linestyle='--', label=f' VaR = ${round(initPV - var, 2)}', lw=2)
    ax2.vlines((initPV - cvar), 0, np.interp((initPV - cvar), ax2_x, ax2_y),
               color='red', linestyle='--', label=f'CVaR = ${round(initPV - cvar, 2)}', lw=2)

    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')
    plt.title(f'Portfolio Value Distribution for {len(portSims)} Days Out\n '
              f'VaR = \${round(var2, 2)}, CVaR = \${round(cvar2, 2)}, alpha = {alpha}\n'
              f'Annualized Return = {round(arr*100, 2)}%')
    plt.xlabel('Portfolio Value ($)')
    print(med_prob)
    if verbose: plt.show()
    else: plt.close()
    return


def portfolio_metrics(weights, returns, timeRange, index='Trial'):
    cov = returns.cov()
    rp = (returns.mean() * timeRange) @ weights

    port_var = weights @ (cov * timeRange) @ weights
    rf = 0.0697  # risk-free rate as per google
    sharpe = (rp - rf) / np.sqrt(port_var)
    df = pd.DataFrame({"Expected Return": rp,
                       "Portfolio Variance": port_var,
                       'Portfolio Std': np.sqrt(port_var),
                       'Sharpe Ratio': sharpe}, index=[index])
    return df


def mcWeights(stockList, returns, nSims, timeRange):
    portfolios = pd.DataFrame(
        columns=[*stockList, "Expected Return", "Portfolio Variance", "Portfolio Std", "Sharpe Ratio"])
    for i in range(nSims):
        weights = np.random.random(len(stockList))
        weights /= np.sum(weights)
        portfolios.loc[i, stockList] = weights
        metrics = portfolio_metrics(weights, returns, timeRange)
        portfolios["Expected Return"][i] = metrics["Expected Return"][0]
        portfolios["Portfolio Variance"][i] = metrics["Portfolio Variance"][0]
        portfolios["Portfolio Std"][i] = metrics["Portfolio Std"][0]
        portfolios["Sharpe Ratio"][i] = metrics["Sharpe Ratio"][0]
    return portfolios


def withinRisk(weights, returns)->dict:
    # Variance = sigma ** 2
    risk = {}
    assert len(weights) == len(returns.columns)
    for i in range(len(returns.columns)):
        sigma2 = np.var(returns.iloc[:][returns.columns[i]]) * weights[i] ** 2
        risk[returns.columns[i]] = sigma2
    return risk

def betweenRisk(weights, returns):
    combList = []
    wcVar = {}
    for n in range(len(returns.columns) + 1):
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
            scaling = (math.comb(len(allAssets) - 1, combS) ** -1)
            shapley += scaling * (s1[1] - s2[1])
        shapleyValues[i] = shapley / len(allAssets)

    return wcVar, shapleyValues

def plotShap(shapValues):
    plt.style.use('seaborn-v0_8-bright')
    shapDF = pd.DataFrame(shapValues.items(), columns=['Asset', 'Shapley Value'])
    shapDF = shapDF.sort_values(by='Shapley Value', ascending=True)

    f, ax = plt.subplots(figsize=(8,8))
    total = sum(shapDF['Shapley Value'])
    cmp = mpl.colormaps['bwr']
    norm= mpl.colors.Normalize(vmin=shapDF['Shapley Value'].min(), vmax=shapDF['Shapley Value'].max())

    shapDF.plot.barh(x='Asset', y='Shapley Value', ax=ax, color=cmp(norm(shapDF['Shapley Value'])), legend=False)
    ax.set(ylabel="Portfolio Stocks", xlabel="Shapley Value")

    for i in range(len(ax.containers)):
        ax.bar_label(ax.containers[i], fmt='%.2e', color='black', label_type='center', size=15)

    ax.set_xscale('symlog')
    ax.vlines(total, -1, len(shapDF), linestyles='dashed', colors='black',label='Total Shapley Value')
    ax.text(total, 0, r'$\sigma_{total}^2=$'+f'{total:.2e}', rotation = 0, color='black', size=14)
    plt.title('Shapley Values for Portfolio Risk')
    plt.show()
    return

def GAInit(Stocks, portfolioSize, start, end, initPV, alpha, popSize, gens, verbose=False):
    pop = {}
    # Initialize Population
    for i in range(popSize):
        portfolio = random.sample(Stocks, portfolioSize)
        meanReturns, covMatrix, stockPrices, returns = get_data(portfolio, start, end)
        # Monte Carlo Simulation for Weights
        nSims = 10000
        timeRange = len(stockPrices)
        portWeights = mcWeights(portfolio, returns, nSims, timeRange)
        bestWeight = portWeights[portWeights['Sharpe Ratio'] == portWeights['Sharpe Ratio'].max()]
        weights = bestWeight.iloc[0, 0:portfolioSize].to_list()
        # Run Monte Carlo Simulation for Potential returns for all nSims
        # Only care about last row of returns (last day) for plotting
        mcReturns = mcSim(meanReturns, covMatrix, weights, initPV, nSims, timeRange)
        # Plot Initial MC Simulation
        plotDistribution(mcReturns, initPV, alpha, verbose=verbose)
        print(mcReturns[-1, :])
        # Calculate VaR and CVaR
        cvar = mcCVAR(pd.Series(mcReturns[-1, :]), alpha)
        var = mcVAR(pd.Series(mcReturns[-1, :]), alpha)
        # Get initial specific risk (Within Risk)
        specRisk = withinRisk(weights, returns)
        # Get initial systematic risk (Between Risk)
        sysRisk, shapleyValues = betweenRisk(weights, returns)
        # plotShap(shapleyValues)

        pop[f"Portfolio_{i}"] = {'Portfolio': portfolio,
                                 'Weights': weights,
                                 'VaR': [],
                                 'CVaR': [],
                                 'Specific Risk': [],
                                 'Systematic Risk': [],
                                 'Shapley Values': shapleyValues,
                                 'MC Returns': [],}
        pop[f"Portfolio_{i}"]['VaR'].append(var)
        pop[f"Portfolio_{i}"]['CVaR'].append(cvar)
        pop[f"Portfolio_{i}"]['Specific Risk'].append(specRisk)
        pop[f"Portfolio_{i}"]['Systematic Risk'].append(sysRisk)
        pop[f"Portfolio_{i}"]['MC Returns'].append(mcReturns[-1, :].tolist())

    # Main GA Loop:
    for i in range(gens):
        if verbose:
            print(f'Generation {i+1}')
            print('----------------')
            # print(json.dumps(pop[f"Portfolio_{0}"], indent=6))
        for j in range(popSize):
            shapleyValues = pop[f"Portfolio_{j}"]['Shapley Values']
            portfolio = pop[f"Portfolio_{j}"]['Portfolio']
            weights = pop[f"Portfolio_{j}"]['Weights']
            specRisk = pop[f"Portfolio_{j}"]['Specific Risk']
            sysRisk = pop[f"Portfolio_{j}"]['Systematic Risk']
            mcReturns = pop[f"Portfolio_{j}"]['MC Returns']
            StockChoices = [i for i in Stocks if i not in portfolio]

            # higher shapley value = higher risk
            # replace 3 highest stock
            if random.random() > 0.3:
                n = 1
            else :
                n = 3
            worstStocks = heapq.nlargest(n, shapleyValues, key=shapleyValues.get)
            choices = random.sample(StockChoices, n)
            for k, v in pop[f"Portfolio_{j}"].items():
                for i in range(len(worstStocks)):
                    if worstStocks[i] in v:
                        if isinstance(v, list):
                            v.remove(worstStocks[i])
                            v.append(choices[i])
                        elif isinstance(v, dict):
                            v[choices[i]] = v.pop(worstStocks[i])
                        else:
                            print('error')

            meanReturns, covMatrix, stockPrices, returns = get_data(pop[f"Portfolio_{j}"]["Portfolio"], start, end)
            # Monte Carlo Simulation for Weights
            nSims = 1000
            timeRange = len(stockPrices)
            portWeights = mcWeights(pop[f"Portfolio_{j}"]["Portfolio"], returns, nSims, timeRange)
            bestWeight = portWeights[portWeights['Sharpe Ratio'] == portWeights['Sharpe Ratio'].max()]
            pop[f"Portfolio_{j}"]['Weights'] = bestWeight.iloc[0, 0:portfolioSize].to_list()
            mcReturns = mcSim(meanReturns, covMatrix, pop[f"Portfolio_{j}"]['Weights'], initPV, nSims, timeRange)
            cvar = mcCVAR(pd.Series(mcReturns[-1, :]), alpha)
            var = mcVAR(pd.Series(mcReturns[-1, :]), alpha)
            specRisk = withinRisk(pop[f"Portfolio_{j}"]['Weights'], returns)
            sysRisk, shapleyValues = betweenRisk(pop[f"Portfolio_{j}"]['Weights'], returns)
            print(mcReturns[-1,:].tolist())
            pop[f"Portfolio_{j}"]['VaR'].append(var)
            pop[f"Portfolio_{j}"]['CVaR'].append(cvar)
            pop[f"Portfolio_{j}"]['Specific Risk'].append(specRisk)
            pop[f"Portfolio_{j}"]['Systematic Risk'].append(sysRisk)
            pop[f"Portfolio_{j}"]['MC Returns'].append(mcReturns[-1, :].tolist())
            pop[f"Portfolio_{j}"]['Shapley Values'] = shapleyValues
    if verbose:
        plotPrices(stockPrices)
        print(f'Initial Portfolio Value: ${initPV}')
        print(f'Optimal Portfolio Weights via MC:\n{weights}')
        print(f'Initial PV - VaR: ${var}')
        print(f'Initial PV - CVar: ${cvar}')
        print(f'Initial Specific Risk (wVar):')
        print(tb(pd.DataFrame(specRisk, index=[0]).T, headers=['Stocks', 'Specific Risk (wVar)'], tablefmt="psql"))
        print(f'Initial Systematic Risk (wCVar):')
        print(tb(pd.DataFrame(sysRisk, index=[0]).T, headers=['Stock Combinations', 'wCVar'], tablefmt="psql"))
        print(f'Initial Shapley Values:')
        print(tb(pd.DataFrame(shapleyValues, index=[0]).T, headers=['Stocks', 'Shapley Value'], tablefmt="psql"))
        print('Total value', sum(shapleyValues.values()))
    return pop


if __name__ == '__main__':
    blueChipTickers = ['AAPL', 'MSFT', 'AMZN',
                      'GOOG', 'GOOGL', 'BRK-B',
                      'NVDA', 'META', 'TSLA',
                      'V', 'UNH',
                      'XOM', 'TSM', 'JNJ',
                      'WMT', 'JPM', 'LLY',
                      'PG', 'MA', 'NVO',
                      'CVX', 'MRK', 'HD', 'KO']

    endDate = dt.datetime.now()
    startDate = endDate - dt.timedelta(days=365 * 3)

    pop = GAInit(blueChipTickers, 3, startDate, endDate, 10000, alpha=5, popSize=3, gens=10,  verbose=True)

