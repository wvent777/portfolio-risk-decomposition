import GASHAP as gas
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.collections import PolyCollection

f = open('SimulationRes.json', 'r')
simRes = json.load(f)
f.close()

MCReturns = {}
GlobalVAR = {}
GlobalCovar = {}
GlobalShapeley = {}
for i in range(5):
    MCReturns[i] = simRes[f"Portfolio_{i}"]['MC Returns']
    GlobalVAR[i] = simRes[f"Portfolio_{i}"]['VaR']
    GlobalCovar[i] = simRes[f"Portfolio_{i}"]['CVaR']
    GlobalShapeley[i] = simRes[f"Portfolio_{i}"]['Shapley Values'] #Shapley Values of Last Generation

#[individual, generation]
gas.plotDistribution(MCReturns[0][1], initPV=10000, alpha=5, verbose=True)
gas.plotDistribution(MCReturns[0][-1], initPV=10000, alpha=5, verbose=True)


# Plot Var and CVar
GlobalVarDF = pd.DataFrame.from_dict(GlobalVAR)
GlobalVarDF = GlobalVarDF.subtract(10000, axis = 1)
GlobalVarDF['Mean'] = GlobalVarDF.mean(axis = 1)
GlobalVarDF['Max'] = GlobalVarDF.max(axis = 1)
GlobalVarDF['Min'] = GlobalVarDF.min(axis = 1)
GlobalVarDF['Median'] = GlobalVarDF.median(axis = 1)
GlobalCovarDF = pd.DataFrame.from_dict(GlobalCovar)
GlobalCovarDF = GlobalCovarDF.subtract(10000, axis=0)
GlobalCovarDF['Mean'] = GlobalCovarDF.mean(axis = 1)
GlobalCovarDF['Max'] = GlobalCovarDF.max(axis = 1)
GlobalCovarDF['Min'] = GlobalCovarDF.min(axis = 1)
GlobalCovarDF['Median'] = GlobalCovarDF.median(axis=1)

ax = GlobalVarDF['Mean'].plot(color = 'blue', label = 'Mean VaR')
ax.fill_between(GlobalVarDF.index, GlobalVarDF['Min'], GlobalVarDF['Max'], alpha = 0.2)
ax.hlines(0, 0, len(GlobalVarDF['Mean']), color='black', linestyle='--', linewidth=1)
ax.plot(GlobalVarDF['Median'], color='b', label='Median CVaR', linestyle='--', linewidth=1)

ax.plot(GlobalCovarDF['Mean'], color='red', label='Mean CVaR')
ax.plot(GlobalCovarDF['Median'], color='red', label='Median CVaR', linestyle='--', linewidth=1)
ax.fill_between(GlobalCovarDF.index, GlobalCovarDF['Min'], GlobalCovarDF['Max'],
                   color='red', alpha = 0.2)
ax.hlines(0, 0, len(GlobalCovarDF['Mean']), color='black', linestyle='--', linewidth=1)
ax.set_xlim(0, len(GlobalCovarDF['Mean']))
plt.title(f"Portfolio Risk per Generation")
plt.xlabel("Generation")
plt.ylabel("Risk($)")
plt.legend()
plt.tight_layout()
# plt.show()

# Plot Mean Returns Standard Dev
# Looking at Portfolio 0
MCReturnsDF = pd.DataFrame.from_dict(MCReturns[0]).T

x = np.arange(0, len(MCReturnsDF))
z = np.arange(0, len(MCReturnsDF.columns))

def draw_3d_plot(data):
    fig, ax = plt.subplots(figsize=(12, 8))

    sns.kdeplot(data, label='KDE Density', lw=0.7,
                legend=True, palette="flare", multiple="stack", fill=True, alpha=0.8, ax=ax,
                common_norm=False, gridsize=len(data), bw_method='scott', levels=1000)
    plt.xlabel('Portfolio Value ($)')
    plt.ylabel('Density')
    plt.title('Portfolio Value Density Distribution')
    ax.set_xlim(0, 45000)
    plt.tight_layout()
    plt.show()

# MCReturnsClean = MCReturnsDF.dropna().reset_index(drop=True)
# print(MCReturnsClean)
# draw_3d_plot(MCReturnsClean)

# Plot Shapley Values
gas.plotShap(GlobalShapeley[0])
