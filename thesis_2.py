#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 17 17:07:58 2018

@author: caoxin
"""
import os
os.getcwd()
os.chdir('/Users/caoxin/desktop/thesis')
from scipy import stats
import pandas as pd
import numpy as np
import quandl
import matplotlib.pyplot as plt
quandl.ApiConfig.api_key = '4pzuGmC_14GAhdgSUA-1'
import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault)

# get the table for daily stock prices and,
# filter the table for selected tickers, columns within a time range
# set paginate to True because Quandl limits tables API to 10,000 rows per call
selected= ['GOOG', 'GE', 'WMT', 'PG']
data = quandl.get_table('WIKI/PRICES', ticker = selected, 
                        qopts = { 'columns': ['ticker', 'date', 'adj_close'] }, 
                        date = { 'gte': '2013-12-18', 'lte': '2018-12-18' }, 
                        paginate=True)
data.head()

# create a new dataframe with 'date' column as index
new = data.set_index('date')

# use pandas pivot function to sort adj_close by tickers
clean_data = new.pivot(columns='ticker')
clean_data = clean_data.dropna()
#drop the first level index
clean_data.columns = clean_data.columns.droplevel()
#clean the index
clean_data.index.name = None
#delete the column name
del clean_data.columns.name
# check the head of the output
clean_data.head(5)
#go back to the easy from
data=clean_data
#%%plot the return
(data/data.ix[0]*100).plot(figsize=(9,6),grid=True)
plt.title('Stocks Price 2014-2019')
plt.tight_layout()
plt.savefig('price', dpi=150)
#%%plot the return
log_returns = np.log(data / data.shift(1))
log_returns.head()
log_returns.dropna()
log_returns.hist(bins=100, figsize=(9, 6))
plt.savefig('returns', dpi=150)
describe=log_returns.describe().T
#%%
#drop the na
GE=log_returns['GE'][1:]
GOOG=log_returns['GOOG'][1:]
PG=log_returns['PG'][1:]
WMT=log_returns['WMT'][1:]

fig = plt.figure(figsize=(9, 6))

ax1 =fig.add_subplot(221)
x1= np.linspace(GE.min(), GE.max(), len(GE))
loc, scale = stats.norm.fit(GE)
param_density1 = stats.norm.pdf(x1, loc=loc, scale=scale)
ax1.hist(GE, bins=50,density=True)
ax1.plot(x1, param_density1, 'r-')
ax1.set_title('GE')

ax2 =fig.add_subplot(222)
x2= np.linspace(GOOG.min(), GOOG.max(), len(GOOG))
loc, scale = stats.norm.fit(GOOG)
param_density2 = stats.norm.pdf(x2, loc=loc, scale=scale)
ax2.hist(GOOG, bins=50,density=True)
ax2.plot(x2, param_density2, 'r-')
ax2.set_title('GOOG')

ax3 =fig.add_subplot(223)
x3= np.linspace(PG.min(), PG.max(), len(PG))
loc, scale = stats.norm.fit(PG)
param_density3 = stats.norm.pdf(x3, loc=loc, scale=scale)
ax3.hist(PG, bins=50,density=True)
ax3.plot(x3, param_density3,'r-')
ax3.set_title('PG')

ax4 =fig.add_subplot(224)
x4= np.linspace(WMT.min(), WMT.max(), len(WMT))
loc, scale = stats.norm.fit(WMT)
param_density4 = stats.norm.pdf(x4, loc=loc, scale=scale)
ax4.hist(WMT, bins=50,density=True)
ax4.plot(x4, param_density4,'r-')
ax4.set_title('WMT')

plt.tight_layout()
plt.show()
fig.savefig('Histogram-Normal', dpi=150)
#%%covraince portfolio
returns_annual = log_returns.mean() * 250

cov_annual=log_returns.cov()*250
# empty lists to store returns, volatility and weights of imiginary portfolios
port_returns = []
port_volatility = []
sharpe_ratio = []
stock_weights = []

return_VaR_ratio= []
port_VaR = []

return_VaR_ratio2=[]
port_VaR2=[]

return_VaR_ratio3=[]
port_VaR3=[]


# set the number of combinations for imaginary portfolios
num_assets = 4
num_portfolios = 50000

#set random seed for reproduction's sake
np.random.seed(101)

# populate the empty lists with each portfolios returns,risk and weights
for single_portfolio in range(num_portfolios):
    weights = np.random.random(num_assets)
    weights /= np.sum(weights)
    returns = np.dot(weights, returns_annual)
    volatility = np.sqrt(np.dot(weights.T, np.dot(cov_annual, weights)))
    sharpe = returns / volatility
    sharpe_ratio.append(sharpe)
    port_returns.append(returns)
    port_volatility.append(volatility)
    stock_weights.append(weights)
    #VaR
    VaR= np.sqrt(np.dot(weights.T, np.dot(cov_annual, weights)))*(1.65)- np.dot(weights, returns_annual)
    retrun_VaR = returns / VaR
    return_VaR_ratio.append(retrun_VaR)
    port_VaR.append(VaR)
    #Var 97.5%
    VaR2= np.sqrt(np.dot(weights.T, np.dot(cov_annual, weights)))*(1.96)- np.dot(weights, returns_annual)
    retrun_VaR2 = returns / VaR2
    return_VaR_ratio2.append(retrun_VaR2)
    port_VaR2.append(VaR2)
    #var 99%
    VaR3= np.sqrt(np.dot(weights.T, np.dot(cov_annual, weights)))*(2.33)- np.dot(weights, returns_annual)
    retrun_VaR3 = returns / VaR
    return_VaR_ratio3.append(retrun_VaR3)
    port_VaR3.append(VaR3)

# a dictionary for Returns and Risk values of each portfolio
portfolio = {'Returns': port_returns,
             'Volatility': port_volatility,
             'Sharpe Ratio': sharpe_ratio,
             'Value at Risk 95%': port_VaR,
             'return_VaR_ratio':return_VaR_ratio,
             'Value at Risk 97.5%': port_VaR2,
             'return_VaR_ratio2':return_VaR_ratio2,
             'Value at Risk 99%': port_VaR2,
             'return_VaR_ratio3':return_VaR_ratio3
             }

# extend original dictionary to accomodate each ticker and weight in the portfolio
for counter,symbol in enumerate(selected):
    portfolio[symbol+' Weight'] = [Weight[counter] for Weight in stock_weights]

# make a nice dataframe of the extended dictionary
df = pd.DataFrame(portfolio)

# get better labels for desired arrangement o columns
column_order = ['Returns', 'Volatility', 'Sharpe Ratio','return_VaR_ratio','Value at Risk 95%','Value at Risk 97.5%',
                'return_VaR_ratio2', 'Value at Risk 99%', 'return_VaR_ratio3' ] + [stock+' Weight' for stock in selected]

# reorder dataframe columns
df = df[column_order]

# plot frontier, max sharpe & min Volatility values with a scatterplot

#df.plot.scatter(x='Volatility', y='Returns', c='Sharpe Ratio',cmap='RdYlGn', marker= 'o',grid=True)
#plt.xlabel('Volatility (Std. Deviation)')
#plt.ylabel('Expected Returns')
#plt.title('Efficient Frontier for Volatility')
volatility=df['Volatility']
returns= df['Returns']
sharpe_ratio=df['Sharpe Ratio']
value_at_risk=df['Value at Risk 95%']
return_VaR_ratio=df['return_VaR_ratio']

fig = plt.figure(figsize=(12, 6))

ax1 =fig.add_subplot(121)
ax1.scatter(x=volatility, y=returns, c=sharpe_ratio,cmap='RdYlGn', marker= 'o')
ax1.set_xlabel('Volatility (Std. Deviation)')
ax1.set_ylabel('Expected Returns')
ax1.set_title('Efficient Frontier for Volatility')


ax2=fig.add_subplot(122)
ax2.scatter(x=value_at_risk, y=returns, c=return_VaR_ratio, cmap='RdYlGn',marker= 'o')
ax2.set_xlabel("Value at Risk 95%")
#ax2.set_ylabel('Expected Returns')
ax2.set_title('Efficient Frontier for VaR')

plt.tight_layout()
plt.show()

fig.savefig('mean-var',dpi=150)
#%%97.5% and 99%
returns= df['Returns']
value_at_risk2=df['Value at Risk 97.5%']
return_VaR_ratio2=df['return_VaR_ratio2']
value_at_risk3=df['Value at Risk 99%']
return_VaR_ratio3=df['return_VaR_ratio3']

fig1 = plt.figure(figsize=(12, 6))

ax1 =fig1.add_subplot(121)
ax1.scatter(x=value_at_risk2, y=returns, c=return_VaR_ratio2,cmap='RdYlGn', marker= 'o')
ax1.set_xlabel('VaR at 97.5%')
ax1.set_ylabel('Expected Returns')
ax1.set_title('Efficient Frontier for VaR')


ax2=fig1.add_subplot(122)
ax2.scatter(x=value_at_risk3, y=returns, c=return_VaR_ratio3, cmap='RdYlGn',marker= 'o')
ax2.set_xlabel("Value at Risk 99%")
#ax2.set_ylabel('Expected Returns')
ax2.set_title('Efficient Frontier for VaR')

plt.tight_layout()
plt.show()

fig1.savefig('var_comparison',dpi=150)
#%%WITH RISK FREE RATE
port_returns1=[]
stock_weights1=[]
return_VaR_ratio1=[]
port_VaR1=[]
returns_annual1=returns_annual.copy(deep=True)
returns_annual1['risk free']=0.01

port_returns2=[]
return_VaR_ratio2=[]
port_VaR2=[]
returns_annual2=returns_annual.copy(deep=True)
returns_annual2['risk free']=0.04
# set the number of combinations for imaginary portfolios
num_assets = 4
num_portfolios = 50000

#set random seed for reproduction's sake
np.random.seed(129)
for single_portfolio in range(num_portfolios):
    weights1 = np.random.random(num_assets+1)
    weights1 /= np.sum(weights1)
    returns1= np.dot(weights1, returns_annual1)
    port_returns1.append(returns1)
    stock_weights1.append(weights1)
    #VaR
    VaR1= np.sqrt(np.dot(weights1[1:].T, np.dot(cov_annual, weights1[1:])))*(1.65)- np.dot(weights1, returns_annual1)
    retrun_VaR1= returns1 / VaR1
    return_VaR_ratio1.append(retrun_VaR1)
    port_VaR1.append(VaR1)
    
    returns2= np.dot(weights1, returns_annual2)
    port_returns2.append(returns2)
    VaR2= np.sqrt(np.dot(weights1[1:].T, np.dot(cov_annual, weights1[1:])))*(1.65)- np.dot(weights1, returns_annual2)
    retrun_VaR2= returns2 / VaR2
    return_VaR_ratio2.append(retrun_VaR2)
    port_VaR2.append(VaR2)
    
portfolio1 = {'Returns1': port_returns1,
              'Value at Risk 95% 1': port_VaR1,
             'return_VaR_ratio1':return_VaR_ratio1,
             'Returns2': port_returns2,
             'Value at Risk 95% 2': port_VaR2,
             'return_VaR_ratio2':return_VaR_ratio2}

selected1 = list(selected)
selected1.append('RF')
# extend original dictionary to accomodate each ticker and weight in the portfolio
for counter,symbol in enumerate(selected1):
    portfolio1[symbol+' Weight'] = [Weight[counter] for Weight in stock_weights1]

# make a nice dataframe of the extended dictionary
df1 = pd.DataFrame(portfolio1)
column_order = ['Returns1','return_VaR_ratio1','Value at Risk 95% 1','Returns2','return_VaR_ratio2',
                'Value at Risk 95% 2'] + [stock+' Weight' for stock in selected1]

# reorder dataframe columns
df1 = df1[column_order]

returns1= df1['Returns1']
value_at_risk1=df1['Value at Risk 95% 1']
return_VaR_ratio1=df1['return_VaR_ratio1']

returns2= df1['Returns2']
value_at_risk2=df1['Value at Risk 95% 2']
return_VaR_ratio2=df1['return_VaR_ratio2']


fig2 = plt.figure(figsize=(12, 6))

ax1 =fig2.add_subplot(121)
ax1.scatter(x=value_at_risk1, y=returns1, c=return_VaR_ratio1,cmap='RdYlGn', marker= 'o')
ax1.set_xlabel('VaR at 95%')
ax1.set_ylabel('Expected Returns')
ax1.set_title('Efficient Frontier with Low Risk Free Rate Bond(0.01) ')

ax2 =fig2.add_subplot(122)
ax2.scatter(x=value_at_risk2, y=returns2, c=return_VaR_ratio2,cmap='RdYlGn', marker= 'o')
ax2.set_xlabel('VaR at 95%')
#ax2.set_ylabel('Expected Returns')
ax2.set_title('Efficient Frontier with High Risk Free Rate Bond(0.04)')

plt.tight_layout()
plt.show()

fig2.savefig('with_risk_free',dpi=150)
#%%CALCULATE THE VAR
GE_rets = log_returns['GE'][1:]
GE_rets = GE_rets.sort_values(axis=0, ascending=True)
VaR_GE = -GE_rets.quantile(0.05)*np.sqrt(250)

GOOG_rets = log_returns['GOOG'][1:]
GOOG_rets = GOOG_rets.sort_values(axis=0, ascending=True)
VaR_GOOG = -GOOG_rets.quantile(0.05)*np.sqrt(250)

PG_rets = log_returns['PG'][1:]
PG_rets = PG_rets.sort_values(axis=0, ascending=True)
VaR_PG = -PG_rets.quantile(0.05)*np.sqrt(250)

WMT_rets = log_returns['WMT'][1:]
WMT_rets = WMT_rets.sort_values(axis=0, ascending=True)
VaR_WMT = -WMT_rets.quantile(0.05)*np.sqrt(250)
VaR=[VaR_GE,VaR_GOOG,VaR_PG,VaR_WMT]

#%%For parametric
VaR_GE_p = (-GE.mean()+1.65*GE.std())*np.sqrt(250)
VaR_GOOG_p = (-GOOG.mean()+1.65*GOOG.std())*np.sqrt(250)
VaR_PG_p = (-PG.mean()+1.65*PG.std())*np.sqrt(250)
VaR_WMT_p = (-WMT.mean()+1.65*WMT.std())*np.sqrt(250)
#%%t distribution

from scipy.stats import skew, kurtosis, kurtosistest
from scipy.stats import norm, t
dx = 0.0001  # resolution
x1 = np.linspace(GE.min(), GE.max(), len(GE))
parm1 = t.fit(GE_rets )
nu1, mu_t1, sig_t1 = parm1
pdf1 = t.pdf(x1, nu1, mu_t1, sig_t1)
print("Integral t.pdf(x1; mu1, sig1) dx = %.2f" % (np.sum(pdf1*dx)))
print("nu1 = %.2f" % nu1)
print()
# Compute VaR
alpha = 0.05 
lev = 100*(1-alpha)
mu_norm1, sig_norm1 = norm.fit(GE_rets)
h = 1  # days
StudenthVaR1 = (h*(nu1-2)/nu1)**0.5 * t.ppf(1-alpha, nu1)*sig_norm1 - h*mu_norm1
print("%g%% %g-day GE Student t VaR = %.6f%%" % (lev, h, StudenthVaR1*np.sqrt(250)))

x2 = np.linspace(GOOG.min(), GOOG.max(), len(GOOG))
parm2 = t.fit(GOOG_rets )
nu2, mu_t2, sig_t2 = parm2
pdf2 = t.pdf(x2, nu2, mu_t2, sig_t2)
print("Integral t.pdf(x2; mu2, sig2) dx = %.2f" % (np.sum(pdf2*dx)))
print("nu2 = %.2f" % nu2)
print()
# Compute VaR
mu_norm2, sig_norm2 = norm.fit(GOOG_rets)
StudenthVaR2 = (h*(nu2-2)/nu2)**0.5 * t.ppf(1-alpha, nu2)*sig_norm2 - h*mu_norm2
print("%g%% %g-day GOOG Student t VaR = %.6f%%" % (lev, h, StudenthVaR2*np.sqrt(250)))

x3 = np.linspace(PG.min(), PG.max(), len(PG))
parm3 = t.fit(PG_rets )
nu3, mu_t3, sig_t3 = parm3
pdf3 = t.pdf(x3, nu3, mu_t3, sig_t3)
print("Integral t.pdf(x3; mu3, sig3) dx = %.2f" % (np.sum(pdf3*dx)))
print("nu3 = %.2f" % nu3)
print()
# Compute VaR
mu_norm3, sig_norm3 = norm.fit(PG_rets)
StudenthVaR3 = (h*(nu3-2)/nu3)**0.5 * t.ppf(1-alpha, nu3)*sig_norm3 - h*mu_norm3
print("%g%% %g-day PG Student t VaR = %.6f%%" % (lev, h, StudenthVaR3*np.sqrt(250)))

x4 = np.linspace(WMT.min(), WMT.max(), len(WMT))
parm4 = t.fit(WMT_rets )
nu4, mu_t4, sig_t4 = parm4
pdf4 = t.pdf(x4, nu4, mu_t4, sig_t4)
print("Integral t.pdf(x4; mu4, sig4) dx = %.2f" % (np.sum(pdf4*dx)))
print("nu4 = %.2f" % nu4)
print()
# Compute VaR
mu_norm4, sig_norm4 = norm.fit(WMT_rets)
StudenthVaR4 = (h*(nu4-2)/nu4)**0.5 * t.ppf(1-alpha, nu4)*sig_norm4 - h*mu_norm4
print("%g%% %g-day WMT Student t VaR = %.6f%%" % (lev, h, StudenthVaR4*np.sqrt(250)))
#%%  t distribution

fig3 = plt.figure(figsize=(9, 6))

ax1 =fig3.add_subplot(221)
ax1.hist(GE, bins=50,density=True)
ax1.plot(x1, pdf1, 'r-')
ax1.set_title('GE')


ax2 =fig3.add_subplot(222)
ax2.hist(GOOG, bins=50,density=True)
ax2.plot(x2, pdf2, 'r-')
ax2.set_title('GOOG')

ax3 =fig3.add_subplot(223)
ax3.hist(PG, bins=50,density=True)
ax3.plot(x3, pdf3, 'r-')
ax3.set_title('PG')

ax4 =fig3.add_subplot(224)
ax4.hist(WMT, bins=50,density=True)
ax4.plot(x4, pdf4, 'r-')
ax4.set_title('WMT')

plt.tight_layout()
plt.show()

fig3.savefig('student_t',dpi=150)

#%% EWMA
from arch import arch_model
returns=GOOG
am = arch_model(returns, vol='Garch', p=1, o=0, q=1, dist='Normal')
res = am.fit(update_freq=5)
forecasts = res.forecast()
print(forecasts.mean.iloc[-3:])
print(forecasts.residual_variance.iloc[-3:])
print(forecasts.variance.iloc[-3:])
print(res.summary())
#%%LONGER HORIZON
import sys
forecasts = res.forecast(horizon=5)
print(forecasts.residual_variance.iloc[-3:])
#%%%
index = returns.index
start_loc = 0
end_loc = np.where(index >= '2016-8-13')[0].min()
forecasts = {}
for i in range(350):
    sys.stdout.write('.')
    sys.stdout.flush()
    res = am.fit(first_obs=i, last_obs=i+end_loc, disp='off')
    temp = res.forecast(horizon=3).variance
    fcast = temp.iloc[i+end_loc-1]
    forecasts[fcast.name] = fcast
print()
print(pd.DataFrame(forecasts).T)
pred=pd.DataFrame(forecasts).T

pred['GARCH VaR']=np.sqrt(pred['h.1'])*1.65-0.000584
#%%moving average 
log_returns['MA'] = log_returns['GOOG'].rolling(window=100,center=False).mean()
log_returns['STD'] = log_returns['GOOG'].rolling(window=100,center=False).std()
log_returns['MA100']= 1.65*log_returns['STD']-log_returns['MA']
#%%plot VaR
import matplotlib.dates as mdates


dailyreturn=log_returns['GOOG'][600:950]
MA100=log_returns['MA100'][600:950]
garch=pred['GARCH VaR']
plt.figure(figsize=(8,4))
plt.plot(dailyreturn,color='lightgray',label="")
plt.plot(garch)
plt.plot(MA100,label="MA100 VaR")
plt.axhline(y=0.0153563,color='r', linestyle='-',label='Staic VaR')
plt.xticks(rotation=60)
plt.ylabel("Daily Return")
plt.legend()
plt.tight_layout()
plt.savefig('garch',dpi=150)
plt.show()
#%%PLOT VARIANCE
ABSO=abs(dailyreturn)
MA100STD=log_returns['STD'][600:950]
GARCHSTD=np.sqrt(pred['h.1'])
plt.figure(figsize=(8,4))
plt.plot(ABSO,color='lightgray',label="")
plt.plot(GARCHSTD,label="GARCH STD")
plt.plot(MA100STD,label="MA100 STD")
plt.axhline(y=0.0098636,color='r', linestyle='-',label='Staic STD')
plt.xticks(rotation=60)
plt.ylabel("Daily  Absolute Return")
plt.legend()
plt.tight_layout()
plt.savefig('STD',dpi=150)
plt.show()
#%%no trade area
log_returns['MA50'] = log_returns['GOOG'].rolling(window=50,center=False).mean()
log_returns['STD50'] = log_returns['GOOG'].rolling(window=50,center=False).std()
log_returns['MA50']= 1.65*log_returns['STD50']-log_returns['MA50']
#plot
dailyreturn=log_returns['GOOG'][600:]

MA100=log_returns['MA100'][600:]
MA50=log_returns['MA50'][600:]
plt.figure(figsize=(8,4))
plt.plot(dailyreturn,color='lightgray',label="")
plt.plot(MA100,label="MA100 VaR")
plt.plot(MA50,label="MA50 VaR")
#plt.plot(garch)
#plt.axhline(y=0.0153563,color='r', linestyle='-',label='Staic VaR')
plt.xticks(rotation=60)
plt.ylabel("Daily Return")
plt.legend()
plt.tight_layout()
plt.savefig('notrade',dpi=150)
plt.show()