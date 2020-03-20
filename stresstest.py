#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
import quandl
import math
from math import *
from scipy.stats import norm
from datetime import datetime
from datetime import date
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats


quandl.ApiConfig.api_key = '4onR_NWsqDYxNhNfKf6y'

def stock_US(ticker, startdate):
        dfz=yf.download(ticker, startdate)
        dfz.drop(['High','Low','Close','Adj Close','Volume'], axis='columns', inplace=True)
        return (dfz)

def stock_EU(ticker, stardate, enddate):
    dfz=quandl.get(ticker,start_date=stardate, end_date=enddate)
    #A mettre dans la fonction de saisie
    df.rename(columns={'Open': 'Close' }, inplace=True)
    dfz.drop(['High', 'Low','Last','Turnover','Volume'], axis='columns', inplace=True)
    return (dfz)

def stock_metal(ticker, stardate):
    dfz=quandl.get(ticker,start_date=stardate)
    dfz.rename(columns={'USD AM ': 'price PALL' }, inplace=True)
    dfz.drop(['EUR AM', 'GBP AM', 'USD PM','EUR PM','GBP PM'], axis='columns', inplace=True)
    return (dfz)

def stock_crypto(ticker, stardate):
    dfz=quandl.get(ticker,start_date=stardate)
    #A mettre dans la fonction de saisie
    #df.rename(columns={'Open': name }, inplace=True)
    dfz.drop(['High', 'Low', 'Volume'], axis='columns', inplace=True)
    return (dfz)

def stock(ticker, startdate):
        dfz=yf.download(ticker, startdate)
        dfz.drop(['High','Low','Open','Adj Close','Volume'], axis='columns', inplace=True)
        return (dfz)
    
def Brownian(seed, N):
    
    np.random.seed(seed)                         
    dt = 1./N                                    
    b = np.random.normal(0., 1., int(N))*np.sqrt(dt)  
    W = np.cumsum(b)                             
    return W, b


def daily_return(adj_close):
    returns = []
    for i in range(0, len(adj_close)-1):
        today = adj_close[i+1]
        yesterday = adj_close[i]
        daily_return = (today - yesterday)/yesterday
        returns.append(daily_return)
    return returns

#########################Phase1##################################################################""
def GBM(So, drift, sigma, W, T, N):    
    t = np.linspace(0.,1.,N+1)
    S = []
    S.append(So)
    for i in range(1,int(N+1)):
        drift1 = (drift*3 - 0.5 * sigma**2) * t[i]
        diffusion = sigma * W[i-1]
        S_temp = So*np.exp(drift1 + diffusion)
        S.append(S_temp)
    return S, t
  
###CHOC#######################################################################################################################
def CHOC(T,soln,choc): 
    soln[T]=soln[T]-soln[T]*choc 
    T=T
    return soln, T
###Phase2########################################################################################################################

def GBM2(So, drift, sigma, W, T, N):    
    t = np.linspace(0.,1.,N+1)
    S = []
    S.append(So)
    for i in range(1,int(N+1)):
        drift1 = (drift*0.3 - 0.5 * sigma**2) * t[i]
        diffusion = sigma * W[i-1]*2
        S_temp = So*np.exp(drift1 + diffusion)
        S.append(S_temp)
    return S, t

def phase2(So2, drift, sigma, W, T, N):
    
    W = Brownian(seed, N)[0]
    time2=np.linspace(b+1, N+b, N+1)
    soln3, t = GBM2(So2, drift, sig, W, T, N)  
    
    return soln3
##################################################################################Phase 3


def GBM3(So, drift, sigma, W, T, N):    
    t = np.linspace(0.,1.,N+1)
    S = []
    S.append(So)
    for i in range(1,int(N+1)):
        drift1 = (drift*2 - 0.5 * sigma**2) * t[i]
        diffusion = sigma * W[i-1]
        S_temp = So*np.exp(drift1 + diffusion)
        S.append(S_temp)
    return S, t

def phase3(So3, drift, sigma, W, T, N):           
    W = Brownian(seed, N)[0]
    time2=np.linspace(b+1, N+b, N+1)
    soln4, t = GBM3(So3, drift, sig, W, T, N)  
  
    return soln4

###########################################################################################""
def converti(df):
    s=pd.DataFrame(df)
    today = date.today()
    d1 = today.strftime("%d/%m/%Y")
    date_rng = pd.date_range(start='31/01/2020', end='01/02/2022', freq='D')
    s1 = pd.DataFrame(date_rng, columns=['date'])
    s2=s1.join(s)
    s2=s2.set_index('date')
    s2=s2.dropna()
    #s2['date'] = pd.to_datetime(s2['date'])
    s2.index = pd.to_datetime(s2.index)
    s2.rename(columns={0: 'Close' }, inplace=True)
    return s2

########################################################################################
def Prixactif(actif,S):
    df= yf.download('^FCHI',start = "2015-01-01")
    df.drop(['High','Low','Open','Adj Close','Volume'], axis='columns', inplace=True)
    Dp=pd.concat([df,actif], axis=1)
    Dp=Dp.dropna()
    Dp.columns = ['CAC','Actif']
    dr=Dp.pct_change(1)
    dl=dr.dropna(axis=0)
    X=dl['CAC']
    y=dl['Actif']
    slope, intercercept, r_value,p_vakue,std_err = stats.linregress(X,y)
    returnsoln = daily_return(S)   
    rt=slope*np.array(returnsoln)
    ap=[]
    ap=[actif.iloc[-1]*rt[0]+actif.iloc[-1]]
    for i in range(0, len(rt)-1):
        ap.append(ap[i]*rt[i]+ap[i])
    Pa=pd.DataFrame(ap)
    Pa=Pa.to_numpy()
    s2=converti(Pa)
    
    return s2


# In[6]:




####################MAIN########################
######################Initialisation#########################
tickers = ['^FCHI']
multpl_stocks = yf.download(tickers,
start = "2015-01-01",end="2020-02-01")
df=multpl_stocks
df.drop(['High','Low','Open','Adj Close','Volume'], axis='columns', inplace=True)
adj_close = df['Close']
b=df.Close.count()
time = np.linspace(1, b, 1)
returns = daily_return(adj_close)
drift = np.mean(returns)*252.           
sig1 = np.std(returns)
sig =np.mean(sig1)*np.sqrt(252.)
###############################################Phase1##########
seed = 2
N = 40
So = adj_close[b-1]           
W = Brownian(seed, N)[0]
T = 40
time2=np.linspace(b+1, N+b, N+1)
soln, t = GBM(So, drift, sig, W, T, N)  
##############################################################################Choc#################""
soln, T = CHOC(31, soln, 0.2)
##############################################################Phase2#######################################
So2=soln[T]
seed = 15
N = 20
T = 20
W = Brownian(seed, N)[0]
soln3 = phase2(So2,drift,sig,W,T,N)
###############################Phase3###################
So3=soln3[N]
seed = 24
N = 500
T = 500
W = Brownian(seed, N)[0]
soln4 = phase3(So3,drift,sig,W,T,N)
solnfi=soln+soln3+soln4
############################Final#####################################
Sfinal=converti(solnfi)
#########################affichage cac40###############
fig = go.Figure()
fig.add_trace(go.Scatter(x=df.reset_index().Date, y=df['Close'], name='CAC 40'))
fig.add_trace(go.Scatter(x=Sfinal.reset_index().date, y=Sfinal['Close'],name='CAC 40 simulation'))
fig.update_layout(title='Stress Test du CAC40', xaxis_title='Temps en Jour'
                 , yaxis_title='Points')
fig.show()

#############Actif avec affichage ################
actif=stock_EU("EURONEXT/DG","2015-01-02","2020-02-01")
actif2=stock_EU("EURONEXT/BN", "2015-01-02","2020-02-01")
actif3=stock_EU("EURONEXT/FP", "2015-01-02","2020-02-01")
actif4=stock_EU("EURONEXT/GLE", "2015-01-02","2020-02-01")
Pa1=Prixactif(actif,solnfi)
Pa2=Prixactif(actif2,solnfi)
Pa3=Prixactif(actif3,solnfi)
Pa4=Prixactif(actif4,solnfi)
fig = go.Figure()
fig.add_trace(go.Scatter(x=actif.reset_index().Date, y=actif['Open'], name="Danone"))
fig.add_trace(go.Scatter(x=Pa1.reset_index().date, y=Pa1['Close']))
fig.add_trace(go.Scatter(x=actif2.reset_index().Date, y=actif2['Open']))
fig.add_trace(go.Scatter(x=Pa2.reset_index().date, y=Pa2['Close']))
fig.add_trace(go.Scatter(x=actif3.reset_index().Date, y=actif3['Open']))
fig.add_trace(go.Scatter(x=Pa3.reset_index().date, y=Pa3['Close']))
fig.add_trace(go.Scatter(x=actif4.reset_index().Date, y=actif4['Open']))
fig.add_trace(go.Scatter(x=Pa4.reset_index().date, y=Pa4['Close']))
fig.show()


# In[ ]:




