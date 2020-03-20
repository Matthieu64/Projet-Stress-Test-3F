# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 15:12:52 2020

@author: merin
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from IPython import get_ipython
import seaborn as sns
import yfinance as yf
import quandl
import math
from scipy import stats
import datetime
from scipy.stats import norm

quandl.ApiConfig.api_key = '4onR_NWsqDYxNhNfKf6y'


#=====================================================================
#Fonctions
#=====================================================================

#US
def stock_US(ticker, startdate,enddate=None):
        dfz=yf.download(ticker, startdate,enddate)
        dfz.drop(['High','Low','Close','Adj Close','Volume'], axis='columns', inplace=True)
        return (dfz)

#EU
def stock_EU(ticker, stardate):
    dfz=quandl.get(ticker,start_date=stardate)
    #A mettre dans la fonction de saisie
    #df.rename(columns={'Open': name }, inplace=True)
    dfz.drop(['High', 'Low','Last','Turnover','Volume'], axis='columns', inplace=True)
    return (dfz)
#Metal
def stock_metal(ticker, stardate):
    dfz=quandl.get(ticker,start_date=stardate)
    dfz.rename(columns={'USD AM ': 'price PALL' }, inplace=True)
    dfz.drop(['EUR AM', 'GBP AM', 'USD PM','EUR PM','GBP PM'], axis='columns', inplace=True)
    return (dfz)
#CRYPTO
def stock_crypto(ticker, stardate):
    dfz=quandl.get(ticker,start_date=stardate)
    #A mettre dans la fonction de saisie
    #df.rename(columns={'Open': name }, inplace=True)
    dfz.drop(['High', 'Low', 'Volume'], axis='columns', inplace=True)
    return (dfz)





def SimulationMT(df,portf_WT,portf_value,nb_days,nb_sim,pct_stress,law="normal"):
    
    pct_cov=100
    nb_stock=portf_WT.shape[0]
    df_daily_returns = df['Open'].pct_change()
    df_daily_returns=df_daily_returns.dropna()

    cov = np.cov(np.transpose(df_daily_returns))
    portf_stdev = np.sqrt(portf_WT.T.dot(cov).dot(portf_WT))*pct_cov
    miu = np.mean(df_daily_returns, axis=0)
    Miu = np.full((nb_days,nb_stock),miu)
    Miu = np.transpose(Miu)
    portf_returns = np.full((nb_days,nb_sim),0.)
    np.random.seed(100)
    
    for i in range(0,nb_sim):
        if law=="weibull":
            Z = np.random.weibull(1,size=nb_stock*nb_days)-(1+pct_stress)
        else:
            Z = np.random.normal(-pct_stress,portf_stdev,size=nb_stock*nb_days)
        

        Z = Z.reshape((nb_stock,nb_days))
        L = np.linalg.cholesky(cov)
        daily_returns = Miu + np.inner(L,np.transpose(Z))
        portf_Returns = np.cumprod(np.inner(portf_WT,np.transpose(daily_returns)) + 1)
        portf_returns[:,i] = portf_Returns*portf_value
    
    final_value=sum(portf_returns[nb_days-1,])/nb_sim
    best_case = np.quantile(portf_returns[nb_days-1,:],0.99)
    worst_case = np.quantile(portf_returns[nb_days-1,:],0.01)

    return  portf_returns,final_value,portf_stdev,worst_case,best_case




def modele_BS(S,K,r,T,sigma,option='put'):
   
    d1= (np.log(S/K)+(r+ ((sigma**2)/2))*T)/(sigma*np.sqrt(T))
    
    d2=d1-(sigma*np.sqrt(T))
    
    if option == 'call':
        result=(S*norm.cdf(d1,0,1)-K*np.exp(-r*T)*norm.cdf(d2,0,1))
    if option == 'put':
        result=(K*np.exp(-r*T)*norm.cdf(-d2,0,1)-S*norm.cdf(-d1,0,1))
    
    return result


def put(S,K,r,T,sigma,portefeuille,strike_prec):
    
    #exercer ou pas
    portefeuille=portefeuille+max(strike_prec-S,0)
    
    #pricing option et achat
    
    prime=modele_BS(S,K,r,T,sigma,'put')
    portefeuille=portefeuille-prime
    
    return portefeuille, strike_prec



#Pricing d'option
def option_stresstest(df,sigma=0.05):
    r=0.005
    T=1
    security_put=1.05
    nb_option=50
    counter=T
    df_option=np.zeros((df.shape[0],df.shape[1]))
    
    
    for i in range(df.shape[1]): #Pour la simulation i
        
        variation=0
        K=df[0,i]*security_put
        
        for j in range(df.shape[0]): #Pour la simulation i au temps j
            
            if j==counter:
                tmp=0
                prime=modele_BS(df[j,i],K,r,T,sigma,'put')
                tmp=(prime*nb_option)*(-1)
                tmp+=(max(K-df[j,i],0)*nb_option)
                variation+=tmp
                K=df[j,i]*security_put
            
            df_option[j,i]=df[j,i]+variation
                
            
    final_value=sum(df_option[df.shape[0]-1,])/df.shape[0]
    best_case = np.quantile(df_option[df.shape[0]-1,:],0.99)
    worst_case = np.quantile(df_option[df.shape[0]-1,:],0.01)  
    
    
    return df_option,final_value,worst_case,best_case



#--------------------------------------------------------------------


if __name__ == "__main__":
    
    #Ptf
    data=pd.DataFrame()
    tickers=["LRLCY","LVMHF","SNY","TOT","HESAY","AIQUY","PPRUY"]
    data=stock_US(tickers,"2020-01-02","2020-03-02")
    portf_stock=np.array([1,1,1,1,1,1,1])
    nb_simulation=1000
    nb_day=15
    pct_stress=0.3
    law="normal"
    portf_WT=[]
    portf_price=[]

    #Calcul de la valeur du portefeuille à partir du dernier cours et du nombre d'action
    for i in range(len(tickers)):
        value=data["Open"][tickers[i]][-1]*portf_stock[i]
        portf_price.append(value)    
    portf_value=sum(portf_price)    
    for i in range(len(tickers)):
        portf_WT.append(portf_price[i]/portf_value)    
    portf_WT=np.array(portf_WT)
    
    
    
    #Simulations
    portf_simulation,portefolio_average,portf_sigma,portefolio_worst,portefolio_best=SimulationMT(data,portf_WT,portf_value,nb_day,nb_simulation,pct_stress,law)
    print(portf_simulation)
    print(portf_simulation.shape)
    #Actual results
    current_value=stock_US(tickers,"2020-03-17","2020-03-17")
    current_value=current_value.sum(1)
    
    data["sum_portf"] = data.sum(axis=1)
    data_reshape =np.tile(np.array([data["sum_portf"]]).transpose(), (1, nb_simulation))
    print(data_reshape.shape,portf_simulation.shape)
    simulation=np.vstack((data_reshape, portf_simulation))
    
 
    #get_ipython().run_line_magic('matplotlib', 'qt')
    
    #Simulation 15 jours
    plt.figure()
    plt.plot(portf_simulation)
    plt.axhline(y=portefolio_average,color='b',linestyle='--',label='Average result')
    plt.axhline(y=current_value[0],color='y',linestyle='--',label='Current value')
    plt.axhline(y=portefolio_worst,color='r',linestyle='--',label='1% Worst result')
    plt.axhline(y=portefolio_best,color='g',linestyle='--',label='1% Best result')
    plt.ylabel('Valeur du portefeuille')
    plt.xlabel('Jours')
    plt.legend(loc='upper right')
    plt.title('Simulation ('+law+')')
    plt.show()
        
    #Simulation complete
    plt.figure()
    plt.plot(simulation)
    plt.axhline(y=current_value[0],color='y',linestyle='--',label='Real portefolio value')
    plt.axhline(y=portefolio_average,color='b',linestyle='--',label='Average Simulation result')
    plt.axhline(y=portefolio_worst,color='r',linestyle='--',label='1% Worst result')
    plt.axhline(y=portefolio_best,color='g',linestyle='--',label='1% Best result')
    plt.ylabel('Valeur du portefeuille')
    plt.xlabel('Jours')
    plt.legend(loc='upper right')
    plt.title('Simulation ('+law+')')
    plt.show()
    
    
    #Simulation du portefeuille optionné
    portf_option,option_average,option_worst,option_best=option_stresstest(portf_simulation,portf_sigma/100)
    final_value_option=sum(portf_option[nb_day-1,])/nb_simulation
    #Plot
    plt.figure()
    plt.plot(portf_option)
    plt.axhline(y=final_value_option,color='b',linestyle='--',label='Average result')
    plt.axhline(y=current_value[0],color='y',linestyle='--',label='Current value')
    plt.axhline(y=option_worst,color='r',linestyle='--',label='1% Worst result')
    plt.axhline(y=option_best,color='g',linestyle='--',label='1% Best result')
    plt.ylabel('Valeur du portefeuille')
    plt.xlabel('Jours')
    plt.legend(loc='upper right')
    plt.title('Simulation avec options ('+law+')')
    plt.show()
    


    
    
    
