# -*- coding: utf-8 -*-

#=====================================================================
#Librairies
#=====================================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
import quandl
import math
from scipy import stats
import datetime
from IPython import get_ipython

quandl.ApiConfig.api_key = '4onR_NWsqDYxNhNfKf6y'


#=====================================================================
#Fonctions
#=====================================================================

#US
def stock_US(ticker, startdate):
        dfz=yf.download(ticker, startdate)
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


#Fonction qui constitue le portfolio
def portfolio():
    date="2015-01-01" #Date de debut du portefeuille
    value_list=['US','EU','MP','CP']#Types d'actions
    nb_inputs = int(input("Combien d'actions souhaitez vous ?\n"))
    
    df=pd.DataFrame() #Le dataframe que l'on retournera à la fin
    i=0
    while i<nb_inputs: #Repeté le nombre de fois que l'on veut d'actions
        
        action_type=input('De type : US/EU/MP/CP?\n') #Sur quel marché est référencée l'action
        if action_type not in value_list:
            print("ERROR VALUE - TRY AGAIN")
            continue
        
        ticker=input('Ticker ?\n') #Le nom de l'action (son ticker)
        
        if action_type=='US':
            df_tmp=stock_US(ticker, date)
            
        elif action_type=='EU':
            df_tmp=stock_EU(ticker, date)

        elif action_type=='MP':
            df_tmp=stock_metal(ticker, date)

        elif action_type=='CP':
            df_tmp=stock_crypto(ticker, date)

            
        df_tmp.rename(columns={'Open': ticker }, inplace=True)
        df= pd.concat([df, df_tmp], axis=1, sort=False)
        df=df.dropna()
            
        i+=1
    
    return df
            


def asset_repartition(df):
    assets={}
    i=0
    columns=df.columns
    
    answer=input('Répartition du portefeuille en pourcentage ou valeur ? % ou V\n')
    while answer not in ['V','%']:
        print('ERREUR - REPONSE V OU %')
        answer=input('Répartition du portefeuille en pourcentage ou valeur ? % ou V\n')
        
        
    #Si l'utilisateur veut remplir son portefeuille en pourcentage
    if answer=='%':
        percentage=100
        while i < len(columns):
            print('\nPourcentage à distribuer restant: {}%\nNombre d\'assets restant: {}\n'.format(percentage,(len(columns)-i)))
            value=float(input('Valeur de \'{}\' en pourcentage ?\n'.format(columns[i])))
            if value>100 or value<0:
                print('ERREUR VALEUR INCORRECTE - REESSAYEZ')
                continue
            percentage-=value
            assets[columns[i]]=value/100
            i+=1            
        portfolio_value=int(input('Valeur Globale du portefeuille ?\n'))
        
        return assets,portfolio_value
        
    
    
    #Si l'utilisateur veut remplir son portefeuille en valeur    
    elif answer=='V':
        list_values=[]
        portfolio_value=0
        while i < len(columns):
            
            value=float(input('Valeur de \'{}\' en € ?\n'.format(columns[i])))
            if value<=0:
                print('ERREUR VALEUR INCORRECTE - REESSAYEZ')
                continue
            portfolio_value+=value
            list_values.append(value)
            i+=1
            
        for i in range(len(columns)):
            assets[columns[i]]=round((list_values[i]/portfolio_value),3)
            
        return assets,portfolio_value
            
        
        
    
            
            
def var_historique(data,portfolio_value,confidence_interval,nb_days,**assets):
    keys=list(assets.keys())
    #Verification s'il y a des colonnes qui sont en trop
    for asset in data.columns:
        if asset not in keys:
            data=data.drop(asset,axis=1)
            
    
    data=data.pct_change()
    for col in data.columns:
        data[col]=data[col]*portfolio_value*assets[col]
    
        
    print(data.head())    
    data["Sum_variations"]=data.sum(axis=1)
    data=data["Sum_variations"]
    var=data.quantile(q=1-(confidence_interval/100))
    return round(float(var),2)  
    
    
    
    
def var_parametric(data,portfolio_value,confidence_interval,nb_days,**assets):
    
    keys=list(assets.keys())
    weight_matrix=np.asmatrix([value for key,value in sorted(assets.items())])
    for asset in data.columns:
        if asset not in keys:
            data=data.drop(asset,axis=1)
            
    time_period=data.shape[0] #Nombre de valeurs que l'on a
    data = data.reindex(sorted(data.columns), axis=1)
    Var=np.dot(weight_matrix,np.dot(data.corr(),weight_matrix.T))*(portfolio_value)*(-confidence_interval)*np.sqrt(time_period/250)
    
    return round(float(Var),2)
                 
def var_parametric2(data,portfolio_value,confidence_interval,nb_days,**assets):
    
    keys=list(assets.keys())
    for asset in data.columns:
        if asset not in keys:
            data=data.drop(asset,axis=1)
    nb_stocks=len(assets)
    variance1=0
    for key in assets:
        variance1+=(df[key].var(axis=0))*(assets[key])**2
    covariance_matrix=data.cov()
    variance2=0
    for i in range(0,nb_stocks):
        for j in range(0,nb_stocks):
            variance2+=covariance_matrix.iloc[i,j]*assets[keys[i]]*assets[keys[j]]
                    
    variance=variance1+variance2
    var=math.sqrt(variance)*math.sqrt(data.shape[0])*1.65 #95%
    
    return var
        
    
                


    
def var_montecarlo(ticker):
    
    end = datetime.datetime.now()
    start = end - datetime.timedelta(365)
    stock = quandl.get('EOD/'+ticker, start_date=start, end_date=end)
    rets_1 = (stock['Close']/stock['Close'].shift(1))-1
    mean = np.mean(rets_1)
    std = np.std(rets_1)
    price = stock.iloc[-1]['Close']
    np.random.seed(42)
    n_sims = 1000000
    sim_returns = np.random.normal(mean, std, n_sims)
    montecarlo_VAR = price*np.percentile(sim_returns, 1)
    return montecarlo_VAR    



def VarMT(data,confidence_interval):
    df=pd.DataFrame()
    tickers=["AAPL","TSLA","MSFT"]
    df=stock_US(tickers,"2015-01-01")
    
    df_daily_returns = df['Open'].pct_change()
    df_daily_returns=df_daily_returns.dropna()
    vol = df_daily_returns.std()*math.sqrt(252)
    cagr =df.iloc[-1]['Open'] /df.iloc[1]['Open'] -1 
    print ("Annual Volatility =",str(round(vol,4)*100)+"%")
    mc_rep = 100
    train_days = 30
    portf_WT = np.array([1/3, 1/3, 1/3]) # à modifier cela signifie le pourcentage de nos actions, acteullement 2 acts de 50%
    cov = np.cov(np.transpose(df_daily_returns))
    miu = np.mean(df_daily_returns, axis=0)
    Miu = np.full((train_days,3),miu)
    Miu = np.transpose(Miu)
    print("________________",miu)
    portf_returns_30_m = np.full((train_days,mc_rep),0.)
    np.random.seed(100)
    for i in range(0,mc_rep):
        Z = np.random.normal(size=3*train_days)
        Z = Z.reshape((3,train_days))
        L = np.linalg.cholesky(cov)
        daily_returns = Miu + np.inner(L,np.transpose(Z))
        portf_Returns_30 = np.cumprod(np.inner(portf_WT,np.transpose(daily_returns)) + 1)
        portf_returns_30_m[:,i] = portf_Returns_30
    Avg_CI = np.quantile(portf_returns_30_m[29,:]-1,0.95)
    Vara95MT=300*Avg_CI/np.sqrt(30)
    return  Vara95MT,portf_returns_30_m


  
from pandas_datareader import data as pdr
import datetime as dt
from scipy.stats import norm

def Var_parametric3(data,portfolio_value,confidence_interval,nb_days,**assets):    
    
    #Recuperation valeurs portefeuille
    tickers = list(assets.keys())
    weights=np.array([value for key,value in sorted(assets.items())])
    initial_investment = portfolio_value
    #Necessaire pour utiliser l'API avec Pandas
    yf.pdr_override() 
    data = pdr.get_data_yahoo(tickers, start="2015-01-01", end=dt.date.today())['Open']
    #Matrice de retour
    returns = data.pct_change()
    #Matrice de variance covariance
    cov_matrix = returns.cov()
    #Moyenne et moyenne pondérée du portefeuille
    avg_rets = returns.mean()
    port_mean = avg_rets.dot(weights)
    #Ecart type du portefeuille pondéré
    port_stdev = np.sqrt(weights.T.dot(cov_matrix).dot(weights))
    
    #Moyenne et ecart type
    mean_investment = (1+port_mean) * initial_investment
    stdev_investment = initial_investment * port_stdev
    #Fonction gaussienne de la perte
    cutoff = norm.ppf(confidence_interval, mean_investment, stdev_investment) 
    #Investissement -perte
    var_1d = initial_investment - cutoff

    return(np.round(var_1d, 2))
    
#=====================================================================
#Main
#=====================================================================  

if __name__ == "__main__":
    
    #Création du portfolio souhaité
    #df=portfolio() #Exemple de ticker US :TSLA,MSFT,AAPL
    #df.to_csv("stocks_data.csv")
    #Plot resultats
    df=pd.read_csv('stocks_data.csv',index_col='Date')
    df.plot(figsize=(20,10), linewidth=5, fontsize=20)      
    plt.xlabel('Time', fontsize=20)
    
    #Repartition de la valeur du portefeuille
    assets,portfolio_value=asset_repartition(df)
    confidence_interval=int(input('Quel interval de confiance souhaitez-vous entre (1 et 99) ?\n'))
    print(df.mean(axis=0))
    var=var_parametric2(df,portfolio_value,confidence_interval,df.shape[0],**assets)

    
    Vara95MT,portf_returns_30_m=VarMT(df,95)
    #get_ipython().run_line_magic('matplotlib', 'qt')
    plt.figure()
    plt.plot(portf_returns_30_m)
    plt.ylabel('Portefeuille retour')
    plt.xlabel('Jours')
    plt.title('Simulation')
    plt.show()
    
    Avg_portf_returns = np.mean(portf_returns_30_m[29,:]-1)
    SD_portf_returns = np.std(portf_returns_30_m[29,:]-1)
    Median_portf_returns = np.median(portf_returns_30_m[29,:]-1)
    print(Avg_portf_returns)
    print(SD_portf_returns)
    print(Median_portf_returns)
    
    Avg_CI = np.quantile(portf_returns_30_m[29,:]-1,0.95)
    Vara95MT=300*Avg_CI/np.sqrt(30)
    
    var_param=Var_parametric3(df,portfolio_value,0.05,df.shape[0],**assets)
    var_histo=-var_historique(df,portfolio_value,confidence_interval,df.shape[0],**assets)
    print('\n\n\n\nValeur du portefeuille = ',portfolio_value)
    print('Var historique :',var_histo)
    print('Var parametrique :',var_param)
    print('Var monte carlo :',round(Vara95MT,2))
        
    #////////////////////////////////////////////////////////
    
    #var_montecarlo('AAPL')
    