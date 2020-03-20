# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 14:46:09 2020

@author: merin
"""

import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.stats import norm

beta=0

def mvt_bro(n):
    
    beta=100
    mvt_bro=[beta]
    
    for i in range(1,n):
        beta= np.random.normal(0,1,1)+beta
        #fonction_option(volatité,SJ,maturité,tx sans risque)
        mvt_bro.append(beta)

    return mvt_bro


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
    
    

    
def stress_test():
  
    #Parametres
    cours=100
    nb_action=1
    portefeuille=cours*nb_action
    n=1000
    r=0.02
    T=5
    sigma=0.045
    k=T
    
    #Premiere iteration
    strike_prec=cours
    cours_prec=cours
    prime=modele_BS(cours,cours,r,T,sigma,'put')
    portefeuille=portefeuille-prime
    simulation=[[cours,portefeuille]]
    
    for i in range(1,n):
        cours= np.random.normal(0,1,1)+cours
        variation=(cours-cours_prec)
        portefeuille=portefeuille+variation*nb_action
        if i%k==0:
            portefeuille,strike_prec=put(cours,cours,r,T,sigma,portefeuille,strike_prec)
            strike_prec=cours
            
        cours_prec=cours
        simulation.append([cours,portefeuille])
        
        
        
    return simulation
        


data=stress_test()
plt.plot(data)
plt.gca().legend(('cours','portefeuille'))
plt.show()    