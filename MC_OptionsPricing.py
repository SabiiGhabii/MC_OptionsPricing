#!/usr/bin/env python
# coding: utf-8

# ## Outline
# 
# This report is an investigation on the pricing of Asian and lookback options, two types of path-dependent, exotic derivatives contracts. Both types of options lend themselves to pricing via Monte Carlo simulation given that their expression as closed-form solutions presents challenges. 
# 
# This outline will leverage the Euler-Maruyama scheme for simulating the price paths of the underlying. Option values are expressed as the discounted, expected value of the payoff under the risk-neutral density $\mathbb{Q}$.  

# # Euler-Maruyama Scheme
# 
# We simulate an underlying asset whose price evolves according to a geometric, Brownian motion:
# 
# $dS_t = rS_{t}dt + \sigma{S_t}d{W_t}$
# 
# Such that the asset price changes over the time step $dt$, grows by the risk-free rate, $r$, with a constant volatility, $\sigma$, influenced by a normally distributed random variable $dW_t$.
# 
# Applying the Euler-Maruyama scheme and rearranging terms, our asset price simulations will be dictated by the governing equation, such that the change in the asset price over one time step is given by:
# 
# $S_{t+\partial{t}} = S_{t}e^{((r - \frac{1}{2}\sigma^2)\partial{t} + \sigma\sqrt{{\partial}t}{w_t})}$

# In[237]:


import pandas as pd
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import scipy as sp
from scipy import stats


# # Asian Options
# 
# We define our class for Asian option simulation. Four different subcategories of Asian options and their price variations were explored. We take as a default a fixed-strike Asian option, with an arithmetic and continuously sampled average. The variations thereof include an Asian option with a geometric, continuously sampled average, an arithmetic, discretely sampled average (with sampling once every 20 time steps, which would correspond to roughly once a month), and a floating-strike, arithmetic, continuously sampled average. 
# 
# While there are extensively more variations with respect to the different subcategories of Asian options (exponentially weighted averages, higher dimension variations e.g. anteater options), we limit the investigation to these types as the represent the most common variations thereof. 

# In[1]:


class AsianOptions:
    def __init__(self, S0, strike, rate, sigma, dte, nsim, timesteps:int=252) -> float:
    
        self.S0 = S0
        self.K = strike
        self.r = rate
        self.sigma = sigma
        self.T = dte
        self.N = nsim
        self.ts = timesteps
    
    @property
    def randomnumber(self):
        return np.random.standard_normal(self.N)
    
    @property
    def simulatepath(self):
        np.random.seed(31415)
        
        dt = self.T/self.ts
        S = np.zeros((self.ts, self.N))
        S[0] = self.S0
        
        for i in range(0, self.ts-1):
            w = self.randomnumber
            S[i+1] = S[i] * (1+self.r*dt + self.sigma * np.sqrt(dt)*w)
            
        return S
    
    @property
    def arith(self):
        S = self.simulatepath
        A = np.mean(S, axis=0)
        
        arith_call = np.exp(-self.r*self.T) * np.mean(np.maximum(0, S-self.K))
        arith_put = np.exp(-self.r*self.T) * np.mean(np.maximum(0, self.K-S))
        
        return [arith_call, arith_put]
    
    @property
    def geo(self):
        S = self.simulatepath
        A = sp.stats.gmean(S, axis=0)
        
        geo_call = np.exp(-self.r*self.T) * np.mean(np.maximum(0, A-self.K))
        geo_put = np.exp(-self.r*self.T) * np.mean(np.maximum(0, self.K-A))
        
        return [geo_call, geo_put]
    
    @property
    def discrete(self):
        S = self.simulatepath
        A = np.mean(S[::20], axis=0)
        
        disc_call = np.exp(-self.r*self.T) * np.mean(np.maximum(0, A-self.K))
        disc_put = np.exp(-self.r*self.T) * np.mean(np.maximum(0, self.K-A))
        
        return [disc_call, disc_put]
    
    @property
    def floating(self):
        S = self.simulatepath
        A = np.mean(S, axis=0)
        
        float_call = np.exp(-self.r*self.T) * np.mean(np.maximum(0, S[-1]-A))
        float_put = np.exp(-self.r*self.T) * np.mean(np.maximum(0, A-S[-1]))
        
        return [float_call, float_put]

    @property
    def AsianOptionTable(self):
        output = pd.DataFrame(columns=[
            'Arithmetic',
            'Geometric',
            'Discrete',
            'Rate Strike'],
                              index=[
                                  'Call',
                                  'Put'
                              ])
        output['Arithmetic'] = self.arith
        output['Geometric'] = self.geo
        output['Discrete'] = self.discrete
        output['Rate Strike'] = self.floating
        
        return output


# # Sampling and Payoffs
# 
# Our 'fair-value' for each option is determined by it's expected, discounted, risk-neutral payoff, such that each of our simulations' payoffs are discounted by:
# 
# $V(S,t) = e^{-r(T-t)}\mathbb{E}^{\mathbb{Q}}[$Payoff $(S_T)] $
# 
# Payoffs are contingent on whether or not the option's strike or rate is determined by the average, such that: 
# 
# * Average Strike Call <br/>
# $max(S-A,0)$
# * Average Strike Put <br/>
# $max(A-S,0)$
# * Average Rate Call <br/>
# $max(A-E,0)$
# * Average Rate Put <br/>
# $max(E-A,0)$
# 
# For determining the value of $A$, our sampled averages:
# 
# * Continuous, arithmetic <br/>
# $A = \frac{1}{t} \int_{0}^{t}S(\tau)d\tau $
# * Continuous, geometric <br/>
# $A = exp(\frac{1}{t} \int_{0}^{t}logS(\tau)d\tau) $
# * Discrete, arithmetic <br/>
# $A_{i} = \frac{1}{i} \sum_{k=1}^{i}S(t_{k}) $
# * Discrete, geometric <br/>
# $A_{i} = \frac{i-1}{i}A_{i-1}+\frac{1}{i}S(t_{i}) $

# For this exercise, an individual price path was visualized and compared to the various types of sampling between Asian options. The code below shows a single price path which evolves according to our original Brownian motion, a continuously-sampled arithmetic average of that same price path, in addition to a discretely-sampled arithmetic average. 
# 
# Beyond illustration, the diagram shows the apparent differences in payoff expectations when determined by the running-average of the underlying, namely that individual moves or shocks will ultimately be fairly inconsequential with respect to payouts (particularly in the case of discrete sampling, where a one-time shock may have virtually no impact on the running-average depending on the price moves that follow). 

# In[308]:


sim = AsianOptions(100,100,0.05,0.2,1,1)
data = sim.simulatepath


# In[309]:


def AsianPricePath(data, samples):
    
    csum = 0
    cm_avg = []
    divs = int(len(data)/samples)
    sliced = data[::divs]
    
    for s, num in enumerate(sliced, start=1):
        csum += num
        cm_avg.append(csum / s)
        result = pd.DataFrame(np.repeat(cm_avg, divs))
    return result


# In[310]:


plt.figure(figsize=[9,4])
plt.title('MC Simulated Asian Option Price Path')
plt.plot(AsianPricePath(data,252), color='blue', label='Continuous Arithmetic Avg')
plt.plot(AsianPricePath(sim.simulatepath,12), color='red', label='Discrete Arithmetic Avg')
plt.plot(sim.simulatepath, color='green', label = 'Price Path')
plt.xlabel('Time Steps')
plt.xlim(0,252)
plt.ylabel('Asset Price')
ax = plt.subplot(111)
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))


# # Valuation
# 
# Using the aforementioned methods we produce the value estimations for the various types of Asian options. We use a set of default parameters such that:
# $S_{0} = 100$, $E = 100$, $r = 5\%$, $\sigma = 20\%$, $(T-t) = 252$. 

# In[201]:


final = AsianOptions(100,100,0.05,0.2,1,100000)
final.AsianOptionTable


# # Further Analysis
# 
# As supplement, we investigate how the option prices may change with respect to different parameters. Below, we plot how both an arithmetic, continuously sampled Asian call and put change as volatility of the underlying increases and as the contract approaches expiration. This was done by creating a 3D "price surface" whereby values on the Z-coordinate represent the changing price of the options, X-coordinates represent different volatilities, and Y-coordinates represent time to expiration. 

# In[294]:


AsianCall = pd.DataFrame()
AsianPut = pd.DataFrame()

for i in np.arange(0.05, 1.05, 0.05):
    results = []
    for j in np.arange(0.05, 1.05, 0.05):
        result = AsianOptions(100, 100, 0.05, j, i, 10000)
        results.append(result.arith[0])
    AsianCall[f'{int(i*252)}'] = results
    AsianCall.index = np.arange(0.05, 1.05, 0.05)

for i in np.arange(0.05, 1.05, 0.05):
    results = []
    for j in np.arange(0.05, 1.05, 0.05):
        result = AsianOptions(100, 100, 0.05, j, i, 10000)
        results.append(result.arith[1])
    AsianPut[f'{int(i*252)}'] = results
    AsianPut.index = np.arange(0.05, 1.05, 0.05)


# In[298]:


x1,y1 = np.meshgrid(AsianCall.index.astype(float), AsianCall.columns.astype(float))
z1 = AsianCall.values.astype(float)

x2,y2 = np.meshgrid(AsianPut.index.astype(float), AsianPut.columns.astype(float))
z2 = AsianPut.values.astype(float)

fig = plt.figure(figsize=(8,4))
ax1 = fig.add_subplot(121, projection='3d')
callplot = ax1.plot_surface(x1, y1, z1, cmap='viridis')
ax1.set_xlabel('Volatility')
ax1.set_ylabel('Days to Expiration')
ax1.set_zlabel('Option Value')
ax1.set_title('Volatility v. DTE: Asian Call')

ax2 = fig.add_subplot(122, projection='3d')
putplot = ax2.plot_surface(x2, y2, z2, cmap='viridis')
ax2.set_xlabel('Volatility')
ax2.set_ylabel('Days to Expiration')
ax2.set_zlabel('Option Value')
ax2.set_title('Volatility v. DTE: Asian Put')
plt.tight_layout()
plt.show()


# # Lookback Options
# 
# Next, we will explore lookback options using the same methods as for Asian options. We will be using the Euler-Maruyama method to simulate the price evolution of the underlying, and value our options as the expected, discounted payoff under the risk-neutral density $\mathbb{Q}$.
# 
# The three subcategories that will be valued include continuously sampled, floating and fixed strike payoff lookback options, in addition to a fixed strike, discretely sampled lookback option. 

# In[57]:


class LookbackOptions:
    def __init__(self, S0, strike, rate, sigma, dte, nsim, timesteps:int=252) -> float:
    
        self.S0 = S0
        self.K = strike
        self.r = rate
        self.sigma = sigma
        self.T = dte
        self.N = nsim
        self.ts = timesteps
    
    @property
    def randomnumber(self):
        return np.random.standard_normal(self.N)
    
    @property
    def simulatepath(self):
        np.random.seed(31415)
        
        dt = self.T/self.ts
        S = np.zeros((self.ts, self.N))
        S[0] = self.S0
        
        for i in range(0, self.ts-1):
            w = self.randomnumber
            S[i+1] = S[i] * (1+self.r*dt + self.sigma * np.sqrt(dt)*w)
            
        return S
    
    @property
    def lb_fixed(self):
        S = self.simulatepath
        M_max = np.amax(S, axis=0)
        M_min = np.amin(S, axis=0)
        
        lbfix_call = np.exp(-self.r*self.T) * np.mean(np.maximum(0, M_max-self.K))
        lbfix_put = np.exp(-self.r*self.T) * np.mean(np.maximum(0, self.K-M_min))
        
        return [lbfix_call, lbfix_put]
    
    @property
    def lb_float(self):
        S = self.simulatepath
        M_max = np.amax(S, axis=0)
        M_min = np.amin(S, axis=0)
        
        lbfloat_call = np.exp(-self.r*self.T) * np.mean(np.maximum(0, S[-1]-M_min))
        lbfloat_put = np.exp(-self.r*self.T) * np.mean(np.maximum(0, M_max-S[-1]))
        
        return [lbfloat_call, lbfloat_put]
    
    @property
    def disc_lb_fixed(self):
        S = self.simulatepath
        M_max = np.amax(S[::20], axis=0)
        M_min = np.amin(S[::20], axis=0)
        
        disc_lbfix_call = np.exp(-self.r*self.T) * np.mean(np.maximum(0, M_max-self.K))
        disc_lbfix_put = np.exp(-self.r*self.T) * np.mean(np.maximum(0, self.K-M_min))
        
        return [disc_lbfix_call, disc_lbfix_put]
    
    @property
    def LBOptionTable(self):
        output = pd.DataFrame(columns=[
            'Fixed',
            'Floating',
            'Discrete'],
                              index=[
                                  'Call',
                                  'Put'
                              ])
        output['Fixed'] = self.lb_fixed
        output['Floating'] = self.lb_float
        output['Discrete'] = self.disc_lb_fixed
        
        return output


# # Sampling and Payoffs
# 
# In the case of lookback options, we will still use the same method to determine present value by discounting the expected payoff under a risk-neutral density. Lookback options however are unique in that the option price is a function of three variables, $V(S,M,t)$, with the addition of $M$ representing either the realized minimum or maximum subject to the restriction $0 \leq{S} \leq{M}$. 
# 
# * Fixed Strike Call <br/>
# $max(M-E,0)$, where $M$ is the realized maximum.
# * Fixed Strike Put <br/>
# $max(E-M,0)$, where $M$ is the realized minimum.
# * Floating Strike Call <br/>
# $max(M-S,0)$, where $M$ is the realized minimum.
# * Average Rate Put <br/>
# $max(S-M,0)$, where $M$ is the realized maximum.
# 
# Additionally we consider the condition for discrete sampling, whereby the realized maximum or minimum is updated by:
# $M_{i} = max(S(t_{i}), M_{i-1})$. 

# Following in the footsteps of our previous work with Asian options, we may visualize a hypothetical price path and compare it to the path used to determine the payoff of a lookback option.  

# In[311]:


vis = LookbackOptions(100,100,0.05,0.2,1,1)
path = vis.simulatepath


# In[312]:


def LBPricePath(data):
    max_so_far = float('-inf')
    result = []

    for num in data:
        if num > max_so_far:
            max_so_far = num
            result.append(num)
        else:
            result.append(max_so_far)

    return result


# In[315]:


plt.figure(figsize=[9,4])
plt.title('MC Simulated Lookback Option Price Path')
plt.plot(LBPricePath(path), color='blue', label='Lookback Option Path')
plt.plot(path, color='red', label='Price Path')
plt.xlabel('Time Steps')
plt.xlim(0,252)
plt.ylabel('Asset Price')
ax = plt.subplot(111)
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))


# # Valuation
# 
# We continue to valuation for our lookback options using the same parameters as those of our Asian options:
# $S_{0} = 100$, $E = 100$, $r = 5\%$, $\sigma = 20\%$, $(T-t) = 252$. 

# In[316]:


sim2 = LookbackOptions(100,100,0.05,0.2,1,100000)
LB_Options = sim2.LBOptionTable
LB_Options


# # Further Analysis
# 
# We note that from the price surface of the lookback options that the rate at which the value of puts increases tends to slow as volatility peaks, whereas the rate at which the value of lookback calls increases does not seem to curve off at the higher ranges of asset volatility.

# In[321]:


LBCall = pd.DataFrame()
LBPut = pd.DataFrame()

for i in np.arange(0.05, 1.05, 0.05):
    results = []
    for j in np.arange(0.05, 1.05, 0.05):
        result = LookbackOptions(100, 100, 0.05, j, i, 10000)
        results.append(result.lb_fixed[0])
    LBCall[f'{int(i*252)}'] = results
    LBCall.index = np.arange(0.05, 1.05, 0.05)

for i in np.arange(0.05, 1.05, 0.05):
    results = []
    for j in np.arange(0.05, 1.05, 0.05):
        result = LookbackOptions(100, 100, 0.05, j, i, 10000)
        results.append(result.lb_fixed[1])
    LBPut[f'{int(i*252)}'] = results
    LBPut.index = np.arange(0.05, 1.05, 0.05)


# In[322]:


x1,y1 = np.meshgrid(LBCall.index.astype(float), LBCall.columns.astype(float))
z1 = LBCall.values.astype(float)

x2,y2 = np.meshgrid(LBPut.index.astype(float), LBPut.columns.astype(float))
z2 = LBPut.values.astype(float)

fig = plt.figure(figsize=(8,4))
ax1 = fig.add_subplot(121, projection='3d')
callplot = ax1.plot_surface(x1, y1, z1, cmap='viridis')
ax1.set_xlabel('Volatility')
ax1.set_ylabel('Days to Expiration')
ax1.set_zlabel('Option Value')
ax1.set_title('Volatility v. DTE: Lookback Call')

ax2 = fig.add_subplot(122, projection='3d')
putplot = ax2.plot_surface(x2, y2, z2, cmap='viridis')
ax2.set_xlabel('Volatility')
ax2.set_ylabel('Days to Expiration')
ax2.set_zlabel('Option Value')
ax2.set_title('Volatility v. DTE: Lookback Put')
plt.tight_layout()
plt.show()


# # References
# 
# Wilmott, Paul. *Paul Wilmott on Quantitative Finance*. 2nd ed., vol. 2, John Wiley & Sons, 2018. 
