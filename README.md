# MC_OptionsPricing
Monte Carlo simulation used to price path dependent Asian and Lookback options


This file is used to simulate the price of certain path-dependent, exotic options using Monte Carlo simulation. 
The underlying asset price evolution is simulated using the Euler-Maruyama scheme. 
Parameters can be adjusted to take account for differences in underlying volatility, risk-free rate, TTE, S0, K, and of course the desired number of path simulations. 
Option prices are calculated to account for differences in contract structure, e.g. if the average price path for an Asian option is calculated discretely or continuously, using geometric or arithmetic averaging, etc. 
Hypothetical price surfaces are generated to visualize the change in value of a given option with respect to changes in volatility and time to expiry.
