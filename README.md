# Machine Learing Pairs Trading 

## Introduction
This program implements pairs trading using LSTM networks to inform when to open or close short or long positoins in the market.

The general structure for the code and the math/ general scrture of the algorithm comes from 

	A Machine Learning based Pairs Trading Investment Strategy (SpringerBriefs in Applied Sciences and Technology) 1st ed. 2021

For a more in depth understanding of pairs trading the math behind this algorithm give it a read, but here's a TLDR.

Before we get into how this works it's important to note that, obiviously, this doesn't work. 
There is no magic money making algorithm, even in the test environment that this program runs on it doesn't consitently profit.
Despite this, it is still an intresting apporoach to implementing ML into predictive stock/secutriy trading. 

Pairs trading is an approach to automated securities trading that involves identifying a pair of securites that have a consistent mean reverting spread, 
and then shorting the the security that is increaing in value and going long on the security that is dropping in value. If the spread of the securites reverts back to 
the same historical mean, then there would be a profit on both the short and long positions as the security that dropped in value would go back up, and the security that increaseed in value would go back down. 
For simplicity I defined spread as simply the difference in value between securites.  

The most common example of a "pair" is corn and ethanol securites. The value of corn and ethanol are intrisictly linked since roughly 40% of all ethanol in the US is produced from corn. 
So, Looking at corn and ethanol futures/etfs they tend to have slight variations in price but consitently tend to revert to a similar mean. 

![plot](./corn_ethanol_spread.png)

In order to profit off of this trend, if the security increases above or decreaces below the mean you open a short or long position on the security under the assumption that it will revert back to a historical mean, making some profit if it does revert. 
In the graph above its clear that if ethanol was shorted at its peak there would have been a massive profit. The idea behind pairs trading is to find pairs of secirties that can help predict and profit off of these kind of fluxuations. 


## Algorithm Overview

The alogorithm can be broken down into three steps  

1. Idenfity pairs using cointegration and the Hurst exponent.
2. Calculate trading thresholds using indiviualized LSTM Nerual Networks.
3. Test the thresholds in a vitural environment.  

### 1. Identify pairs  
	
In order to identify a profiable pair the securities must be cointegrated and the spread of the securites must have a hurst exponent less then 0.5.  

A pair of time series, x_t and y_t, are cointegrated if they are non-stationary and have order of integration d=1 and they can form a linear combonation for some value β and u_t where u_t is stationary.
	
$x_t - βy_t = u_t$
		
This ensures that the mean and variance of the spread are constant over time, which implies that the spread is mean reverting over a long enough peroid of time. 

The Hurst exponent describes whether the time series consitently reverts to a mean or if trends in some coniststent direction.
Formally defined on [the wikipedia article for the Hurst Exponent](https://en.wikipedia.org/wiki/Hurst_exponent) as 
	
$E[R(n)/S(n)] = Cn^{h}$ as $n \rightarrow \infty$

where  

1. R(n) is the range of the first n cumulative deviations from the mean
2. S(n) is the series (sum) of the first n standard deviations
3. $E[x]$ is the expected value
4. n is the time span of the observation (number of data points in a time series)
5. C is a constant.

If h < 0.5 then the series is considered mean-revertint/ anit-persistent and can qualify as a pair.

### 2. Calculate Thresholds 

Thresholds determine the change in value of a security that triggers opeing a position on that security.  
Two thresholds are calcualted for each security, one to detrmine the daily change in value to open a long position and one to open a short position. 

These thresholds are derived from predicitions geneerated from the LSTM networks created by SimpleLSTM.py  
Individual nerual netowrks are trained for each securiy in the pair, then predictions are generated over the testing data using these NNs.

Thresholds are calcualted from these predictions by calulating the delta of each time step using $


Once pairs are indientified, LSTM networks are trained on data from each pair in order to predict the prices of each pair.

Predictions for the desired time peroid, in this case the testing split, are generated and then the change in every time step is calcualted using the following formula.  
$D = V_{i} - V_{i-1}/V_{i-1}$  
Where D is the normalized delta between time steps and $V_{i}$ is the value of a security at time step i.  
Then the deltas are split into poisitive and negative groups, the idea being to create separate distributions for when the value of the security increases and when it decreases.  
The quantile and decile of each distribution are calcualted. These become the thresholds to open a long or short position on a security.



### 3. Test Thresholds

Once the thresholds are calcualted we end up with eight thresholds per secuirty.  
The upper and lower quanitle and decile for both the positive deltas and negative delats from the predcited time series.

These thresholds are then tested over the testing data from the train/test split used to train the NN.  

The trading algorithm works as follows:
1. Check for short sell conditions
	1. if open reverts to invest price or higher (accept loss)
	2. if open drops equal to or below 1 std deviation below initial short price
2. Check for long sell conditions
	1. if open reverts to invest price or lower (accept loss)
	2. open gains at least a std deviation above invest price
3. Check for opening short positions
	1. if open increases past one of the positive thresholds 
4. Check for opeing long positions
	1. if open decrueses past of the negative thresholds
	
	
The quanitle and decile thresholds are tested and the better performing threshold is chosen.

Theoretically at this point if the thresholds profit over a long peroid of time then you could run it on a with real money and hope to profit. 

## File descriptions

aribitrage.py
	- executable
	- run using ./arbitrage.py -f filename.csv -fp -ct -t 
	- flag descriptions
		-f:  Argument to input the filename of your dataset, assumes csv format by default, only required flag
		-fp: finds pairs in dataset
		-ct: calcualtes thresholds of found pairs
		-l: secifies columns labels defualt=['Open','Date','Name'] assumes that order in your input
		-c: runs a OPTICS clustering over the dataset (not used, but intresting for visualization)
		-de: specifies delmiter in dataset
		-lr: specifies learing rate for NN training
		-e: specifiies epochs for NN training
		-t: tests thresholds
		-i: speicifies initial capital for testing thresholds

Portfolio.py
	- called by -t in arbitrage.py
	- contains algorithm for testing thresholds
	
SimpleLSTM.py
	- conains NN framework 
	
	


