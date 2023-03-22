#!/usr/bin/env python3
import argparse
from sklearn.cluster import OPTICS
from sklearn.decomposition import PCA
import statsmodels.tsa.stattools as stats
from hurst import compute_Hc
import finnhub
import numpy as np
import pandas as pd
import os
import sys
from ast import literal_eval
from datetime import datetime
import SimpleLSTM
import Portfolio as pf


def cmdInterpret(cmdLine):
    '''
    Command line interpeter

    '''
    findPairs = cmdLine.args.findPairs
    calcThresholds = cmdLine.args.calcThresholds
    labels = cmdLine.args.labels
    cluster = cmdLine.args.cluster
    filename = cmdLine.args.filename
    delim = cmdLine.args.delimiter
    learningRate = float(cmdLine.args.learningRate)
    epochs = int(cmdLine.args.epochs)
    test = cmdLine.args.test
    initInvest = int(cmdLine.args.initInvest)
    
    if os.path.isdir(filename):
        data = pd.DataFrame()
        for file in os.listdir(filename):
            print(file)
    else:     
        data = pd.read_csv(filename,sep=delim)
    
    
    data.rename(columns={labels[2]:'name'},inplace=True)
    labels[2] = 'name'
    data[labels[1]] = pd.to_datetime(data[labels[1]])
    
    if findPairs:
        pairs = pd.DataFrame(find_pairs(data,labels))
        pairs.to_csv(os.getcwd()+'/Pairs_'+filename.split('/')[-1])     
        pairs.to_csv('Pairs_'+filename.split('/')[-1])

    if calcThresholds:
        pairs = pd.read_csv('Pairs_'+filename.split('/')[-1])
        print(data[[labels[0],'name']].head())
        predictions = SimpleLSTM.run(data[[labels[0],'name']],pairs,learningRate,epochs)
        predictionsDF = pd.DataFrame(predictions)
        predictionsDF.to_csv('Predictions_'+filename.split('/')[-1])
        thresholds = pd.DataFrame(calc_thresholds(predictions))
        thresholds.to_csv('Thresholds_'+filename.split('/')[-1],index=False)
        
    if test:
        pairs = pd.read_csv('Pairs_'+filename.split('/')[-1])
        pairs = [tuple(x[1:]) for x in pairs.to_numpy()]
        thresholds = pd.read_csv('Thresholds_'+filename.split('/')[-1])
        print(thresholds.head())
        for col in thresholds.columns:
            for i in range(thresholds[col].size):
                thresholds[col][i] = literal_eval(thresholds[col][i])

        
        port = pf.Portfolio(data,pairs,thresholds,initInvest)
        ledger = port.test_thresholds(0.7)
        
        for day,transactions in ledger.items():
            print('Day ', str(day),' : ',str(transactions))
            
    '''
    if trade:
        pairs = pd.read_csv('Pairs_'+filename.split('/')[-1]).to_numpy()
        pairs = [x[1:] for x in pairs]
        data = getPrices(pairs)
        pd.DataFrame(data).to_csv('Prices_'+str(datetime.now())+'_'+filename.split('/')[-1])
        
        trade(data,pairs,thresholds,initInvest,0.7)
    '''

def calc_thresholds(predictions):
    """
    calculates trading thresholds based on the LSTM model's predcitions 

    :param predictions: list of time series predictions output by SimpleLSTM.run()
    :return: dictionary of trading thresholds with key value pairs, name of security: list of thresholds
    """

    thresholds = {}
    for ts in predictions:
        Dp_t = []
        Dn_t = []
        for i in range(1,len(ts[1])):
            D_i = (ts[1][i]-ts[1][i-1])/ts[1][i-1]
            if D_i >= 0:
                Dp_t.append(D_i)
            else:
                Dn_t.append(D_i)

        Dp_t = pd.DataFrame(Dp_t, columns=['Dp_t'])
        Dn_t = pd.DataFrame(Dn_t, columns=['Dn_t'])
        

        thresholds_pos = Dp_t.quantile([0.1,0.25,0.75,0.9])
        thresholds_neg = Dn_t.quantile([0.1,0.25,0.75,0.9])

        thresholds[ts[0]] = [[x[0] for x in thresholds_pos.to_numpy()], [x[0] for x in thresholds_neg.to_numpy()]]
    
    
    return thresholds


def cluster(secs,labels): # currently has no use other than data visualization 
    '''
    runs OPTICS on the dataset and writes output to individual dataset files 
    naming convention: arbitrageData[name of securty].csv

    :param secs: dataset of securites
    :param lables: lables of open,date and name columns
    :return: N/a

    '''

    X = pd.DataFrame(0,index=np.arange(secs.shape[0]),columns=secs['Name'].unique())

    for name in X.columns:
        X[name] = secs[secs['Name'] == name]['open']


    X.insert(0, 'date_delta', np.tile(np.arange(1259),len(X.columns)))
    X = X.fillna(0).copy()

    # in dimensions above 15 density based clustering algorithms tend to be much less effective
    # in this case PCA is used to reduce dimensionaltiy to 15 before running OPTICS
    if X.shape[1] < 15:
        clusters = OPTICS(xi=.2).fit_predict(X)
    else: 
        pca = PCA(n_components=15)

        X_pca = pca.fit_transform(X)
        clusters = OPTICS(xi=.2).fit_predict(X_pca)


    with open('clusters','w') as f:
        f.write(np.array2string(clusters))



    clusters_values = np.reshape(clusters_values,(1,len(clusters_values)))
    clusters = pd.DataFrame(data=clusters_values,columns=['clusterLabel'])

    X['clusterLabel'] = clusters

    names = list(X.columns)
    print(names)
    print(X.shape)
    print(X.head())
    for name in names:
        if name == 'clusterLabel' or name == 'date_delta':
            continue
        ind = X[name].to_numpy().nonzero()
        Y = X[[name,'clusterLabel']].iloc[ind]

        with open('arbitrageData'+name+'.csv', 'w') as f:
            f.write(Y.to_csv())



def find_pairs(secs,labels):
    '''

    :param secs:
    :return:
    '''


    names = pd.unique(secs[labels[2]])
    cointegrated = []
    for i in range(len(names)):
        for j in range(i+1,len(names)):

            co1 = secs[secs[labels[2]] == names[i]]
            co2 = secs[secs[labels[2]] == names[j]]
            co1 = co1[co1[lables[1]] == co2[lables[1]]]
            co2 = co2[co1[lables[1]] == co2[lables[1]]]
            print(co1)
            print(co2)
            #print()
            coint = stats.coint(co1[labels[0]], co2[labels[0]])

            if coint[1] < 0.05:
                print((names[i],names[j]))
                cointegrated.append((names[i],names[j]))


    co_pairs = set(cointegrated)
    coH_pairs = []
    print(co_pairs)
    for pair in co_pairs:
        
        spread = np.subtract(secs[secs[labels[2]] == pair[0]][labels[0]].to_numpy(), secs[secs[labels[2]] == pair[1]][labels[0]].to_numpy())
        spread = np.absolute(spread)
        spread[spread==0] = np.finfo(np.float64).eps

        H, c, data = compute_Hc(spread, kind='price', simplified=True)
        if H < 0.5:
            coH_pairs.append(pair)

    return coH_pairs





class CommandLine():
    '''
    Command Line program
    attributes:


    all arguments received from the commandline using .add_argument will be
    avalable within the .args attribute of ob-1ect instantiated from CommandLine.
    For example, if myCommandLine is an ob-1ect of the class, and requiredbool was
    set as an option using add_argument, then myCommandLine.args.requiredbool will
    name that option.

    '''

    def __init__(self, inOpts=None):
        '''
        Implement a parser to interpret the command line argv string using argparse.

        args:
            findPairs: optional, runs find_pairs() 
            calcThresholds: optional, runs calc_thresholds()
            labels: specifies labels of columns in the dataset, needs names of open,date,name in that order. 
                Where open is the opening price, date is the date and name is the name of the security
                Defualts to ['Open','Date','Name']
            cluster: runs OPTICS on the dataset to cluster the securites. Currently not used in algorithm.
            filename: REQURIED, flag specifies the filename of the dataset
            delim: optional, specifies delimiter of the dataset, ie tsv,csv,etc.. Defaults to csv.
            learingRate: optional, specify learing rate of LSTM models. Defualts to 0.01.
            epochs: optional, speficy number of epochs used to train LSTM models. Defualts to 1000.
            test: optional, runs a test enviromment that trades using thresholds created from calc_thresholds()
            initInvest: optional, specifies initial investment amount used in test(). Defualts to 10,000$.
        '''

        import argparse
        self.parser = argparse.ArgumentParser(description='Program prolog - Finds orfs in a Fasta file',
                                              epilog='Program epilog - ',
                                              add_help=True,  # default is True
                                              prefix_chars='-',
                                              usage='%(prog)s [options] -option1[default] <input >output'
                                              )
        self.parser.add_argument('-f', '--filename',action='store',required=True,help='Argument to input the filename of your dataset, assumes csv format by default')
        self.parser.add_argument('-fp', '--findPairs', action='store_true',help='runs findPairs in arbitrage')
        self.parser.add_argument('-ct', '--calcThresholds', action='store_true', help='runs calcThresholds in arbitrage')
        self.parser.add_argument('-l', '--labels', action='store', default=['Open','Date','Name'], nargs='*',help='Enter the titles of the open, date and name columns in the dataset in that order, default= open date name')
        self.parser.add_argument('-c', '--cluster', action='store_true', help='clusters data using OPTICS before finding pairs, very slow if working with a large dateset')
        self.parser.add_argument('-de','--delimiter',action='store',default=',',help='Enter delimiter for your dataset, defaults to comma (csv)')
        self.parser.add_argument('-lr', '--learningRate',action='store',default='0.01',nargs='?',help='Set learning rate for the Neural Networks, default=0.01')
        self.parser.add_argument('-e','--epochs',action='store',default='1000',nargs='?',help='Set number of epochs for training Nerual Networks, default=1000')
        self.parser.add_argument('-t', '--test', action='store_true',help='runs testThresholds in arbitrage')
        self.parser.add_argument('-i','--initInvest',action='store',default='10000',nargs='?',help='Set initial trading capital in US dollars default=10000')

        if inOpts is None:
            self.args = self.parser.parse_args()
        else:
            self.args = self.parser.parse_args(inOpts)



def main(inCL=None):
    if inCL is None:
        myCommandLine = CommandLine()
    else:
        myCommandLine = CommandLine(inCL)

    cmdInterpret(myCommandLine)

if __name__ == "__main__":
    main()  
    
