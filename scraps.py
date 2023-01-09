import pandas
from sklearn.cluster import OPTICS
from sklearn.decomposition import PCA
import statsmodels.tsa.stattools as stats
from hurst import compute_Hc

import numpy as np
import pandas as pd
import os
import sys

import SimpleLSTM


def main():
    np.set_printoptions(threshold=sys.maxsize)
    '''
    coins = pd.DataFrame()
    minDates = []
    for filename in os.scandir('./data/Historical/Binance'):
        if filename.is_file():

            if str(filename).split('_')[2][0] == 'd':
                frame = pd.read_csv(filename,header=1)
                #print(frame.columns)
                frame = frame[['Name','date','open']]
                frame['date'] = pd.to_datetime(frame['date'])
                #print(frame)
                if coins.empty:
                    coins = frame
                    minDates.append(min(frame['date']))
                else:
                    minDates.append(min(frame['date']))
                    coins = pd.concat([coins,frame],ignore_index=True)
            #coins = pd.merge(coins,frame,on='Date',how='inner')

            #coins.append(frame)
    #print(coins)
    coins = coins[coins['date'] >= max(minDates)]
    coins['date_delta'] = (coins['date'] - coins['date'].min())  / np.timedelta64(1,'D')
    maxD = max(coins['date_delta'])
    #print(maxD)
    for coin in pd.unique(coins['Name']):
        #print(coin)
        #print(coins[coins['Name'] == coin]['date_delta'])
        if max(coins[coins['Name'] == coin]['date_delta']) < maxD:
            coins.drop(coins[coins.Name == coin].index, inplace = True)
    '''

    coins = pd.read_csv('./data/Historical/Kaggle/all_stocks5yr.csv')
    coins['date'] = pd.to_datetime(coins['date'])
    for name in coins['Name'].unique():
        if len(coins[coins['Name'] == name]) < 1259:
            coins.drop(coins[coins.Name == name].index, inplace=True)
    coins['date_delta'] = (coins['date'] - coins['date'].min()) / np.timedelta64(1, 'D')
    # pairs = findPairs(coins)
    # print(pairs)
    pairs = [('PH', 'PKG'), ('ADI', 'LH'), ('CRM', 'RHT'), ('ED', 'MAS'), ('BBY', 'JBHT'), ('GLW', 'USB'),
             ('SYY', 'TSN'), ('D', 'HCA'), ('HRL', 'NOV'), ('ADP', 'MMC'), ('AET', 'EA'), ('DISH', 'SLG'),
             ('EXPD', 'MAS'), ('ECL', 'TMK'), ('CDNS', 'EXPD'), ('BDX', 'CTAS'), ('MRK', 'SRE'), ('CL', 'UDR'),
             ('CL', 'MAS'), ('AEE', 'PEP'), ('BRK.B', 'MAR'), ('MAS', 'SYY'), ('CA', 'CL'), ('CB', 'GPN'),
             ('BBT', 'FITB'), ('K', 'T'), ('D', 'SCHW'), ('EXC', 'UNM'), ('HSY', 'WY'), ('ADP', 'IT'), ('BXP', 'GPC'),
             ('FIS', 'IFF'), ('CMA', 'MTB'), ('HCP', 'RHT'), ('HSY', 'PRU'), ('CB', 'TXN'), ('ADM', 'SLG'),
             ('ALL', 'TEL'), ('BF.B', 'VLO'), ('CL', 'INCY'), ('HSY', 'LKQ'), ('ECL', 'LUV'), ('MNST', 'PEG'),
             ('HON', 'TXN'), ('BAC', 'BBT'), ('HSY', 'USB'), ('FFIV', 'UAL'), ('ACN', 'AVY'), ('NBL', 'OXY'),
             ('EA', 'MHK'), ('SWK', 'TXN'), ('MMC', 'PEP'), ('EMN', 'MU'), ('GWW', 'OXY'), ('HSY', 'PBCT'),
             ('LH', 'UNH'), ('SEE', 'WBA'), ('HSY', 'MGM'), ('CL', 'MGM'), ('ADP', 'BBY'), ('CL', 'INTC'), ('MU', 'PX'),
             ('PEP', 'RSG'), ('CHRW', 'CSCO'), ('GGP', 'KR'), ('ATVI', 'AVY'), ('IT', 'LUV'), ('CTXS', 'GPN'),
             ('FBHS', 'MOS'), ('CA', 'ZION'), ('COST', 'GPN'), ('HSY', 'IP'), ('AMT', 'ATVI'), ('CL', 'GLW'),
             ('NDAQ', 'PEP'), ('AIG', 'SBUX'), ('MOS', 'XRAY'), ('MAC', 'SPG'), ('KO', 'PEG'), ('EQR', 'JEC'),
             ('FBHS', 'PEP'), ('AOS', 'PEP'), ('CCI', 'PAYX'), ('D', 'UDR'), ('CBOE', 'MSI'), ('OXY', 'SBUX'),
             ('D', 'MAS'), ('CCI', 'MAS'), ('ALXN', 'GILD'), ('EQR', 'HBI'), ('BAX', 'CVX'), ('CRM', 'VRSN'),
             ('CA', 'PHM'), ('ADP', 'SCHW'), ('CBS', 'HSY'), ('CCI', 'SYY'), ('COF', 'LNC'), ('NTRS', 'ZTS'),
             ('CCI', 'STI'), ('ADP', 'ETFC'), ('ADP', 'ATVI'), ('AEP', 'MAS'), ('BBT', 'MS'), ('PG', 'SYMC'),
             ('COST', 'CRM'), ('EMN', 'PX'), ('COST', 'HUM'), ('ATVI', 'LH'), ('ADBE', 'BLK'), ('GGP', 'PDCO'),
             ('HSIC', 'KMB'), ('D', 'TJX'), ('AMT', 'APH'), ('ESS', 'SWKS'), ('MAS', 'PLD'), ('HSY', 'KEY'),
             ('PEP', 'WEC'), ('TRV', 'ZTS'), ('INTC', 'UPS'), ('HSY', 'PLD'), ('A', 'DOV'), ('CLX', 'NDAQ'),
             ('D', 'GLW'), ('LLY', 'SEE'), ('MMC', 'NEE'), ('IPG', 'XRAY'), ('ETFC', 'NTRS'), ('AIG', 'ALK'),
             ('VZ', 'WU'), ('EXPD', 'HBAN'), ('MON', 'WY'), ('L', 'UAA'), ('ADP', 'SNPS'), ('ACN', 'CME'),
             ('ADP', 'CCI'), ('CA', 'HSY'), ('ETN', 'PX'), ('D', 'MRK'), ('UPS', 'XYL'), ('MTB', 'RJF'), ('MU', 'TIF'),
             ('ADP', 'PKI'), ('MGM', 'MSI'), ('ADP', 'APH'), ('CL', 'LNT'), ('CL', 'DRE'), ('CCI', 'HIG'),
             ('GPC', 'LNT'), ('CCI', 'TSS'), ('ARE', 'LH'), ('CTXS', 'SYK'), ('FAST', 'NWL'), ('HIG', 'RRC'),
             ('AOS', 'CCI'), ('ADP', 'ITW'), ('ETN', 'UTX'), ('ECL', 'FIS'), ('AMGN', 'CNC'), ('MAR', 'TEL'),
             ('AIZ', 'CTXS'), ('BSX', 'CMCSA'), ('ALK', 'AVB'), ('D', 'LEG'), ('AIV', 'XRAY'), ('GRMN', 'PX'),
             ('AEP', 'CL'), ('BAX', 'KSU'), ('CL', 'XEL'), ('IDXX', 'MSI'), ('IFF', 'MNST'), ('D', 'WFC'),
             ('DTE', 'RSG'), ('INTC', 'VAR'), ('CHRW', 'INTC'), ('UPS', 'XEL'), ('ADP', 'ALL'), ('KMB', 'XRAY'),
             ('IT', 'ROP'), ('FIS', 'IT'), ('COST', 'DPS'), ('CCI', 'PLD'), ('ADSK', 'PKG'), ('UPS', 'USB'),
             ('ADBE', 'ROP'), ('AIG', 'GT'), ('D', 'UPS'), ('ADSK', 'PLD'), ('ACN', 'TSS'), ('MAS', 'WEC'),
             ('BSX', 'LNT'), ('CCI', 'RSG'), ('AVGO', 'CCI'), ('ANSS', 'BBY'), ('CA', 'RRC'), ('DUK', 'PG'),
             ('CL', 'CMS'), ('ADP', 'AOS'), ('CA', 'KEY'), ('BSX', 'CCI'), ('BLK', 'FDX'), ('PEP', 'XRAY'),
             ('D', 'TXT'), ('CME', 'TXN'), ('CL', 'EIX'), ('UPS', 'WEC'), ('AMGN', 'HCA'), ('SNPS', 'SWK'),
             ('ABT', 'CBG'), ('AMGN', 'SWKS'), ('DTE', 'MMC'), ('GPC', 'PEG'), ('COST', 'LOW'), ('EOG', 'GRMN'),
             ('GGP', 'HBI'), ('BSX', 'EXPD'), ('ADP', 'TMK'), ('KSU', 'NRG'), ('ADP', 'UNM'), ('ROST', 'VRSK'),
             ('INTU', 'SPGI'), ('MRK', 'PCG'), ('CMS', 'PEP'), ('CA', 'PG'), ('BK', 'CCI'), ('TRV', 'WM'),
             ('BSX', 'CMS'), ('NTRS', 'STI'), ('CCI', 'EXPD'), ('BSX', 'WEC'), ('D', 'USB'), ('SCHW', 'STI'),
             ('CB', 'WM'), ('CCL', 'ETFC'), ('COST', 'NDAQ'), ('HSY', 'UNM'), ('PEP', 'ZTS'), ('AMAT', 'IR'),
             ('JNJ', 'WM'), ('AEE', 'MAS'), ('D', 'MGM'), ('D', 'INTC'), ('LUV', 'TRV'), ('DUK', 'GLW'), ('CB', 'FB'),
             ('HON', 'V'), ('KSU', 'SJM'), ('CCI', 'CMS'), ('HSY', 'MA'), ('AAL', 'FFIV'), ('BF.B', 'LEN'),
             ('BSX', 'PEP'), ('AMT', 'SYK'), ('ITW', 'NEE'), ('PG', 'WY'), ('CCI', 'PGR'), ('CA', 'D'),
             ('CSCO', 'EXPD'), ('MKC', 'NDAQ'), ('CL', 'GT'), ('CL', 'CMCSA'), ('DUK', 'MRK'), ('CCI', 'DHI'),
             ('AIG', 'FTI'), ('A', 'EMN'), ('JBHT', 'SCHW'), ('AFL', 'SRCL'), ('LLL', 'MMM'), ('BSX', 'FBHS'),
             ('ETR', 'IRM'), ('FBHS', 'XRAY'), ('ADP', 'WM'), ('HSY', 'NUE'), ('CA', 'PGR'), ('IPG', 'KMB'),
             ('ADP', 'AMAT'), ('PX', 'WMT'), ('ADP', 'V'), ('CTXS', 'MAS'), ('SPGI', 'TMO'), ('EBAY', 'KSU'),
             ('CB', 'DTE'), ('D', 'HIG'), ('ETFC', 'TEL'), ('CCI', 'FBHS'), ('CTXS', 'SYY'), ('ROST', 'TRV'),
             ('ADI', 'DRI'), ('HSY', 'SYMC'), ('LOW', 'MNST'), ('ADP', 'ZTS'), ('ADP', 'STI'), ('FBHS', 'MAS'),
             ('BLK', 'NVDA'), ('AEP', 'PEP'), ('IT', 'PEP'), ('PFE', 'XEL'), ('LEN', 'RMD'), ('BLL', 'D'),
             ('CMCSA', 'EXPD'), ('UDR', 'XRAY'), ('CCI', 'SCHW'), ('CELG', 'OXY'), ('LOW', 'TRV'), ('KO', 'MAR'),
             ('HSY', 'RF'), ('ACN', 'GPN'), ('CNP', 'L'), ('CCI', 'ETFC'), ('D', 'PKG'), ('SPGI', 'VAR'), ('D', 'GT'),
             ('D', 'LLY'), ('ANSS', 'PGR'), ('GPC', 'ZBH'), ('CCI', 'CMCSA'), ('LLL', 'TMO'), ('UPS', 'ZTS'),
             ('MDLZ', 'XRAY'), ('CB', 'FIS'), ('HCP', 'PAYX'), ('LUV', 'PEP'), ('AWK', 'CINF'), ('ADP', 'TEL'),
             ('GT', 'MDLZ'), ('GPC', 'INTC'), ('BDX', 'SPGI'), ('NDAQ', 'TRV'), ('CCI', 'CDNS'), ('PAYX', 'VRSK'),
             ('AIG', 'HOLX'), ('EW', 'TDG'), ('AMGN', 'COST'), ('MAS', 'VRSK'), ('JBHT', 'ZTS'), ('KO', 'SO'),
             ('FBHS', 'VRSK'), ('AOS', 'VRSK'), ('ADP', 'BSX'), ('LH', 'SNPS'), ('D', 'LNT'), ('MON', 'PCAR'),
             ('ETR', 'WY'), ('BLL', 'CL'), ('CTXS', 'DLR'), ('DUK', 'WY'), ('HCP', 'LKQ'), ('CL', 'PEG'), ('LH', 'V'),
             ('PEP', 'WM'), ('JNJ', 'UPS'), ('MMM', 'SPGI'), ('LH', 'RMD'), ('VRSK', 'ZTS'), ('HCP', 'IT'),
             ('CCI', 'PKI'), ('HON', 'MSFT'), ('D', 'XEL'), ('ADBE', 'JPM'), ('APH', 'LH'), ('BBT', 'KEY'),
             ('ADP', 'XLNX'), ('PFE', 'UNM'), ('CA', 'MAS'), ('CRM', 'TSS'), ('COG', 'GPS'), ('CCI', 'ZTS'),
             ('PFE', 'SCHW'), ('DRE', 'XEL'), ('EOG', 'MU'), ('AET', 'MHK'), ('COST', 'MNST'), ('EMN', 'FLIR'),
             ('BSX', 'VRSK'), ('MSI', 'XYL'), ('AIG', 'OXY'), ('CA', 'RF'), ('NEE', 'TXN'), ('ADP', 'RSG'),
             ('ADP', 'XYL'), ('ADP', 'MSFT'), ('CA', 'ZTS'), ('CB', 'MMC'), ('COG', 'MRO'), ('ETN', 'MON'),
             ('JNJ', 'MMC'), ('ITW', 'SYK'), ('HSY', 'ZION'), ('FIS', 'LH'), ('MON', 'PWR'), ('AAPL', 'SBAC'),
             ('CCI', 'PEG'), ('DUK', 'IRM'), ('ATVI', 'CCI'), ('AOS', 'TRV'), ('ALL', 'RMD'), ('PCAR', 'UTX'),
             ('CL', 'EXC'), ('ESS', 'EXPE')]

    predictions = SimpleLSTM.run(coins[['open', 'Name']], pairs, 0.001, 1000)

    thresholds = calc_thresholds(predictions)
    print(thresholds)

    '''
    Predicted Change = delta_t+1 = (S_t+1 - S_t)/(S_t) * 100
    S(t): the spread of pair at time t
    Where S_t+1 is the predicted and S_t is the observed value at time t
    if delta_t+1 < alpha_l 
        open short position
    if delta_t+1 > alpha_s
        open long position

    Spread Percentage Change
        x_t = (S_t - S_t-1)/S_t-1 * 100

    create distribution based on Spread Percentage Change as f(x)
    Select top decile and quantile from f(x) > 0 and f(x) < 0

    test decile and quantile performance in validation set and use better performing



    '''


def calc_thresholds(predictions):
    """

    :param predictions:
    :return:
    """

    thresholds = {}
    for ts in predictions:
        # print(ts)
        Dp_t = []
        Dn_t = []
        for i in range(1, len(ts[1])):
            D_i = (ts[1][i] - ts[1][i - 1]) / ts[1][i - 1]
            # print(D_i)
            if D_i >= 0:
                Dp_t.append(D_i)
            else:
                Dn_t.append(D_i)

        Dp_t = pd.DataFrame(Dp_t, columns=['D_t'])
        Dn_t = pd.DataFrame(Dn_t, columns=['D_t'])

        # print(ts[0])
        # print('neg:',Dp_t)
        # print('pos:',Dn_t)
        thresholds_pos = Dp_t.quantile([0.1, 0.25, 0.75, 0.9])
        thresholds_neg = Dn_t.quantile([0.1, 0.25, 0.75, 0.9])
        thresholds[ts[0]] = [thresholds_pos, thresholds_neg]

    return thresholds


def findPairs(coins):
    '''

    :param coins:
    :return:
    '''

    '''
    X = pd.DataFrame(0,index=np.arange(coins.shape[0]),columns=coins['Name'].unique())

    for name in X.columns:
        X[name] = coins[coins['Name'] == name]['open']


    X.insert(0, 'date_delta', np.tile(np.arange(1259),len(X.columns)))
    X = X.fillna(0).copy()
    '''

    '''
    pca = PCA(n_components=5)
    X_pca = pca.fit_transform(X)
    print(X_pca)
    print(pca)
    clusters = OPTICS(xi=.2).fit_predict(X_pca)
    with open('clusters','w') as f:
        f.write(np.array2string(clusters))
    '''

    '''
        print(clusters_values)
    print(np.shape(clusters_values))
    clusters_values = np.reshape(clusters_values,(1,len(clusters_values)))
    clusters = pd.DataFrame(data=clusters_values,columns=['clusterLabel'])
    print(clusters.head())
    #print(clusters)

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
    '''

    '''
        coins_noOutliers = X.copy()
    coins_noOutliers.drop(X[X.clusterLabel == -1].index, inplace = True)
    coins_noOutliers.drop('date_delta',axis=1,inplace=True)
    cointegrated = []
    clusteredCoins = set()
    for i in range(coins_noOutliers['clusterLabel'].max()+1):
        cluster = coins_noOutliers[coins_noOutliers['clusterLabel'] == i]
        names = cluster.columns[cluster.nonzero()]
        if len(names) > 1:
            clusteredCoins.add(tuple(names))
        #print(names)
    print('Finished Clustering with',len(clusteredCoins),'clusters found')

    print(clusteredCoins)
    '''

    names = pd.unique(coins['Name'])
    cointegrated = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):

            co1 = coins[coins['Name'] == names[i]]
            co2 = coins[coins['Name'] == names[j]]
            # print(co1)
            # print(co2)
            # print()
            coint = stats.coint(co1['open'], co2['open'])

            if coint[1] < 0.05:
                print((names[i], names[j]))
                cointegrated.append((names[i], names[j]))

    co_pairs = set(cointegrated)
    coH_pairs = []
    print(co_pairs)
    for pair in co_pairs:
        # print(coins[coins['Name'] == pair[0]]['open'].to_numpy())
        # print(coins[coins['Name'] == pair[1]]['open'].to_numpy())
        spread = np.subtract(coins[coins['Name'] == pair[0]]['open'].to_numpy(),
                             coins[coins['Name'] == pair[1]]['open'].to_numpy())
        spread = np.absolute(spread)
        spread[spread == 0] = np.finfo(np.float64).eps
        # print(spread)
        H, c, data = compute_Hc(spread, kind='price', simplified=True)
        if H < 0.5:
            coH_pairs.append(pair)

    return coH_pairs


if __name__ == "__main__":
    main()





