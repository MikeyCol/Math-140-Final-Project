import pandas as pd
import numpy as np
import datetime

class Investment():
    '''
    container for invidiaul investments
    initIn: float, initial investment amount
    day: int, records the day the investment was made as number of days passed since the first 
    '''
    def __init__(self,initIn,day):
        self.initIn = initIn
        self.day = day
    
    
class Portfolio():
    '''
    Contains methods for testing the thresholds on the dataset
    also keeps track of all investments made durring testing
    '''
    
    def __init__(self,data,pairs,thresholds,initIn):
        '''
        '''
        self.data = data # dataset
        self.pairs = pairs # list of pairs produced by arbitrage.find_pairs()
        self.thresholds = thresholds # dictionary of thresholds for each pair
        self.freeCash = initIn # cash available for trading
        self.ledger = {} # dictinoary used to keep track of every resovled position. 
        #key, value pairs are key = day | value = list of tuples describing each trade in format (name of security, long or short, open price, profit)
        

        self.secs = set()
        for pair in pairs:
            self.secs.add(pair[0])
            self.secs.add(pair[1])
        self.investmentsLong = {} # dictionary representing the amount held in long positions per security
        self.investmentsShort = {} # dictionary representing the amount held in short positions per security
        '''
        for sec in secs:
            self.investmentsShort[sec] = Investment(0, datetime.date.min)
            self.investmentsLong[sec] = Investment(0, datetime.date.min)
        '''
    def sell(self,sec,open,day,isLong):
        '''
        sells position and updates freeCash and Ledger

        :param sec: name of the security to sell
        :param open: current open price
        :param day: current day
        :param isLong: boolean determined long or short position
        '''
        if isLong:
            self.freeCash += open
            net = open - self.investmentsLong[sec].initIn
            if day in self.ledger.keys():
                self.ledger[day].append((sec,'sell long',open,net))
            else:
                self.ledger[day] = [(sec,'sell long',open,net)]
            del self.investmentsLong[sec]
            
        else:
            net = self.investmentsShort[sec].initIn - open
            self.freeCash += net + self.investmentsShort[sec].initIn
            if day in self.ledger.keys():
                self.ledger[day].append((sec,'sell short',open,net))
            else:
                self.ledger[day] = [(sec,'sell short',open,net)]
            del self.investmentsShort[sec]
            
    def test_thresholds(self,split):
        '''
        
        this method tests the efficacy of the thresholds by 
        '''
        self.ledger = {}
        percentInvest = 0.05

        for sec in self.secs:
            
            opens = self.data[self.data['name'] == sec]['open']
            opens = opens[int(split*opens.shape[0]):].to_numpy()
            
            opensStd = np.std(opens)
            opensThresholds = self.thresholds[sec]
            print(opensStd)
            

            for i in range(1,opens.shape[0]): # iterates through each day
                open = opens[i]
                #open2 = sec2[i]
                prevOpen = opens[i-1]
                #prevOpen2 = sec2[i-1]
                # tests exit points for every investment
                
                # Short sell conditions:
                # - open reverts to invest price or higher (accept loss)
                # - open drops equal to or below 1 std deviation below initial short price
                # - open gains price above one of the positive thresholds???
                
                if sec in self.investmentsShort.keys():
                    initIn = self.investmentsShort[sec].initIn
                    if initIn <= open:
                        self.sell(sec,open,i,False)
                    elif open < initIn - opensStd:
                        self.sell(sec,open,i,False)
                            
                            
                # Long sell conditions:
                # - open reverts to invest price or lower (accept loss)
                # - open gains at least a std deviation above invest price
                # - open loses price below one of the negative thresholds???
                
                if sec in self.investmentsLong.keys():
                    initIn = self.investmentsLong[sec].initIn
                    if initIn >= open:
                        self.sell(sec,open,i,True)
                    elif open > initIn + opensStd:
                        self.sell(sec,open,i,True)
                    
                    
               
                print('Day ',  str(i))
                print()
                print(sec,' open: ', str(open))
                #print('sec 2 open: ', str(open2))
                
                print()
                print('sec 1 short adjusted threshold: ' + str(opensThresholds[1][1]*prevOpen + prevOpen))
                if sec not in self.investmentsShort.keys() and open < opensThresholds[1][1]*prevOpen + prevOpen: # tests for opening short 
                    print('opened short on '+str(open)+' on day ' + str(i))
                    invest = self.freeCash*percentInvest
                    self.freeCash -= invest
                    self.investmentsShort[sec] = Investment(invest,i)
                    
                print('sec 1 long adjusted threshold: ' + str(opensThresholds[0][2]*prevOpen + prevOpen)) 
                if sec not in self.investmentsLong.keys() and open > opensThresholds[0][2]*prevOpen + prevOpen: # tests for opening long 
                    print('opened long on '+str(open)+' on day ' + str(i))
                    invest = self.freeCash*percentInvest
                    self.freeCash -= invest
                    self.investmentsLong[sec] = Investment(invest,i)
                
                print()
            print('Free Cash: ', str(self.freeCash))
        print('remaining free cash: ', self.freeCash)
        print('long investments')
        for key,value in self.investmentsLong.items():
            print(key,value)
        print('short investments')
        for key,vlaue in self.investmentsShort.items():
            print(key,value)
            
        return self.ledger
                
        
    
    def get_prices(secs):
        '''
        :param secs: list of strings representing security names 
        :return: dict of str:float key:value pairs representing current security prices
        '''
        
        for pair in pairs:
            for sec in pair:
                if len(secs) >= 60: #finnhub daily limit of 60 prices, could get another key for another 60
                    break
                print(sec)
                secs.add(sec)

        
        finnhub_client = finnhub.Client(api_key="cc0v7kqad3ifk6taktqg")
        
        data = {}
        
        for sec in secs:
            data[sec] = finnhub_client.quote(sec)
            print(sec)
            print(str(data[sec]))
        
        return data

    