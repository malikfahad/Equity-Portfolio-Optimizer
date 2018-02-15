
import QSTK.qstkutil.qsdateutil as du
import QSTK.qstkutil.tsutil as tsu
import QSTK.qstkutil.DataAccess as da
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
import pandas as pd



def simulate(dt_start, dt_end, ls_symbols, ls_allocation):
    dt_timeofday = dt.timedelta(hours=16)
    ldt_timestamps = du.getNYSEdays(dt_start, dt_end, dt_timeofday)
    c_dataobj = da.DataAccess('Yahoo')
    ls_key = ['close']
    ldf_data = c_dataobj.get_data(ldt_timestamps, ls_symbols, ls_key)
    d_data = dict(zip(ls_key, ldf_data))
    #Portfolio Value
    NUM_TRADING_DAYS = (len(ldt_timestamps))

    temp = d_data['close'].values.copy()
    d_normal = temp / temp[0,:]
    alloc = np.array(ls_allocation).reshape(4,1)
    portVal = np.dot(d_normal, alloc)    
    # Caluclate the daily returns
    dailyVal = portVal.copy()
    tsu.returnize0(dailyVal)
    # Calculate statistics
    daily_ret = np.mean(dailyVal)
    vol = np.std(dailyVal)
    sharpe = np.sqrt(NUM_TRADING_DAYS) * daily_ret / vol
    cum_ret = portVal[portVal.shape[0] -1][0]
    return vol, daily_ret, sharpe, cum_ret

def print_simulate( dt_start, dt_end, ls_symbols, ls_allocation ):
        vol, daily_ret, sharpe, cum_ret  = simulate( dt_start, dt_end, ls_symbols, ls_allocation )
        print "Start Date: ", dt_start
        print "End Date: ", dt_end
        print "Symbols: ", ls_symbols
        print "Optimal Allocations: ", ls_allocation
        print "Sharpe Ratio: ", sharpe
        print "Volatility (stdev): ", vol
        print "Average Daily Return: ", daily_ret
        print "Cumulative Return: ", cum_ret
        
        
    
def optimal_allocation_4( dt_start, dt_end, ls_symbols ):

        max_sharpe = -1
        max_alloc = [0.0, 0.0, 0.0, 0.0]
        for i in range(0,11):
                for j in range(0,11-i):
                        for k in range(0,11-i-j):
                                for l in range (0,11-i-j-k):
                                        if (i + j + l + k) == 10:
                                                alloc = [float(i)/10, float(j)/10, float(k)/10, float(l)/10]
                                                vol, daily_ret, sharpe, cum_ret = simulate( dt_start, dt_end, ls_symbols, alloc )
                                                if sharpe > max_sharpe:
                                                        max_sharpe = sharpe
                                                        max_alloc = alloc

        return max_alloc    
    


dt_start = dt.datetime(2010,1,1)
dt_end = dt.datetime(2011,12,31)
ls_symbols = ['GOOG', 'GLD', 'AAPL', 'XOM']
ls_allocation = [0.4, 0.4, 0.0, 0.2]
max_alloc = optimal_allocation_4( dt_start, dt_end, ls_symbols )
print max_alloc
print_simulate( dt_start, dt_end, ls_symbols, max_alloc )

