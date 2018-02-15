
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
    
def plot():
    ls_alloc=optimal_allocation_4( dt_start, dt_end, ls_symbols )
    dt_timeofday = dt.timedelta(hours=16)
    ldt_timestamps = du.getNYSEdays(dt_start, dt_end, dt_timeofday)
    c_dataobj = da.DataAccess('Yahoo')
    ls_key = ['close']
    ldf_data = c_dataobj.get_data(ldt_timestamps, ls_symbols, ls_key)
    d_data = dict(zip(ls_key, ldf_data))
    df_rets = d_data['close'].copy()
    df_rets = df_rets.fillna(method='ffill')
    df_rets = df_rets.fillna(method='bfill')
    
    na_rets = df_rets.values

    tsu.returnize0(na_rets)
    na_portrets = np.sum(na_rets * ls_alloc, axis=1)
    na_port_total = np.cumprod(na_portrets + 1)
    na_component_total = np.cumprod(na_rets + 1, axis=0)
    plt.clf()
    fig = plt.figure()
    fig.add_subplot(111)
    plt.plot(ldt_timestamps, na_component_total, alpha=0.4)
    plt.plot(ldt_timestamps, na_port_total)
    ls_names = ls_symbols
    ls_names.append('Portfolio')
    plt.legend(ls_names)
    plt.ylabel('Cumulative Returns')
    plt.xlabel('Date')
    fig.autofmt_xdate(rotation=45)    







    
dt_start = dt.datetime(2011,1,1)
dt_end = dt.datetime(2011,12,31)
ls_symbols = ['GOOG', 'GLD', 'AAPL', 'XOM']
max_alloc = optimal_allocation_4( dt_start, dt_end, ls_symbols )
#ls_allocation = max_alloc
print_simulate( dt_start, dt_end, ls_symbols, max_alloc )
plot()

#https://github.com/alexcpsec/coursera-compinvesting1-hw/commit/df4c500e14d7ffa62ff64849d490eb77b77a64b6


