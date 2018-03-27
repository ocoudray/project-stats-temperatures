import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from BuysBallot import BuysBallotModel

path = "Data/ukcp09_gridded-land-obs-daily_timeseries_mean-temperature_000000E_500000N_19600101-20161231.csv"
data = pd.read_csv(path, header=[0,1], index_col=0)
dates_to_remove = [str(y)+"-02-29" for y in range(1960, 2017, 4)]
data = data.drop(dates_to_remove)

def f2(t, i, j, length, subd = 1):
    return np.power(t,i) * (t>=j*length/subd).astype(int)*(t<(j+1)*length/subd).astype(int)


def g(t, i, k = 365):
    return (t%k == i).astype(int) - (t%k == 0).astype(int)

BBM = BuysBallotModel(f2, g, subd=1, dim=2)

def transform(series, weights, shifts):
    l = [weights[k] * series.shift(shifts[k]) for k in range(len(shifts))]
    return sum(l)

weights1 = [1/1095]*1095
shifts1 = list(range(-547, 548))

weights2 = [0]*1461
weights2[0] = 1/9
weights2[365] = 2/9
weights2[730] = 3/9
weights2[1095] = 2/9
weights2[1460] = 1/9
shifts2 = list(range(-365, 366))

def decompose(data, easting, northing, weights_t, shifts_t, weights_s, shifts_s, plot=False):
    if plot:
        plt.figure(figsize=(20,20))
    
    X = data[easting][northing]
    if plot:
        plt.subplot(221)
        plt.plot(list(X)[2000:4000])
        plt.title("Série brute")
    X_t1 = transform(X, weights1, shifts1)
    X_t1.to_csv("Decomposition/Trend/"+easting+"-"+northing+".csv")
    X1 = X - X_t1
    
    if plot:
        plt.subplot(222)
        plt.plot(list(X_t1))
        plt.title("Tendance")
    
    X_s1 = transform(X1, weights2, shifts2)
    X_S1 = X_s1 - transform(X_s1, weights1, shifts1)
    X_S1.to_csv("Decomposition/Seasons/"+easting+"-"+northing+".csv")
    
    if plot:
        plt.subplot(223)
        plt.plot(list(X_S1)[2000:4000])
        plt.title("Saisonnalités")
    
    X_CVS = X - X_S1 - X_t1
    X_CVS.to_csv("Decomposition/Noise/"+easting+"-"+northing+".csv")

    if plot:
        plt.subplot(224)
        plt.plot(list(X_CVS)[2000:4000])
        plt.title("Bruit")
        plt.show()

for col in data.columns:
    print(col)
    decompose(data,col[0], col[1], weights1, shifts1, weights2, shifts2)
    BBM.plot_decomposition(data, col[0], col[1], plot = False)