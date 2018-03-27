##############################################
############ Buys Ballot Model ###############
##############################################

import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.stats import norm
import pandas as pd

class BuysBallotModel():
    '''
    Buys Ballot model for time series
    '''
    def __init__(self, f, g, subd = 1, dim = 1, period = 365):
        '''
        Parameters :
            - f is a function defining the class of function considered for the trend
            ex : f(t, i) = t^i (i = 0 to dim)
            - g is a function for seasonalities
            ex : f(t, i) = int(t%period == i)
            - subd : number of subdivisions for trend estimation
            - dimension of polynome for trend estimation
            - period
        return :
            - model
        '''
        self.f = f
        self.g = g
        self.subd = subd
        self.dim = dim
        self.period = period
        self.alpha = None
        self.beta = None
    
    def fit(self, data, verbose=False):
        '''
        Fit Buys Ballot model on data
        Parameters:
            - data : time series passed
            - verbose : optional, prints
        '''
        self.F = np.zeros((len(data), self.subd*(self.dim + 1)))
        for k in range(self.dim + 1):
            for l in range(self.subd):
                self.F[:,(self.dim+1)*l+k] = self.f(np.arange(len(data)), k, l, len(data), self.subd)
        self.G = np.zeros((len(data), self.period - 1))
        for k in range(self.period - 1):
            self.G[:,k] = self.g(np.arange(len(data)), k+1)
        Row1 = np.concatenate((np.dot(self.F.T, self.F), np.dot(self.F.T, self.G)),axis=1)
        Row2 = np.concatenate((np.dot(self.G.T, self.F), np.dot(self.G.T, self.G)),axis=1)
        M = np.concatenate((Row1,Row2),axis=0)
        row1 = np.dot(self.F.T, data)
        row2 = np.dot(self.G.T, data)
        X = np.concatenate((row1, row2), axis = 0)
        params = np.dot(np.linalg.inv(M), X)
        self.alpha = params[:self.subd*(self.dim + 1)]
        self.beta = params[self.subd*(self.dim + 1):]
        if verbose:
            print("Trend coefficients : ")
            print(self.alpha)
            print("Saisonalités : ")
            print(self.beta)
    
    def get_trend(self):
        '''
        To execute after fitting. Provide the trend.
        '''
        return np.copy(np.dot(self.F, self.alpha))
    
    def get_seasonalities(self):
        '''
        To execute after fitting. Provide the trend.
        '''
        return np.copy(np.dot(self.G, self.beta))
    
    def plot_decomposition(self, data, easting, northing, n_points = 1000, plot = True):
        '''
        Plot the decomposition after fitting the model on data
        Parameters :
            - data : time series passed
            - n_points : number of points in the plot (cut)
        '''
        data = data[easting][northing]
        data = list(data)
        self.fit(data)
        trend = self.get_trend()
        pd.Series(trend).to_csv("Decomposition/Trend/BB"+easting+"-"+northing+".csv")
        seasons = self.get_seasonalities()
        pd.Series(seasons).to_csv("Decomposition/Seasons/BB"+easting+"-"+northing+".csv")
        noise = data - trend - seasons
        pd.Series(noise).to_csv("Decomposition/Noise/BB"+easting+"-"+northing+".csv")

        if plot:
            plt.figure(figsize=(20,20))
            plt.subplot(221)
            plt.plot(data[:n_points], label = "Série brute")
            plt.plot(trend[:n_points], label = "Trend")
            plt.plot(trend[:n_points] + seasons[:n_points], label = "Trend + Saisonnalités")
            plt.title("Décomposition températures ({} jours)".format(n_points))
            plt.legend()
            plt.subplot(222)
            plt.plot(trend, label = "Trend")
            plt.legend()
            plt.title('Tendance')
            plt.subplot(223)
            plt.plot(noise[:n_points])
            plt.title('Bruit (série temporelle)')
            plt.subplot(224)
            plt.hist((noise-np.mean(noise))/np.std(noise), bins=int((2*len(data))**(1/3)), normed=True)
            x = np.linspace(-3,3,500)
            plt.plot(x, norm.pdf(x), label = "N(0,1)")
            plt.title('Bruit renormalisé (distribution)')
            plt.legend()
            plt.show()