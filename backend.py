import numpy as np
import matplotlib as mp
import pandas as pd
import os as os
import matplotlib.pyplot as plt
import scipy as sp
import scipy.signal as signal
from pandas import Series
from pandas import DataFrame
import math as math
from RANSAC import RANSAC
from numpy.random import default_rng
from copy import copy



'''
Parts to finish:
    1) Identify linear sections of the data: 
        - clean up the signal
        - identify which largest,contiguous region of the graph has the most "similar" differential/slope(?)
    2) 
        - use this section (x axis region) in the original data to determine the slope of that section(?)
        - maybe compare this calculated x intercept with the x intercept yielded by using the cleaned up model?

    3) idk, maybe try doing a front end in C++ for fun? or stick to java, whatever
'''

def square_error_loss(y_true, y_pred):
    return (y_true - y_pred)**2


def mean_square_error(y_true, y_pred):
    return np.sum(square_error_loss(y_true, y_pred)) / y_true.shape[0]


class LinearRegressor:
    def __init__(self):
        self.params = None
    #find the weights or parameters
    def fit(self, x: np.ndarray, y: np.ndarray):
        r, _ = x.shape
        x = np.hstack([np.ones((r, 1)), x])
        self.params = np.linalg.inv(x.T @ x) @ x.T @ y
        return self

    def predict(self, x: np.ndarray):
        r, _ = x.shape
        x = np.hstack([np.ones((r, 1)), x])
        return x @ self.params
    def predictXint(self):
        return -self.params[0]/self.params[1]

# class variables
path = ""
while (True):
    print("provide a path to the file")
    path = input()
    if (path == 'q'):
        print("program exited")
        break
    while not os.path.isfile(path):
        print("given file is not a path: try again")
        path=input()
        continue

    excelFile = pd.ExcelFile(path)
    data = pd.read_excel(excelFile)

    print(data)

    print("name of independent variable")
    indName=input()
    print("name of dependent variable")
    depName=input()
    # read the excelFile

    print("select lower bound for B.E")
    lowerBound = float(input())
    while lowerBound < data[indName].min():
        print("lower bound is not in the domain. try again")
        lowerBound=float(input())
    print("select upper bound for B.E")
    upperBound = float(input())
    while upperBound > data[indName].max():
        print("upper bound is not in the domain. try again")
        upperBound=float(input())
    while (upperBound < lowerBound):
        print("upper bound cannot be lower than lower bound: try again")
        print("select upper Bound")
        upperBound=float(input())
        print("select lower Bound")
        lowerBound=float(input())



    restData = data[(data[indName] > lowerBound) & (data[indName] < upperBound)]
    restData = restData.reset_index(drop=True)
    # create a plot (X axis should be backward?)
    fig, axes = plt.subplots(1, 2)
    ax1 = axes[0]
    ax1.set_title("XPS Spectrum of sample - Raw")
    ax2 = axes[1]
    ax2.set_title("Noise Reduction + Linear Extrapolation of Valence Band Edge")
    window_length = len(restData)//10
    polyOrder = 5

    def savgolFilter(data, diff):
        return signal.savgol_filter(data, window_length, polyOrder)

    '''
    method that gives the minimum index given a 1d array
    '''

    ySavgol = signal.savgol_filter(restData[depName], window_length, polyOrder)
    # yDeriv=signal.savgol_filter(ySavgol,window_length,polyOrder,deriv=1)
    # yDeriv=np.gradient(ySavgol,restData[indName])
    # yDoubleDeriv=np.gradient(yDeriv,restData[indName])
    yDeriv = np.diff(ySavgol)/np.diff(restData[indName])
    # IMPORTANT: append a repeated value at the end
    yDeriv = np.append(yDeriv, yDeriv[-1])
    # IMPORTANT: append a repeated value at the end
    # yDoubleDeriv=signal.savgol_filter(restData[depName],window_length,polyOrder,deriv=2)

    savgolData = restData.copy(deep=True)
    savgolData["filtCPS"] = ySavgol
    savgolData["filtDerivCPS"] = yDeriv
    deriv_window_length = max(50, len(restData)//10)
    deriv_poly = 8
    savgolData["smoothedDeriv"] = signal.savgol_filter(
        yDeriv, deriv_window_length, deriv_poly)
    yDoubleDeriv = np.diff(
        savgolData["smoothedDeriv"])/np.diff(restData[indName])
    yDoubleDeriv = np.append(yDoubleDeriv, yDoubleDeriv[-1])
    savgolData["filtDoublDerivCPS"] = yDoubleDeriv
    savgolData["derivVariance"] = savgolData["smoothedDeriv"].rolling(
        window=len(savgolData)//20).var()
    savgolData["derivVariance2"] = savgolData["derivVariance"].rolling(
        window=len(savgolData)//20).var()

    minRow = savgolData[savgolData["derivVariance2"]
                        == savgolData["derivVariance2"].min()]
    # from the minrow, we can find the slope, and intercept(?)
    a = minRow["smoothedDeriv"]
    x = np.linspace(restData[indName].min(), restData[indName].max())
    b = minRow.filtCPS-minRow.smoothedDeriv*minRow[indName]
    y = [a*p+b for p in x]
    xint = -b/a


    # find the minimum of the derivVariance
    regressor=RANSAC(model=LinearRegressor(),loss=square_error_loss,metric=mean_square_error)
    regressor.fit(savgolData[indName].to_numpy().reshape(-1,1),savgolData["filtCPS"].to_numpy().reshape(-1,1))
    line=np.linspace(lowerBound,upperBound,num=100*math.ceil((upperBound-lowerBound))).reshape(-1,1)
    

    ax1.plot(data[indName],data[depName],label="raw Data")
    ax1.set_xlabel("Binding Energy (eV)")
    ax1.set_ylabel("Counts Per Second (CPS)")
    ax2.plot(savgolData[indName], savgolData[depName],label="Raw Data")
    ax2.set_xlabel("Binding Energy (eV)")
    ax2.set_ylabel("Counts Per Second (CPS)")
    ax2.plot(savgolData[indName], savgolData["filtCPS"],label="Savitzky-Golay")
    ax2.plot(line,regressor.predict(line),label="RANSAC linear fit")
    ax2.legend(loc='upper right',fontsize=10,bbox_to_anchor=(0.95,0.93))
    print(regressor.predictXint())
    textString="Title: "+os.path.basename(path)+"\n"+"x-intercept: "+str(regressor.predictXint())

    stats="Dataset: "+os.path.basename(path)+"\n"+'x-intercept: '+str(regressor.predictXint())
            
    bbox = dict(boxstyle='round', fc='blanchedalmond', ec='orange', alpha=0.5)
    ax2.text(0.95, 0.95, stats, fontsize=9, bbox=bbox,
            transform=ax2.transAxes, horizontalalignment='right')

    signChanges = np.diff(np.sign(savgolData["filtDoublDerivCPS"])) < 0
    localMaxIndicies = np.where(signChanges)[0]
    maxRow = savgolData[savgolData.filtDerivCPS ==
                        savgolData.filtDerivCPS[localMaxIndicies].max()]

    # 2 points to form a line: one must pass through savgolData[indName],savgolData.filtCPS
    # another can pass through  the x interceplocalMaxIndicies
    # maxRow.filtCPS=maxRow.filtDerivCPS*maxRow[indName]+c

    ax1.invert_xaxis()
    ax2.invert_xaxis()
    plt.rcParams.update({'font.size':12})
    plt.tight_layout()
    plt.show()
