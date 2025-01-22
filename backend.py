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


# class variables
path = ""
while (True):
    print("provide a path to the file")
    path = input()
    if (path == 'q'):
        print("program exited")
        break
    if not os.path.isfile(path):
        print("given file is not a path: try again")
        continue

    excelFile = pd.ExcelFile(path)

    # read the excelFile
    data = pd.read_excel(excelFile)

    print("select lower bound for B.E")
    lowerBound = float(input())
    if lowerBound < data["B.E"].min():
        print("lower bound is not in the domain. try again")
        continue
    print("select upper bound for B.E")
    uppderBound = float(input())
    if uppderBound > data["B.E"].max():
        print("upper bound is not in the domain. try again")
        continue
    if (uppderBound < lowerBound):
        print("upper bound cannot be lower than lower bound: try again")
        continue

    restData = data[(data["B.E"] > lowerBound) & (data["B.E"] < uppderBound)]
    restData = restData.reset_index(drop=True)
    # create a plot (X axis should be backward?)
    fig, axes = plt.subplots(2, 2)
    ax1 = axes[0, 0]
    ax1.set_title("Butter low pass filter")
    ax2 = axes[0, 1]
    ax2.set_title("savgol filter")
    ax3 = axes[1, 1]
    ax3.set_title("smoothed derivative of savgol filter with variance window")
    ax4 = axes[1, 0]
    ax4.set_title("extrapolated linear fit")

    # recall: row, column
    window_length = len(restData)//5
    polyOrder = 5

    def savgolFilter(data, diff):
        return signal.savgol_filter(data, window_length, polyOrder)

    def butter_lowpass_filter(data, minSamplingRate, order):
        # sampling frequency should be the number of samples per unit change of B.E
        fCrit = 1/minSamplingRate
        fsample = 1/(data.iloc[0, 0]-data.iloc[1, 0])
        fn = fsample/2
        fcNorm = fCrit/fn

        filter = signal.butter(N=order, Wn=minSamplingRate,
                               btype='low', analog=False, fs=fsample, output='sos')
        filteredSignal = signal.sosfilt(filter, data.iloc[:, 1])
        return filteredSignal

    a = butter_lowpass_filter(restData, 10, 5)
    butterData = pd.DataFrame()
    butterData["B.E"] = restData["B.E"]
    butterData["CPS"] = a

    butterData["butterDeriv"] = np.gradient(
        butterData["CPS"], butterData["B.E"])
    butterData["butterDoubleDeriv"] = np.gradient(np.gradient(
        butterData["CPS"], butterData["B.E"]), butterData["B.E"])

    '''
    method that gives the minimum index given a 1d array
    '''

    ySavgol = signal.savgol_filter(restData["CPS"], window_length, polyOrder)
    # yDeriv=signal.savgol_filter(ySavgol,window_length,polyOrder,deriv=1)
    # yDeriv=np.gradient(ySavgol,restData["B.E"])
    # yDoubleDeriv=np.gradient(yDeriv,restData["B.E"])
    yDeriv = np.diff(ySavgol)/np.diff(restData["B.E"])
    # IMPORTANT: append a repeated value at the end
    yDeriv = np.append(yDeriv, yDeriv[-1])
    # IMPORTANT: append a repeated value at the end
    # yDoubleDeriv=signal.savgol_filter(restData["CPS"],window_length,polyOrder,deriv=2)

    savgolData = restData.copy(deep=True)
    savgolData["filtCPS"] = ySavgol
    savgolData["filtDerivCPS"] = yDeriv
    deriv_window_length = max(50, len(data)//10)
    deriv_poly = 8
    savgolData["smoothedDeriv"] = signal.savgol_filter(
        yDeriv, deriv_window_length, deriv_poly)
    yDoubleDeriv = np.diff(
        savgolData["smoothedDeriv"])/np.diff(restData["B.E"])
    yDoubleDeriv = np.append(yDoubleDeriv, yDoubleDeriv[-1])
    savgolData["filtDoublDerivCPS"] = yDoubleDeriv
    savgolData["derivVariance"] = savgolData["smoothedDeriv"].rolling(
        window=len(data)//25).var()

    minRow = savgolData[savgolData["derivVariance"]
                        == savgolData["derivVariance"].min()]
    # from the minrow, we can find the slope, and intercept(?)
    a = minRow["smoothedDeriv"]
    x = np.linspace(restData["B.E"].min(), restData["B.E"].max())
    b = minRow.filtCPS-minRow.smoothedDeriv*minRow["B.E"]
    y = [a*p+b for p in x]
    xint = -b/a

    # find the minimum of the derivVariance

    ax1.plot(restData["B.E"], restData["CPS"])
    ax1.plot(butterData["B.E"], butterData["CPS"])

    ax2.plot(savgolData["B.E"], savgolData["CPS"])
    ax2.plot(savgolData["B.E"], savgolData["filtCPS"])

    ax3.plot(savgolData["B.E"], savgolData["filtDerivCPS"])
    ax3.plot(savgolData["B.E"], savgolData["smoothedDeriv"])
    ax3.plot(savgolData["B.E"], savgolData["derivVariance"])
    ax3.scatter(minRow["B.E"], minRow["derivVariance"])

    ax4.plot(savgolData["B.E"], savgolData["filtCPS"])
    ax4.plot(x, y)
    ax4.scatter(xint, 0)
    print(xint)

    signChanges = np.diff(np.sign(savgolData["filtDoublDerivCPS"])) < 0
    localMaxIndicies = np.where(signChanges)[0]
    maxRow = savgolData[savgolData.filtDerivCPS ==
                        savgolData.filtDerivCPS[localMaxIndicies].max()]
    ax3.scatter(savgolData["B.E"].iloc[localMaxIndicies],
                savgolData["filtDerivCPS"].iloc[localMaxIndicies])

    # 2 points to form a line: one must pass through savgolData["B.E"],savgolData.filtCPS
    # another can pass through  the x interceplocalMaxIndicies
    # maxRow.filtCPS=maxRow.filtDerivCPS*maxRow["B.E"]+c

    ax1.invert_xaxis()
    ax2.invert_xaxis()
    ax3.invert_xaxis()
    ax4.invert_xaxis()
    plt.show()
