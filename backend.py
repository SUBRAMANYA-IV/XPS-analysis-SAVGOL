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
path=""
while(True): 
    print("provide a path to the file")
    path=input()
    if(path=='q'):
        print("program exited")
        break
    path="/home/vedaant/python-for-data-science/linear-regression/VBM2.xlsx"
    if not os.path.isfile(path):
        print("given file is not a path: try again")
        continue
    

    excelFile=pd.ExcelFile(path)

    #read the excelFile
    data=pd.read_excel(excelFile)
    print(data["B.E"].min())

    print("select lower bound for B.E")
    lowerBound=float(input())
    if lowerBound<data["B.E"].min():
        print("lower bound is not in the domain. try again")
        continue
    print("select upper bound for B.E")
    uppderBound=float(input())
    if uppderBound>data["B.E"].max():
        print("upper bound is not in the domain. try again")
        continue
    if(uppderBound<lowerBound):
        print("upper bound cannot be lower than lower bound: try again")
        continue

    restData=data[(data["B.E"]>lowerBound)& (data["B.E"]<uppderBound)]
    restData=restData.reset_index(drop=True)
    #create a plot (X axis should be backward?)
    fig,axes=plt.subplots(2,2)
    ax1=axes[0,0]
    ax1.set_title("Butter low pass filter")
    ax2=axes[0,1]
    ax2.set_title("savgol filter")
    ax3=axes[1,1]
    ax3.set_title("first and second derivative of function")
    ax4=axes[1,0]

    #perform some nosie filtering?
    #using a savitzky-golay filter?

    window_length=len(restData["CPS"])
    polyOrder=window_length-5
    if(window_length%2)==0:
        window_length=window_length-1
    else:
        polyOrder=polyOrder-1


    #recall: row, column
    window_length=21
    polyOrder=5

    def savgolFilter(data,diff):
        return signal.savgol_filter(data,window_length,polyOrder)

    def butter_lowpass_filter(data,minSamplingRate,order):
        #sampling frequency should be the number of samples per unit change of B.E 
        fCrit=1/minSamplingRate
        fsample=1/(data.iloc[0,0]-data.iloc[1,0])
        fn=fsample/2
        fcNorm=fCrit/fn

        filter=signal.butter(N=order,Wn=minSamplingRate,btype='low',analog=False,fs=fsample,output='sos')
        filteredSignal=signal.sosfilt(filter,data.iloc[:,1])
        return filteredSignal

    a=butter_lowpass_filter(restData,8,5)
    butterData=pd.DataFrame()
    butterData["B.E"]=restData["B.E"]
    butterData["CPS"]=a

    butterData["butterDeriv"]=np.gradient(butterData["CPS"],butterData["B.E"])
    butterData["butterDoubleDeriv"]=np.gradient(np.gradient(butterData["CPS"],butterData["B.E"]),butterData["B.E"])



    ySavgol=signal.savgol_filter(restData["CPS"],window_length,polyOrder)
    yDeriv=signal.savgol_filter(restData["CPS"],window_length,polyOrder,deriv=1)
    yDoubleDeriv=signal.savgol_filter(restData["CPS"],window_length,polyOrder,deriv=2)

    savgolData=restData.copy(deep=True)
    savgolData["filtCPS"]=ySavgol
    savgolData["filtDerivCPS"]=yDeriv
    savgolData["filtDoublDerivCPS"]=yDoubleDeriv


    ax1.plot(restData["B.E"],restData["CPS"])
    ax1.plot(butterData["B.E"],butterData["CPS"])

    ax2.plot(restData["B.E"],savgolData["filtCPS"])
    ax2.plot(savgolData["B.E"],savgolData["CPS"])

    ax3.plot(savgolData["B.E"],savgolData["filtDerivCPS"])
    ax3.plot(savgolData["B.E"],savgolData["filtDoublDerivCPS"])

    ax4.plot(butterData["B.E"],butterData["butterDeriv"])
    ax4.plot(butterData["B.E"],butterData["butterDoubleDeriv"])

    

    #most linear section should occur when doubleDeriv=0 AND when firstDeriv is a min

    #find where the second derivative changes sign
    signChanges=np.diff(np.sign(savgolData["filtDoublDerivCPS"]))<0
    #SHOULD BE MAX INDICIES
    localMaxIndicies=np.where(signChanges)[0]
    maxRow=savgolData[savgolData.filtDerivCPS==savgolData.filtDerivCPS[localMaxIndicies].max()]
    ax3.scatter(savgolData["B.E"][localMaxIndicies],savgolData["filtDerivCPS"][localMaxIndicies])
    #ax3.scatter(maxRow["B.E"],maxRow["filtDerivCPS"],marker="X")

    # 2 points to form a line: one must pass through savgolData["B.E"],savgolData.filtCPS
    # another can pass through  the x interceplocalMaxIndicies
    # maxRow.filtCPS=maxRow.filtDerivCPS*maxRow["B.E"]+c
    p1=(maxRow["B.E"],maxRow.filtCPS)
    b=maxRow["B.E"]*-maxRow.filtDerivCPS+maxRow.filtCPS
    xint=-b/maxRow.filtDerivCPS

    plt.show()


