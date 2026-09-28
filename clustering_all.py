

#gather all our cleaned data for clustering! 
#data mining algoritm starts here! 

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 
from sklearn.preprocessing import StandardScaler


#reading the files we worked with but user is only an id should not be sued for clustering 

"""[ ] Merge all features by user

[ ] Remove user ID from clustering

[ ] Decide whether gender is excluded from clustering

[ ] Check missing values / -1 values

[ ] Check distributions / extreme outliers

[ ] Check highly correlated features

[ ] Scale numerical features

[ ] Run soft clustering
"""

sms = pd.read_excel("data_mining_sms.xlsx")

bt = pd.read_excel("data_mining_bt.xlsx")

calls = pd.read_excel("data_mining_calls.xlsx")







print("SMS")
print(sms.head())
print(sms.shape)
print(sms.columns)

print("\nBLUETOOTH")
print(bt.head())
print(bt.shape)
print(bt.columns)

#Gemmas excel fil ska in här 
"""print("\nCALLS")
print(calls.head())
print(calls.shape)
print(calls.columns)"""






