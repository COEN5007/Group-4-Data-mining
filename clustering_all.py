

#gather all our cleaned data for clustering! 
#data mining algoritm starts here! 

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 


#reading the files we worked with 

sms = pd.read_excel("data_mining_sms.xlsx")

bt = pd.read_excel("data_mining_bt.xlsx")

"""här ska gemmas vara"""


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






