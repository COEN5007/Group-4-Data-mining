import csv

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

gender= pd.read_csv("genders.csv")
fbf= pd.read_csv('fb_friends.csv')
calls= pd.read_csv('calls.csv')
sms= pd.read_csv('sms.csv')
bt= pd.read_csv('bt_symmetric.csv')

#clean the gender so that all numbers are filled in so it is easier with the user tag
gender.columns = ["user", "female"]
all_users = range(848)
for user in all_users:
    if user not in gender["user"].values:
        gender.loc[len(gender)] = [user, 2]


fbf.columns= ['user_1', 'user_2']
ammount_fbf=np.zeros(848,dtype=int) 
for _, row in fbf.iterrows():
    user1 = int(row["user_1"])
    user2 = int(row["user_2"])

    ammount_fbf[user1] += 1
print(ammount_fbf) #list with all number of friends in the right order 
all_users['ammount_fbf']=ammount_fbf
print(all_users)