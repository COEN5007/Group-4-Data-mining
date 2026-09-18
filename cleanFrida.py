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
gender.columns = ["user", "gender"]

all_users = pd.DataFrame({"user": range(851)})

gender = all_users.merge(gender, on="user", how="left")
gender["gender"] = gender["gender"].fillna("2").astype(int)

fbf.columns= ['user_1', 'user_2']
ammount_fbf=np.zeros(851,dtype=int) 
max_user = int(max(fbf["user_1"].max(), fbf["user_2"].max()))

ammount_fbf = [0] * (max_user + 1)

for _, row in fbf.iterrows():
    user1 = int(row["user_1"])
    user2 = int(row["user_2"])

    ammount_fbf[user1] += 1
    ammount_fbf[user2] += 1


total = gender.iloc[:, 1:].reset_index(drop=True)
total['ammount_fbf']=ammount_fbf
print(total)

