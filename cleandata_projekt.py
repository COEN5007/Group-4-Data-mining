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

#Cleaning the gender data, since there are some numbers missing in users,
#these are written out with gender 2 to make it easeier to merge with the 
#other lists

gender.columns = ["user", "gender"]
all_users = pd.DataFrame({"user": range(848)}) #Creating a list with all numbers
gender = all_users.merge(gender, on="user", how="left") #Merging the new list with the gender list
gender["gender"] = gender["gender"].fillna("2").astype(int) #Write 2 as an integral if there is no annoted gender for that user

fbf.columns= ['user_1', 'user_2'] #Names the columns so both can be checked
ammount_fbf=np.zeros(848,dtype=int) #New list with zeros from 0->848
for _, row in fbf.iterrows(): #Checking through the facebook friends
    user1 = int(row["user_1"])
    user2 = int(row["user_2"])
    ammount_fbf[user1] += 1 #Adds all of the friends to the list

gender = gender.iloc[:, 1:].reset_index(drop=True)
gender['ammount_fbf']=ammount_fbf
print(gender)