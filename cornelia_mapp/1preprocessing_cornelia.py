import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os


# plan is to make a csv file that preprocessed the final total list from october 2nd. 
# the data is clean, no duplicates, no -1 for bt connections, still a lot of zeros in sms and calls but there is still data for calls where sms is zero so its hard to determine what to do
# plan now: get the data to fit for the algoritm

# user is and id and has nothing to do with statistics for now
# gender is removed and can be taken into account later but for now we use all other data 
# 29 social features are left
# now we transitioning into the machine learning part
# this yt video helped me understand what needs to be done 
# https://www.youtube.com/watch?v=lP7TOc1DnaI 

# start with preprocessing for k means
# https://www.geeksforgeeks.org/python/os-module-python-examples/ need to change directory to cornelia_mapp for now with os 
# anyways, dropping user and gender
mapp = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(mapp, 'FinalTotalList2oct.csv'))
print(df.head(10))

# Ta bort user (id) och gender (inte socialt beteende), som customer_id i videon
# x are our variables, the fit method, and then transofrm to 
X = df.drop(['user', 'gender'], axis=1)

# clustering is based on distance between points and centroid, so having very spread values confuses our model 
# scaling makes everything in a smaller scale. Our 0's are acutal values and not NULL values, therefore MinMax-scaler is the best alternative
scaler = MinMaxScaler()
scaler.fit(X)
# our scaled data is turnet into an numpy array but it's going to be re-transformed back into a dataframe soon 
X_scaled = scaler.transform(X)
print(X_scaled) 


# clustering
# how do we even know how many clusters there is? 
# elbow rule - 



