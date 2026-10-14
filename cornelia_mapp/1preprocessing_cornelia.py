import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
import os
import matplotlib.pyplot as plt
import seaborn as sns 


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


# scaling
# clustering is based on distance between points and centroid, so having very spread values confuses our model 
# https://numpy.org/doc/stable/reference/generated/numpy.log1p.html
# Need to use Log1p before MinMaxScaler cause we have a large variation in each column.
# Log1p does also leave 0 as a 0, cause log(0+1) is always 0. 
X_log = np.log1p(X)  


# scaling makes everything in a smaller scale. Our 0's are acutal values and not NULL values, therefore MinMax-scaler is the best alternative when scaling between the columns
scaler = MinMaxScaler()
scaler.fit(X_log)
# our scaled data is turnet into an numpy array but it's going to be re-transformed back into a dataframe soon 
X_scaled = scaler.transform(X_log)
print(X_scaled) 
# now, all our values are between 0 and 1


# clustering 
#k-means returns a scatter plots with clusters
# how do we even know how many clusters there is? 
# elbow rule https://www.youtube.com/watch?v=lP7TOc1DnaI 
# elbow is a visual way of looking at inertias of different numbers of clusters 

#tar tillbaks X till så som den var innan, en dataframe och INTE en array
X = pd.DataFrame(X_scaled)
# centroid is center of a cluster
# sum of the squred distances between every datapoint and it's corresponding centroid
# the closer all points are to the centroid the smaller the inetria is 
# small inertia = stornger cluster 
# we could have a centroid for all datapoints, but we don't want this! This is why we use the elbow rule, to find the right amount of clusters. 
inertias = []

for k in range(1, 11):
    # https://scikit-learn.org/stable/common_pitfalls.html this is because KMeans starts different each time the code runs, with seed 

    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    y = model.fit_predict(X)
    inertias.append(model.inertia_)


# vi vill plotta våra inertia så vi hittar vår sweet spot. 

"""plt.figure(figsize=(12, 8))
plt.plot(range(1, 11), inertias, marker='o')
plt.xticks(ticks=range(1, 11), labels=range(1, 11))
plt.title('Inertia vs Number of Clusters')

plt.tight_layout()
plt.show()"""


# elbow set to 3 
# no when we know our elbow, we can make a new model with our specified K = 3 clusters. 
# fit and predict on our scaled data and assign it to y that is a number 0, 1, 2 that corresponds to the cluster the data corresponds to. 

model = KMeans (n_clusters=3)
y = model.fit_predict(X_scaled)
print(y)


# adding a new column to our 
# we dont need to use the scaled data a nymore, we want the insights
# therefore we use the dataset FinalTotalList2oct and labelling all 672 users into a cluster

df['CLUSTER'] = y + 1
print(df['CLUSTER'].value_counts())


# lets do an analysis of what the clusters are showing 
# look at the numeric data and datapoints 
# group by clusters and plot the data with bar charts 

# bar charts gives us important information among the clusters 
#are the charts similar to each other or do they look different? 
"""numeric_columns = df.select_dtypes(include=np.number).drop(['user', 'gender', 'CLUSTER'], axis=1).columns

fig = plt.figure(figsize=(25, 30))
for i, column in enumerate(numeric_columns):
    df_plot = df.groupby('CLUSTER')[column].mean()
    ax = fig.add_subplot(6, 5, i+1)
    ax.bar(df_plot.index, df_plot, color=sns.color_palette('Set1'), alpha=0.6)
    ax.set_title(f'Average {column.title()} per Cluster', alpha=0.5)
    ax.xaxis.grid(False)

plt.tight_layout()
plt.show()"""

#plotting further to find more interesting relations 
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 8))
sns.scatterplot(x='amount_fbf_school', y='SMS unique people O-S-H', hue='CLUSTER', data=df, palette='tab10', alpha=0.4, ax=ax1)
sns.scatterplot(x='Calls Unique People O-S-H', y='BT_Unique people O-S-H', hue='CLUSTER', data=df, palette='tab10', alpha=0.4, ax=ax2, legend=False)
sns.scatterplot(x='SMS unique people O-S-H', y='BT_Unique people O-S-H', hue='CLUSTER', data=df, palette='tab10', alpha=0.4, ax=ax3, legend=False)
sns.scatterplot(x='amount_fbf_school', y='BT_Unique people O-S-H', hue='CLUSTER', data=df, palette='tab10', alpha=0.4, ax=ax4, legend=False)

plt.tight_layout()
plt.show()











