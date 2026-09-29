import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


data = pd.read_csv("total_list.csv")

print(data.shape)
print(data.head(10))
print(data.describe())


# creating a histogram
#
plt.hist(data['calls_made_S-H'])
plt.title('School hours')
plt.xlabel('duration')
plt.ylabel('students amt.')
plt.show


plt.hist(data['calls_made_O-S-H'])
plt.title('Off school hours')
plt.xlabel('duration')
plt.ylabel('students amt.')
plt.show()


