import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


data = pd.read_csv("total_list.csv")

print(data.shape)
print(data.head(10))
print(data.describe())


# creating a histogram

#genders 
"""plt.hist(data['gender'])
plt.title('Deviation of gender among the students')
plt.xlabel('1 = female, 0 = male')
plt.ylabel('students amt.')
plt.show()
"""
#facebook friends 
"""plt.hist(data['fb_friends'])
plt.title('FB friends')
plt.xlabel('friends amt.')
plt.ylabel('students amt.')
plt.show()"""

# school hours sms sent
plt.hist(data['sms_sent_S-H'])
plt.title(' school hours')
plt.xlabel('sms sent')
plt.ylabel('students amt.')
plt.show()


plt.hist(data['sms_sent_S-H'])
plt.title(' school hours')
plt.xlabel('duration')
plt.ylabel('students amt.')
plt.show()