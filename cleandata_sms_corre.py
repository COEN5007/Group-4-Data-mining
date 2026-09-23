#cleaning sms conversations 

import pandas as pd #python library for working with data, can give us answers about the data we're analysing 
import numpy as np 
import seaborn as sns 
import matplotlib.pyplot as plt

# duplicatescleaned_df = df.drop_duplicates() # vet ej om denna behövs 
sms = pd.read_csv("sms.csv")
"""sms.info()
sms.head()
sms.duplicated() #checking for duplicates 
sms.isnull().sum()
#checkar om jag verkligen inte har några tomma värden 
print(sms.isnull().sum())


#identifying column data types all are numerical 
num_col = [col for col in sms.columns if sms[col].dtype != 'object']
print('Numerical columns:', num_col)
sms[num_col].nunique()

all_users = pd.DataFrame({"user" : range(851)})
print(all_users)

sms_sent = sms["sender"].value_counts()"""

sms 







#round((sms.isnull().sum() / sms.shape[0]) * 100, 2) #sms.isnull detects missing values and returns boolean dataframe behöver inte denna 




#dropping irrelevant data columns / heavy missing columns that can disturb 

#duplicatescleaned_sms = sns.drop_duplicates() #tar bort duplicates 

#print (duplicatescleaned_sms)

#sms_cleaned = sms.load_dataset("duplicatescleaned_sms")
#vilka som skicar, vem de skickar till och när de skcikar. 











