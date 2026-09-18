#cleaning sms conversations 

import pandas as pd #python library for working with data, can give us answers about the data we're analysing 
import numpy as np 
import seaborn as sns 

# duplicatescleaned_df = df.drop_duplicates() # vet ej om denna behövs 
sms = pd.read_csv("sms.csv")

print (sms)

duplicatescleaned_sms = sms.drop_duplicates() #tar bort duplicates 

print (duplicatescleaned_sms)

sms.cleaned = sns.load_dataset("duplicatescleaned_sms")



