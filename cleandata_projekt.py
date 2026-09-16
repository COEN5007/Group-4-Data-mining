import csv

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

gender= pd.read_csv("genders.csv")
fbfriends= pd.read_csv('fb_friends.csv')
calls= pd.read_csv('calls.csv')
sms= pd.read_csv('sms.csv')
bt= pd.read_csv('bt_symmetric.csv')




