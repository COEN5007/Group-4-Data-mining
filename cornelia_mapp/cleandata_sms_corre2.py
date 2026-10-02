#cleaning sms conversations 

import pandas as pd #python library for working with data, can give us answers about the data we're analysing 
import numpy as np 
import seaborn as sns 
import matplotlib.pyplot as plt
from datetime import datetime
import csv
sms = pd.read_csv("sms.csv")
sms.info()
sms.head()
sms.duplicated() #checking for duplicates 
sms.isnull().sum()
print(sms)
sms.columns = ["timestamp", "sender", "recipient"]

"""#checkar om jag verkligen inte har några tomma värden 
print(sms.isnull().sum())


#identifying column data types all are numerical 
num_col = [col for col in sms.columns if sms[col].dtype != 'object']
print('Numerical columns:', num_col)
sms[num_col].nunique()

all_users = pd.DataFrame({"user" : range(851)})
print(all_users)

sms_sent = sms["sender"].value_counts()

data = pd.read_csv("/content/nba.csv", index_col="Name")

row = data.loc["Avery Bradley"]
print(row)

sms["hour"] = (sms["timestamp"] // 3600) % 24"""


# cleaning sms conversations

import pandas as pd
import numpy as np


#fixing all time stamps för the dataset, started at sunday
# seconds in one day
# seconds in one day
SECONDS_PER_DAY = 24 * 60 * 60

# Relative hour within the day
sms["hour"] = (sms["timestamp"] // 3600) % 24

# Dataset starts on a Sunday
# 0 = Sunday, 1 = Monday, ..., 6 = Saturday
sms["day_of_week"] = (
    sms["timestamp"] // SECONDS_PER_DAY
) % 7

# Everything is initially outside school hours
sms["period"] = "O-S-H"

# School hours:
# Monday-Friday, 08:00-17:00
sms.loc[
    (sms["day_of_week"] >= 1) &
    (sms["day_of_week"] <= 5) &
    (sms["hour"] >= 8) &
    (sms["hour"] < 17),
    "period"
] = "S-H"


#trying first to select one column and then just printing it all 

"""data = pd.read_csv("sms.csv", index_col="timestamp")
print("Dataset")
print(data.head(5))

first = data["sender"]
print("\nsender selected from Dataset")
print(first.head(5))"""


#sms conversation fix 
# need to make the pairs the same
# for example 1 -> 5 and 5 -> 1 should both be 1 and 5

sms["user_1"] = sms[["sender", "recipient"]].min(axis=1)
sms["user_2"] = sms[["sender", "recipient"]].max(axis=1)

# sorting so messages between the same people are in time order frommscratch 

sms = sms.sort_values(["user_1", "user_2", "timestamp"])

# find time between the messages
sms["time_diff"] = sms.groupby(
    ["user_1", "user_2"]
)["timestamp"].diff()

# we decided that a conversation is 30 minutes
conversation_gap = 30 * 60

# if it has been more than 30 minutes it is a new conversation
# isna is needed because the first message does not have a message before it
sms["new_conversation"] = (
    sms["time_diff"].isna() |
    (sms["time_diff"] > conversation_gap)
)

# number the conversations
sms["conversation_id"] = sms.groupby(
    ["user_1", "user_2"]
)["new_conversation"].cumsum()


# making another table where one row is one conversation
conversations = sms.groupby(
    ["user_1", "user_2", "conversation_id"]
).agg(
    start=("timestamp", "min"),
    period=("period", "first")
).reset_index()

# period first means that if conversation starts during school hours
# but ends after school hours it is still counted as school hours
#fixing down under for all the users 

# all users in the data set
users = pd.DataFrame({"user": range(851)})

# sms sent during school hours
school_sms = sms[sms["period"] == "S-H"]
sent_sh = school_sms.groupby("sender").size()
#.map(sant_sh) means to find the eprson in sent_sh and add amount of sms sent 
users["sms sent S-H"] = users["user"].map(sent_sh)
users["sms sent S-H"] = users["sms sent S-H"].fillna(0)
users["sms sent S-H"] = users["sms sent S-H"].astype(int)


# sms sent outside school hours 

off_school_sms = sms[sms["period"] == "O-S-H"]
sent_osh = off_school_sms.groupby("sender").size()

users["sms sent O-S-H"] = users["user"].map(sent_osh)
users["sms sent O-S-H"] = users["sms sent O-S-H"].fillna(0)
users["sms sent O-S-H"] = users["sms sent O-S-H"].astype(int)


# received during school hours
received_sh = school_sms.groupby("recipient").size()

users["sms received S-H"] = users["user"].map(received_sh)
users["sms received S-H"] = users["sms received S-H"].fillna(0)
users["sms received S-H"] = users["sms received S-H"].astype(int)


# received outside school hours
received_osh = off_school_sms.groupby("recipient").size()

users["sms received O-S-H"] = users["user"].map(received_osh)
users["sms received O-S-H"] = users["sms received O-S-H"].fillna(0)
users["sms received O-S-H"] = users["sms received O-S-H"].astype(int)


# need both sent and received sms because we want everyone
# the user has had sms contact with

sent_contacts = sms[["sender", "recipient", "period"]].copy()
sent_contacts.columns = ["user", "contact", "period"]

received_contacts = sms[["recipient", "sender", "period"]].copy()
received_contacts.columns = ["user", "contact", "period"]

contacts = pd.concat([sent_contacts, received_contacts])


# unique people during at school hours 
contacts_sh = contacts[contacts["period"] == "S-H"]

unique_sh = contacts_sh.groupby("user")["contact"].nunique()

users["unique people S-H"] = users["user"].map(unique_sh)
users["unique people S-H"] = users["unique people S-H"].fillna(0)
users["unique people S-H"] = users["unique people S-H"].astype(int)


# unique people outside school hours
contacts_osh = contacts[contacts["period"] == "O-S-H"]

unique_osh = contacts_osh.groupby("user")["contact"].nunique()

users["unique people O-S-H"] = users["user"].map(unique_osh)
users["unique people O-S-H"] = users["unique people O-S-H"].fillna(0)
users["unique people O-S-H"] = users["unique people O-S-H"].astype(int)



# conversation needs to count for both people

# need to count the conversations for both users
# because both people are part of the conversation
#every conversation is between two people so we give the conversations to both before we count amount conversatins per user 

conv1 = conversations[["user_1", "period"]]
conv2 = conversations[["user_2", "period"]]
# changing names so i can put them together
conv1 = conv1.rename(columns={"user_1": "user"})
conv2 = conv2.rename(columns={"user_2": "user"})
#every conversation is connected together 
all_conversations = pd.concat([conv1, conv2])
# conversations in school hours
school_conversations = all_conversations[
    all_conversations["period"] == "S-H"
]

# count how many each user has
conversation_sh = school_conversations.groupby("user").size()
# put the count into the users table
users["conversation S-H"] = users["user"].map(conversation_sh)



# people that dont have any gets 0
users["conversation S-H"] = users["conversation S-H"].fillna(0)

# want whole numbers
users["conversation S-H"] = users["conversation S-H"].astype(int)




# conversations off school hours

offschool_conversations = all_conversations[
    all_conversations["period"] == "O-S-H"
]

conversation_osh = offschool_conversations.groupby("user").size()

users["conversation O-S-H"] = users["user"].map(conversation_osh)

users["conversation O-S-H"] = users["conversation O-S-H"].fillna(0)

users["conversation O-S-H"] = users["conversation O-S-H"].astype(int)



# finalyyy table
print(users.head(20))


users.to_excel("data_mining_sms.xlsx", index=False)

### ta bort sen 
print("\n--- CHECK ---")

print("Total raw SMS:", len(sms))

print(
    "S-H + O-S-H:",
    len(school_sms) + len(off_school_sms)
)

print("\nSMS by period:")
print(sms["period"].value_counts())

print("\nSMS by day:")
print(sms["day_of_week"].value_counts().sort_index())

print("\nDay mapping:")
print("0 = Sunday")
print("1 = Monday")
print("2 = Tuesday")
print("3 = Wednesday")
print("4 = Thursday")
print("5 = Friday")
print("6 = Saturday")

print("\n--- SMS DESCRIPTIVE STATISTICS ---")

# SMS feature columns
sms_columns = [
    "sms sent S-H",
    "sms sent O-S-H",
    "sms received S-H",
    "sms received O-S-H",
    "unique people S-H",
    "unique people O-S-H",
    "conversation S-H",
    "conversation O-S-H"
]

# Descriptive statistics
sms_stats = users[sms_columns].describe().T[
    ["min", "25%", "50%", "mean", "75%", "max", "std"]
].round(2)

print(sms_stats)





# Check how many users have SMS data

sms_columns = [
    "sms sent S-H",
    "sms sent O-S-H",
    "sms received S-H",
    "sms received O-S-H"
]

# User has SMS data if they have sent OR received at least one SMS
has_sms = (users[sms_columns].sum(axis=1) > 0)

print("\n--- SMS USER CHECK ---")
print("Total users:", len(users))
print("Users with SMS data:", has_sms.sum())
print("Users without SMS data:", (~has_sms).sum())














