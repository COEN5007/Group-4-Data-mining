import csv

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

gender= pd.read_csv("genders.csv")
fbf= pd.read_csv('fb_friends.csv')
calls= pd.read_csv('calls.csv') # Den relevanta för mig
sms= pd.read_csv('sms.csv')
bt= pd.read_csv('bt_symmetric.csv')



#våra orignalcolumner utan sh och osh
calls.columns = ["timestamp", "caller", "callee", "duration"]

# ta bort alla samtal som är 0 sek 
calls = calls[calls["duration"] != 0].copy()

# -1 = missed call
calls["missed"] = calls["duration"] == -1
calls["duration"] = calls["duration"].astype(float).mask(calls["missed"])


# Fridas dokument finns detta redan i

#Har gjort om tidsintervallet nu så att alla har samma tidsintervall 
SECONDS_PER_DAY = 24 * 60 * 60
SCHOOL_START = 8 * 60 * 60       # 08:00
SCHOOL_END = 17 * 60 * 60        # 17:00
# Dataset starts on a Sunday
# 0 = Sunday, 1 = Monday, ..., 6 = Saturday
calls["day_of_week"] = (
    calls["timestamp"] // SECONDS_PER_DAY
) % 7

# Time within the relative day
calls["time_of_day"] = (
    calls["timestamp"] % SECONDS_PER_DAY
)

# Everything is initially outside school hours
calls["period"] = "O-S-H"

# School hours:
# Monday-Friday, 08:00-17:00
calls.loc[
    (calls["day_of_week"] >= 1) &
    (calls["day_of_week"] <= 5) &
    (calls["time_of_day"] >= SCHOOL_START) &
    (calls["time_of_day"] < SCHOOL_END),
    "period"
] = "S-H"
# Ta bort raderna ovanför när mergear in i clean data






# Dubbel telefontid; en rad som caller, en rad som callee,
#  (samtalstiden räknas för båda personer)

# caller
caller_calls = calls[
    ["caller", "callee", "duration", "missed", "period"]
].copy()

caller_calls = caller_calls.rename(columns={
    "caller": "user",
    "callee": "contact"
})
caller_calls["role"]= "caller"

# callee
callee_calls = calls[
    ["callee", "caller", "duration", "missed", "period"]
].copy()

callee_calls = callee_calls.rename(columns={
    "callee": "user",
    "caller": "contact"
})

callee_calls["role"] = "callee"

# interaktionen, concat() staplar dem
call_interactions = pd.concat(
    [caller_calls, callee_calls],
    ignore_index=True
)



def create_call_features(data, period):

    d = data[data["period"] == period].copy()

    # Total samtalstid där NaN ignoreras mha av sum
    call_time = (
        d.groupby("user")["duration"]
        .sum()
    )

    # Ringda samtal, användaren är "caller" och samtalet besvaras
    #   ~d betyder där d INTE gäller
    calls_made = (d[(d["role"] == "caller") & (~d["missed"])].groupby("user").size())

    # Mottagna samtal, användaren är "callee" och samtalet besvarades
    calls_received = (d[(d["role"] == "callee") & (~d["missed"])].groupby("user").size())


    # Ringde men fick inget svar (missat samtal, användare är "caller")
    missed_as_caller = (
        d[(d["role"] == "caller") & (d["missed"])]
        .groupby("user")
        .size()
    )

    # Antal missade samtal (som caller eller callee)
    missed_as_callee = (
        d[(d["role"] == "callee") & (d["missed"])].groupby("user").size())

    # Unika personer man haft kontakt med (oavsett missat/ej)
    unique_people = (d.groupby("user")["contact"].nunique())

    users = d["user"].unique()
    result = pd.DataFrame({"user": users})



    result["call_time"] = result["user"].map(call_time).fillna(0)

    result["calls_made"] = result["user"].map(calls_made).fillna(0).astype(int)

    result["calls_received"] = result["user"].map(calls_received).fillna(0).astype(int)

    result["missed_as_caller"] = result["user"].map(missed_as_caller).fillna(0).astype(int)

    result["missed_as_callee"] = result["user"].map(missed_as_callee).fillna(0).astype(int)

    result["unique_people"] = result["user"].map(unique_people).fillna(0).astype(int)

    return result



calls_school = create_call_features(call_interactions, "S-H")
calls_off_school = create_call_features(call_interactions, "O-S-H")

# bara för att kunna testa merge lokalt.  ---------------------
features = pd.DataFrame({
    "user_1": sorted(
        pd.concat([calls["caller"], calls["callee"]]).unique()
    )
})
#---------------------------------------------------------------

# S-H
calls_school = calls_school.rename(columns={
    "call_time": "Duration S-H",
    "calls_made": "Calls Made S-H",
    "calls_received": "Calls Received S-H",
    "missed_as_caller": "Missed As Caller S-H",
    "missed_as_callee": "Missed As Callee S-H",
    "unique_people": "Unique People S-H"
})

features = features.merge(
    calls_school,
    left_on="user_1",
    right_on="user",
    how="left"
).drop(columns=["user"])


# O-S-H
calls_off_school = calls_off_school.rename(columns={
    "call_time": "Duration O-S-H",
    "calls_made": "Calls Made O-S-H",
    "calls_received": "Calls Received O-S-H",
    "missed_as_caller": "Missed As Caller O-S-H",
    "missed_as_callee": "Missed As Callee O-S-H",
    "unique_people": "Unique People O-S-H"
})

features = features.merge(
    calls_off_school,
    left_on="user_1",
    right_on="user",
    how="left"
).drop(columns=["user"])


call_columns = [
    "Duration S-H",
    "Duration O-S-H",
    "Calls Made S-H",
    "Calls Made O-S-H",
    "Calls Received S-H",
    "Calls Received O-S-H",
    "Missed As Caller S-H",
    "Missed As Caller O-S-H",
    "Missed As Callee S-H",
    "Missed As Callee O-S-H",
    "Unique People S-H",
    "Unique People O-S-H"
]

features[call_columns] = features[call_columns].fillna(0)




# TESTAR KODEN MED FÖLJANDE RADER

# print(features[features["user_1"] == 200])

#print(calls["timestamp"].describe())     # stora tal, inte 0-847
#print(calls["period"].value_counts())    # ska visa både S-H och O-S-H
#print(features[features["Duration S-H"] > 0].head(10))
features = (
    features
    .set_index("user_1")
    .reindex(range(851), fill_value=0) #stod 850 här innan men eftersom python är 0 indexerat försvinner den 851 
    .rename_axis("user_1")
    .reset_index()
)

print("\n--- CALL CHECK ---")

print("Number of users:", len(features))

print("\nCalls by period:")
print(calls["period"].value_counts())

print("\nCalls by day:")
print(calls["day_of_week"].value_counts().sort_index())

# Check how many users have any call activity
has_calls = (
    features["Calls Made S-H"] +
    features["Calls Made O-S-H"] +
    features["Calls Received S-H"] +
    features["Calls Received O-S-H"] +
    features["Missed As Caller S-H"] +
    features["Missed As Caller O-S-H"] +
    features["Missed As Callee S-H"] +
    features["Missed As Callee O-S-H"]
) > 0

print("\nUsers with calls:", has_calls.sum())
print("Users without calls:", (~has_calls).sum())

print("\nFirst 20 users:")
print(features.head(20))
#print(features)
features.to_excel("festures_users2.xlsx", index=False)
#features.to_csv("calls_features.csv", index=False, sep=";") #skapar en csv-fil (calls_features.csv) som alla kan se

