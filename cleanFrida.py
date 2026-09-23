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

gender.columns = ["user", "gender"]

all_users = pd.DataFrame({"user": range(851)})

gender = all_users.merge(gender, on="user", how="left")
gender["gender"] = gender["gender"].fillna("2").astype(int)

fbf.columns= ['user_1', 'user_2']
ammount_fbf=np.zeros(851,dtype=int) 
max_user = int(max(fbf["user_1"].max(), fbf["user_2"].max()))

ammount_fbf = [0] * (max_user + 1)

for _, row in fbf.iterrows():
    user1 = int(row["user_1"])
    user2 = int(row["user_2"])

    ammount_fbf[user1] += 1
    ammount_fbf[user2] += 1


total = gender.iloc[:, 1:].reset_index(drop=True)
total['ammount_fbf']=ammount_fbf
#print(total)

bt.columns=["timestamp", "user_1", "user_2", "RSSI"]

interactions=["Count interactions", "Unique users", "Interaction duration", "RSSI_mean"]

bt = bt[bt["RSSI"] != 0]

bt = bt.sort_values(
    ["user_1", "user_2", "timestamp"]
).copy()

bt["time_diff"] = (
    bt.groupby(["user_1", "user_2"])["timestamp"]
      .diff()
)

bt["new_interaction"] = (
    bt["time_diff"].isna() |
    (bt["time_diff"] > 600)
)

bt["interaction_id"] = (
    bt.groupby(["user_1", "user_2"])["new_interaction"]
      .cumsum()
)

interactions = (
    bt.groupby(["user_1", "user_2", "interaction_id"])
      .agg(
          start=("timestamp", "min"),
          end=("timestamp", "max"),
          avg_rssi=("RSSI", "mean")
      )
      .reset_index()
)

interactions["duration"] = (
    interactions["end"] - interactions["start"] + 300
)

interactions = interactions.drop(
    columns=["interaction_id"])
#print(interactions)


SECONDS_PER_DAY = 24 * 60 * 60

SCHOOL_START = 8 * 60 * 60   # 08:00
SCHOOL_END = 17 * 60 * 60     # 17:00

interactions["day_of_week"] = (
    interactions["start"] // SECONDS_PER_DAY
) % 7

interactions["time_of_day"] = (
    interactions["start"] % SECONDS_PER_DAY
)
interactions["period"] = "O-S-H"

interactions.loc[
    interactions["day_of_week"] >= 5,
    "period"
] = "O-S-H"

interactions.loc[
    (interactions["day_of_week"] < 5) &
    (interactions["time_of_day"] >= SCHOOL_START) &
    (interactions["time_of_day"] < SCHOOL_END),
    "period"
] = "S-H"


user_1_interactions = interactions[
    ["user_1", "user_2", "duration", "avg_rssi", "period"]
].copy()

user_1_interactions = user_1_interactions.rename(columns={
    "user_1": "user",
    "user_2": "contact"
})


user_2_interactions = interactions[
    ["user_2", "user_1", "duration", "avg_rssi", "period"]
].copy()

user_2_interactions = user_2_interactions.rename(columns={
    "user_2": "user",
    "user_1": "contact"
})


user_interactions = pd.concat(
    [user_1_interactions, user_2_interactions],
    ignore_index=True
)

def create_features(data, period):

    d = data[data["period"] == period].copy()

    # Alla interaktioner
    result = (
        d.groupby("user")
        .agg(
            interactions=("contact", "size"),
            average_duration=("duration", "mean"),
            average_rssi=("avg_rssi", "mean")
        )
        .reset_index()
    )

    # Unika personer
    # -2 ska INTE räknas som en specifik person
    unique_people = (
        d[d["contact"] >= 0]
        .groupby("user")["contact"]
        .nunique()
    )

    # Antal outsider-interaktioner
    outsiders = (
        d[d["contact"] == -2]
        .groupby("user")
        .size()
    )

    result["unique_people"] = (
        result["user"]
        .map(unique_people)
        .fillna(0)
        .astype(int)
    )

    result["outsiders"] = (
        result["user"]
        .map(outsiders)
        .fillna(0)
        .astype(int)
    )

    return result


school = create_features(
    user_interactions,
    "S-H"
)

off_school = create_features(
    user_interactions,
    "O-S-H"
)

users = sorted(
    user_interactions[
        user_interactions["user"] >= 0
    ]["user"].unique()
)

features = pd.DataFrame({
    "user_1": users
})


# S-H
school = school.rename(columns={
    "interactions": "Interactions S-H",
    "unique_people": "Unique people S-H",
    "outsiders": "Outsiders S-H",
    "average_duration": "Average Duration S-H",
    "average_rssi": "Average RSSI S-H"
})

features = features.merge(
    school,
    left_on="user_1",
    right_on="user",
    how="left"
).drop(columns=["user"])


# O-S-H
off_school = off_school.rename(columns={
    "interactions": "Interactions O-S-H",
    "unique_people": "Unique people O-S-H",
    "outsiders": "Outsiders O-S-H",
    "average_duration": "Average Duration O-S-H",
    "average_rssi": "Average RSSI O-S-H"
})

features = features.merge(
    off_school,
    left_on="user_1",
    right_on="user",
    how="left"
).drop(columns=["user"])


# Saknade värden = 0
features = features.fillna(0)


# Avrunda medelvärden
features["Average Duration S-H"] = (
    features["Average Duration S-H"].round(2)
)

features["Average Duration O-S-H"] = (
    features["Average Duration O-S-H"].round(2)
)

features["Average RSSI S-H"] = (
    features["Average RSSI S-H"].round(2)
)

features["Average RSSI O-S-H"] = (
    features["Average RSSI O-S-H"].round(2)
)


#print(features)

features = features.set_index("user_1")


total = total.join(
    features,
    how="left")

bt_columns = [
    "Unique people S-H",
    "Unique people O-S-H",
    "Outsiders S-H",
    "Outsiders O-S-H",
    "Average Duration S-H",
    "Average Duration O-S-H",
    "Average RSSI S-H",
    "Average RSSI O-S-H",
    "Interactions S-H",
    "Interactions O-S-H"
]

total[bt_columns] = total[bt_columns].fillna(-1)
total.to_excel("data_mining_bt.xlsx", index=False)
print(total)