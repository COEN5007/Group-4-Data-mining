import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# 1. Load
df = pd.read_csv("total_list.csv")

# 2. Features
groups = {
    "digital_school": [
        "fb_friends", "sms_sent_S-H", "sms_rec_S-H",
        "sms_unq_people S-H", "sms_convo_S-H",
        "calls_duration_S-H", "calls_made_S-H",
        "Calls Received S-H", "Unique People S-H"
    ],

    "digital_offschool": [
        "fb_friends", "sms_sent_O-S-H", "sms_rec_O-S-H",
        "sms_unq_people O-S-H", "sms_convo_O-S-H",
        "Duration O-S-H", "Calls Made O-S-H",
        "Calls Received O-S-H", "Unique People O-S-H"
    ],

    "social_school": [
        "bt_interactions_S-H", "bt_avg_duration_S-H",
        "bt_unq_people_S-H", "bt_outside_interactions_S-H"
    ],

    "social_offschool": [
        "bt_interactions_O-S-H", "bt_avg_duration_O-S-H",
        "bt_unq_people_O-S-H", "bt_outside_interactions_O-S-H"
    ]
}

features = sum(groups.values(), [])
features = list(dict.fromkeys(features))

df = df[["user", "gender"] + features].copy()

# 3. Clean numeric data
for col in features:
    df[col] = (
        df[col].astype(str)
        .str.replace("−", "-", regex=False)
        .str.replace(",", ".", regex=False)
    )
    df[col] = pd.to_numeric(df[col], errors="coerce")
    df.loc[df[col] == -1, col] = pd.NA
    df[col] = df[col].fillna(df[col].median())

# 4. Create four scores
for name, cols in groups.items():
    X = StandardScaler().fit_transform(df[cols])
    df[name + "_score"] = X.mean(axis=1)

# 5. Correlation map
score_cols = [x + "_score" for x in groups]

sns.heatmap(
    df[score_cols].corr(),
    annot=True,
    vmin=-1,
    vmax=1,
    cmap="coolwarm"
)
plt.title("Correlation between activity scores")
plt.show()

# 6. H/L profiles
for name in groups:
    df[name + "_level"] = (
        df[name + "_score"] >= 0
    ).map({True: "H", False: "L"})

df["activity_pattern"] = (
    df[[x + "_level" for x in groups]]
    .agg("".join, axis=1)
)

# 7. Cluster the four-dimensional user profiles
X_profile = StandardScaler().fit_transform(df[score_cols])

results = []

for k in range(2, 7):
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = model.fit_predict(X_profile)
    results.append((k, silhouette_score(X_profile, labels)))

print(results)

best_k = max(results, key=lambda x: x[1])[0]

model = KMeans(n_clusters=best_k, random_state=42, n_init=10)
df["profile_cluster"] = model.fit_predict(X_profile)

print("Best k:", best_k)