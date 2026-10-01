import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

FILE = "Final_total_list.csv"
OUT = "clustericlustera"

PLOTS = f"{OUT}/plots"
DATA = f"{OUT}/data"

os.makedirs(PLOTS, exist_ok=True)
os.makedirs(DATA, exist_ok=True)

groups = {
    "digital_school": [
        "sms_sent_S-H",
        "sms_rec_S-H",
        "sms_unq_people S-H",
        "sms_convo_S-H",
        "calls_duration_S-H",
        "calls_made_S-H",
        "Calls Received S-H",
        "Unique People S-H"
    ],

    "digital_offschool": [
        "sms_sent_O-S-H",
        "sms_rec_O-S-H",
        "sms_unq_people O-S-H",
        "sms_convo_O-S-H",
        "Duration O-S-H",
        "Calls Made O-S-H",
        "Calls Received O-S-H",
        "Unique People O-S-H"
    ],

    "social_school": [
        "bt_interactions_S-H",
        "bt_avg_duration_S-H",
        "bt_unq_people_S-H",
        "bt_outside_interactions_S-H"
    ],

    "social_offschool": [
        "bt_interactions_O-S-H",
        "bt_avg_duration_O-S-H",
        "bt_unq_people_O-S-H",
        "bt_outside_interactions_O-S-H"
    ]
}

features = list(dict.fromkeys(sum(groups.values(), [])))

df = pd.read_csv(FILE)[["user", "gender"] + features].copy()

for col in features:
    df[col] = pd.to_numeric(
        df[col]
        .astype(str)
        .str.strip()
        .str.replace("−", "-", regex=False)
        .str.replace(",", ".", regex=False),
        errors="coerce"
    )

before = len(df)

df = df[
    ~df[features].eq(-1).any(axis=1)
].copy()

removed = before - len(df)


df[features] = df[features].fillna(
    df[features].median()
)

dimension_scores = pd.DataFrame(index=df.index)

for dimension, cols in groups.items():

    # Standardisera variablerna inom dimensionen.
    scaled = StandardScaler().fit_transform(df[cols])

    # Varje student får ett genomsnittligt standardiserat
    # aktivitetsvärde för dimensionen.
    dimension_scores[dimension] = scaled.mean(axis=1)


# Spara de fyra dimensionerna
dimension_scores.to_csv(
    f"{DATA}/four_activity_dimensions.csv",
    index=False
)


correlation = dimension_scores.corr()

correlation.to_csv(
    f"{DATA}/dimension_correlations.csv"
)

plt.figure(figsize=(7, 6))

plt.imshow(
    correlation,
    vmin=-1,
    vmax=1,
    aspect="auto"
)

plt.colorbar(label="Correlation")

plt.xticks(
    range(len(correlation.columns)),
    correlation.columns,
    rotation=45,
    ha="right"
)

plt.yticks(
    range(len(correlation.index)),
    correlation.index
)

for i in range(len(correlation)):
    for j in range(len(correlation.columns)):
        plt.text(
            j,
            i,
            f"{correlation.iloc[i, j]:.2f}",
            ha="center",
            va="center"
        )

plt.title("Relationship between activity dimensions")
plt.tight_layout()

plt.savefig(
    f"{PLOTS}/dimension_correlations.png",
    dpi=300
)

plt.close()


X = StandardScaler().fit_transform(
    dimension_scores
)

max_k = min(8, len(df) - 1)

silhouette_scores = {}

for k in range(2, max_k + 1):

    model = KMeans(
        n_clusters=k,
        random_state=50,
        n_init=20
    )

    labels = model.fit_predict(X)

    silhouette_scores[k] = silhouette_score(
        X,
        labels
    )


# Spara resultaten
silhouette_df = pd.DataFrame({
    "k": list(silhouette_scores.keys()),
    "silhouette": list(silhouette_scores.values())
})

silhouette_df.to_csv(
    f"{DATA}/silhouette_scores.csv",
    index=False
)


plt.figure(figsize=(7, 5))

plt.plot(
    list(silhouette_scores.keys()),
    list(silhouette_scores.values()),
    marker="o"
)

best_k = max(
    silhouette_scores,
    key=silhouette_scores.get
)

plt.axvline(
    best_k,
    linestyle="--",
    label=f"Selected k = {best_k}"
)

plt.xlabel("Number of clusters (k)")
plt.ylabel("Silhouette score")

plt.title("Selection of number of clusters")
plt.legend()

plt.tight_layout()

plt.savefig(
    f"{PLOTS}/silhouette_selection.png",
    dpi=300
)

plt.close()


final_model = KMeans(
    n_clusters=best_k,
    random_state=50,
    n_init=20
)

labels = final_model.fit_predict(X)

df["profile_cluster"] = labels + 1

profile_means = (
    dimension_scores
    .assign(profile_cluster=df["profile_cluster"].values)
    .groupby("profile_cluster")
    .mean()
)

profile_means.to_csv(
    f"{DATA}/profile_composition.csv"
)

plt.figure(
    figsize=(8, 5)
)

plt.imshow(
    profile_means,
    aspect="auto"
)

plt.colorbar(
    label="Standardised activity"
)

plt.xticks(
    range(len(profile_means.columns)),
    profile_means.columns,
    rotation=45,
    ha="right"
)

plt.yticks(
    range(len(profile_means.index)),
    [f"Profile {x}" for x in profile_means.index]
)

for i in range(len(profile_means)):
    for j in range(len(profile_means.columns)):

        plt.text(
            j,
            i,
            f"{profile_means.iloc[i, j]:.2f}",
            ha="center",
            va="center"
        )

plt.title("Activity profile composition")

plt.tight_layout()

plt.savefig(
    f"{PLOTS}/profile_heatmap.png",
    dpi=300
)

plt.close()

profile_sizes = (
    df["profile_cluster"]
    .value_counts()
    .sort_index()
)

profile_sizes.to_csv(
    f"{DATA}/profile_sizes.csv",
    header=["students"]
)


plt.figure(figsize=(7, 5))

plt.bar(
    profile_sizes.index.astype(str),
    profile_sizes.values
)

plt.xlabel("Activity profile")
plt.ylabel("Number of students")

plt.title("Number of students in each activity profile")

plt.tight_layout()

plt.savefig(
    f"{PLOTS}/profile_sizes.png",
    dpi=300
)

plt.close()

result = pd.concat(
    [
        df[["user", "gender", "profile_cluster"]].reset_index(drop=True),
        dimension_scores.reset_index(drop=True)
    ],
    axis=1
)

result.to_csv(
    f"{DATA}/student_activity_clusters.csv",
    index=False
)

method_summary = pd.DataFrame({
    "item": [
        "Number of students after removing -1",
        "Students removed because of -1",
        "Number of dimensions",
        "Clustering method",
        "Standardisation",
        "Random state",
        "Selected number of clusters",
        "Best silhouette score"
    ],

    "value": [
        len(df),
        removed,
        4,
        "KMeans",
        "StandardScaler",
        50,
        best_k,
        silhouette_scores[best_k]
    ]
})

method_summary.to_csv(
    f"{DATA}/method_summary.csv",
    index=False
)
print("Analysen är klar.")
print(f"Antal studenter: {len(df)}")
print(f"Valt antal kluster: {best_k}")
print(
    f"Silhouette score: "
    f"{silhouette_scores[best_k]:.3f}"
)
print(f"Resultat sparade i: {OUT}")