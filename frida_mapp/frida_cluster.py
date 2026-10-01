import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

FILE = "Final_total_list.csv"
OUT = "clustering_results"
PLOTS, DATA = f"{OUT}/plots", f"{OUT}/data"
os.makedirs(PLOTS, exist_ok=True)
os.makedirs(DATA, exist_ok=True)

groups = {
    "digital_school": [
        "fb_friends", "sms_sent_S-H", "sms_rec_S-H", "sms_unq_people S-H",
        "sms_convo_S-H", "calls_duration_S-H", "calls_made_S-H",
        "Calls Received S-H", "Unique People S-H"
    ],
    "digital_offschool": [
        "fb_friends", "sms_sent_O-S-H", "sms_rec_O-S-H", "sms_unq_people O-S-H",
        "sms_convo_O-S-H", "Duration O-S-H", "Calls Made O-S-H",
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

features = list(dict.fromkeys(sum(groups.values(), [])))
df = pd.read_csv(FILE)[["user", "gender"] + features].copy()

# Clean data
for c in features:
    df[c] = pd.to_numeric(
        df[c].astype(str).str.strip()
        .str.replace("−", "-", regex=False)
        .str.replace(",", ".", regex=False),
        errors="coerce"
    )

# Remove entire rows containing -1
df = df[~df[features].eq(-1).any(axis=1)].copy()

# Replace remaining missing values with median
df[features] = df[features].fillna(df[features].median())


def cluster(X):
    X = StandardScaler().fit_transform(X)
    scores = {}

    for k in range(2, min(10, len(X) - 1) + 1):
        labels = KMeans(
            n_clusters=k,
            random_state=50,
            n_init=20
        ).fit_predict(X)
        scores[k] = silhouette_score(X, labels)

    k = max(scores, key=scores.get)
    labels = KMeans(
        n_clusters=k,
        random_state=50,
        n_init=20
    ).fit_predict(X)

    return X, labels, k, scores


def gender_plot(data, group, order, filename, title):
    table = pd.crosstab(
        data[group], data.gender, normalize="columns"
    ).reindex(order).fillna(0) * 100

    table.to_csv(f"{DATA}/{filename.replace('.png', '.csv')}")

    x = np.arange(len(table.columns))
    width = .8 / len(order)

    plt.figure(figsize=(9, 5))
    for i, name in enumerate(table.index):
        bars = plt.bar(
            x + (i - (len(order)-1)/2) * width,
            table.loc[name], width, label=name
        )
        for bar, value in zip(bars, table.loc[name]):
            if value > 3:
                plt.text(
                    bar.get_x() + bar.get_width()/2,
                    value + 1, f"{value:.0f}%",
                    ha="center", fontsize=8
                )

    plt.xticks(x, table.columns)
    plt.xlabel("Gender")
    plt.ylabel("Percentage (%)")
    plt.ylim(0, 100)
    plt.title(title)
    plt.legend(title="Activity group")
    plt.tight_layout()
    plt.savefig(f"{PLOTS}/{filename}", dpi=300)
    plt.close()


results = []

for name, cols in groups.items():

    X, labels, k, scores = cluster(df[cols])

    means = {
        c: X[labels == c].mean()
        for c in np.unique(labels)
    }

    ordered = sorted(means, key=means.get)
    names = {
        2: ["Low", "High"],
        3: ["Low", "Medium", "High"],
        4: ["Very Low", "Low", "High", "Very High"],
        5: ["Very Low", "Low", "Medium", "High", "Very High"]
    }.get(k, [f"Group {i+1}" for i in range(k)])

    mapping = dict(zip(ordered, names))
    group_col = f"{name}_group"
    df[group_col] = pd.Series(labels, index=df.index).map(mapping)
    ordered_names = [mapping[c] for c in ordered]

    # K selection
    plt.figure(figsize=(7, 4))
    plt.plot(list(scores), list(scores.values()), marker="o")
    plt.xlabel("K")
    plt.ylabel("Silhouette score")
    plt.title(f"K selection - {name.replace('_', ' ').title()}")
    plt.xticks(list(scores))
    plt.grid(alpha=.3)
    plt.tight_layout()
    plt.savefig(f"{PLOTS}/{name}_k_selection.png", dpi=300)
    plt.close()

    # Group size
    counts = df[group_col].value_counts().reindex(ordered_names).fillna(0)

    plt.figure(figsize=(7, 4))
    bars = plt.bar([f"Group {i+1}" for i in range(k)], counts)
    for bar, value in zip(bars, counts):
        plt.text(
            bar.get_x() + bar.get_width()/2,
            bar.get_height() + 1,
            str(int(value)), ha="center"
        )
    plt.xlabel("Activity group")
    plt.ylabel("Number of students")
    plt.title(f"Students by activity group - {name.replace('_', ' ').title()}")
    plt.tight_layout()
    plt.savefig(f"{PLOTS}/{name}_group_size.png", dpi=300)
    plt.close()

    # Activity level
    activity = [means[c] for c in ordered]

    plt.figure(figsize=(7, 4))
    bars = plt.bar([f"Group {i+1}" for i in range(k)], activity)
    for bar, value in zip(bars, activity):
        plt.text(
            bar.get_x() + bar.get_width()/2,
            value, f"{value:.2f}",
            ha="center", va="bottom"
        )
    plt.xlabel("Activity group")
    plt.ylabel("Mean standardized activity")
    plt.title(f"Activity level - {name.replace('_', ' ').title()}")
    plt.tight_layout()
    plt.savefig(f"{PLOTS}/{name}_activity_level.png", dpi=300)
    plt.close()

    gender_plot(
        df, group_col, ordered_names,
        f"{name}_gender_distribution.png",
        f"Gender distribution - {name.replace('_', ' ').title()}"
    )

    results.append({
        "dimension": name,
        "best_k": k,
        "silhouette": scores[k]
    })


# Activity profiles
group_cols = [f"{x}_group" for x in groups]

codes = {
    "Very Low": "VL", "Low": "L", "Medium": "M",
    "High": "H", "Very High": "VH"
}

df["activity_profile"] = df[group_cols].replace(codes).agg("-".join, axis=1)

numeric = {
    "Very Low": 0, "Low": 1, "Medium": 2,
    "High": 3, "Very High": 4
}

X, profile_labels, profile_k, profile_scores = cluster(
    df[group_cols].replace(numeric)
)

df["profile_cluster"] = profile_labels

profile_names = {
    c: f"Group {i+1}"
    for i, c in enumerate(sorted(np.unique(profile_labels)))
}

df["profile_group"] = df.profile_cluster.map(profile_names)
profile_order = list(profile_names.values())


# Final profile plots
plt.figure(figsize=(7, 4))
plt.plot(list(profile_scores), list(profile_scores.values()), marker="o")
plt.xlabel("K")
plt.ylabel("Silhouette score")
plt.title("K selection - Final activity profiles")
plt.xticks(list(profile_scores))
plt.grid(alpha=.3)
plt.tight_layout()
plt.savefig(f"{PLOTS}/profile_k_selection.png", dpi=300)
plt.close()

counts = df.profile_group.value_counts().reindex(profile_order).fillna(0)

plt.figure(figsize=(7, 4))
bars = plt.bar(profile_order, counts)
for bar, value in zip(bars, counts):
    plt.text(
        bar.get_x() + bar.get_width()/2,
        bar.get_height() + 1,
        str(int(value)), ha="center"
    )
plt.xlabel("Final profile")
plt.ylabel("Number of students")
plt.title("Students by final activity profile")
plt.tight_layout()
plt.savefig(f"{PLOTS}/profile_cluster_size.png", dpi=300)
plt.close()


# Profile composition
profile_means = (
    df.groupby("profile_group")[group_cols]
    .agg(lambda x: x.map(numeric).mean())
    .reindex(profile_order)
)

profile_means.to_csv(f"{DATA}/profile_cluster_composition.csv")

plt.figure(figsize=(9, 6))
plt.imshow(profile_means, aspect="auto")
plt.colorbar(label="Mean activity level")
plt.xticks(
    range(4),
    ["Digital School", "Digital Off-School",
     "Social School", "Social Off-School"],
    rotation=25, ha="right"
)
plt.yticks(range(len(profile_means)), profile_order)
plt.xlabel("Activity dimension")
plt.ylabel("Final profile")
plt.title("Activity composition of final profiles")

for i in range(len(profile_means)):
    for j in range(4):
        plt.text(
            j, i, f"{profile_means.iloc[i, j]:.1f}",
            ha="center", va="center"
        )

plt.tight_layout()
plt.savefig(f"{PLOTS}/profile_heatmap.png", dpi=300)
plt.close()


# Activity profile frequency
freq = df.activity_profile.value_counts().sort_index()
freq.to_csv(f"{DATA}/activity_profile_frequency.csv", header=["students"])

plt.figure(figsize=(14, 6))
bars = plt.bar(freq.index, freq)
for bar, value in zip(bars, freq):
    plt.text(
        bar.get_x() + bar.get_width()/2,
        bar.get_height() + 1,
        str(int(value)), ha="center", fontsize=8
    )
plt.xlabel("Activity pattern")
plt.ylabel("Number of students")
plt.title("Digital and social activity patterns")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig(
    os.path.join(PLOTS, "activity_profile_frequency.png"),
    dpi=300
)
plt.close()


gender_plot(
    df, "profile_group", profile_order,
    "profile_cluster_gender.png",
    "Gender distribution by final profile"
)


# Save data
pd.DataFrame(results).to_csv(
    f"{DATA}/best_k_results.csv", index=False
)

df.to_csv(
    f"{DATA}/student_activity_clusters.csv", index=False
)


# Print results
for r in results:
    print(
        f"{r['dimension']:25s} "
        f"K={r['best_k']} "
        f"silhouette={r['silhouette']:.3f}"
    )

print(
    f"\nFinal profiles: K={profile_k}, "
    f"silhouette={profile_scores[profile_k]:.3f}"
)

print("\nExample profiles:")
print(df[["user", "activity_profile", "profile_group"]].head(10))

corr = df[group_cols].replace(numeric).corr()

sns.heatmap(corr, annot=True, vmin=-1, vmax=1, cmap="coolwarm")
plt.title("Correlation between activity groups")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS, "group_correlation_heatmap.png"), dpi=300)
plt.show()

print(f"\nResults saved in: {OUT}/")