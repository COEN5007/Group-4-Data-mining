import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

FILE = "total_list.csv"
RANDOM_STATE = 50

OUT = "clustering_results"
PLOTS = os.path.join(OUT, "plots")
DATA = os.path.join(OUT, "data")

os.makedirs(PLOTS, exist_ok=True)
os.makedirs(DATA, exist_ok=True)


# ============================================================
# FEATURE GROUPS
# ============================================================

groups = {
    "digital_school": [
        "fb_friends",
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
        "fb_friends",
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


# ============================================================
# LOAD + CLEAN
# ============================================================

features = list(dict.fromkeys(sum(groups.values(), [])))

df = pd.read_csv(FILE)[
    ["user", "gender"] + features
].copy()

for col in features:

    df[col] = (
        df[col]
        .astype(str)
        .str.strip()
        .str.replace("−", "-", regex=False)
        .str.replace(",", ".", regex=False)
    )

    df[col] = pd.to_numeric(
        df[col],
        errors="coerce"
    )

    df.loc[df[col] == -1, col] = np.nan

df = df.dropna(
    subset=features,
    how="all"
).copy()

for col in features:
    df[col] = df[col].fillna(
        df[col].median()
    )


# ============================================================
# K-MEANS
# ============================================================

def cluster_data(X, max_k=10):

    X_scaled = StandardScaler().fit_transform(X)
    scores = {}

    for k in range(
        2,
        min(max_k, len(X_scaled) - 1) + 1
    ):

        model = KMeans(
            n_clusters=k,
            random_state=RANDOM_STATE,
            n_init=20
        )

        labels = model.fit_predict(X_scaled)

        scores[k] = silhouette_score(
            X_scaled,
            labels
        )

    best_k = max(
        scores,
        key=scores.get
    )

    model = KMeans(
        n_clusters=best_k,
        random_state=RANDOM_STATE,
        n_init=20
    )

    labels = model.fit_predict(X_scaled)

    return (
        X_scaled,
        labels,
        best_k,
        scores
    )


# ============================================================
# ACTIVITY NAMES
# ============================================================

def activity_names(k):

    names = {
        2: ["Low", "High"],
        3: ["Low", "Medium", "High"],
        4: ["Very Low", "Low", "High", "Very High"],
        5: ["Very Low", "Low", "Medium", "High", "Very High"]
    }

    return names.get(
        k,
        [f"Group {i+1}" for i in range(k)]
    )


# ============================================================
# GENDER DISTRIBUTION
# ============================================================
# Shows:
# "Of all people of this gender, what percentage
# belongs to each activity group?"
# Therefore each gender sums to 100%.
# ============================================================

def gender_plot(
    data,
    group_col,
    order,
    filename,
    title
):

    table = pd.crosstab(
        data[group_col],
        data["gender"],
        normalize="columns"
    ) * 100

    table = table.reindex(
        order
    ).fillna(0)

    table.to_csv(
        os.path.join(
            DATA,
            filename.replace(".png", ".csv")
        )
    )

    genders = list(table.columns)
    x = np.arange(len(genders))
    width = 0.8 / len(order)

    plt.figure(figsize=(9, 5))

    for i, group in enumerate(table.index):

        values = table.loc[group].values

        bars = plt.bar(
            x + (
                i - (len(order)-1)/2
            ) * width,
            values,
            width,
            label=str(group)
        )

        for bar, value in zip(
            bars,
            values
        ):

            if value > 3:

                plt.text(
                    bar.get_x()
                    + bar.get_width()/2,
                    value + 1,
                    f"{value:.0f}%",
                    ha="center",
                    fontsize=8
                )

    plt.xticks(
        x,
        genders
    )

    plt.xlabel("Gender")
    plt.ylabel("Percentage (%)")
    plt.ylim(0, 100)
    plt.title(title)
    plt.legend(title="Activity group")

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            PLOTS,
            filename
        ),
        dpi=300
    )

    plt.close()


# ============================================================
# CLUSTER THE FOUR DIMENSIONS
# ============================================================

results = []

for name, cols in groups.items():

    X, labels, best_k, scores = cluster_data(
        df[cols]
    )

    # Mean standardized activity per cluster
    means = {
        c: X[labels == c].mean()
        for c in np.unique(labels)
    }

    # Order clusters from low to high activity
    ordered = sorted(
        means,
        key=means.get
    )

    names = activity_names(best_k)

    cluster_names = {
        c: names[i]
        for i, c in enumerate(ordered)
    }

    group_col = f"{name}_group"

    df[group_col] = [
        cluster_names[c]
        for c in labels
    ]

    ordered_names = [
        cluster_names[c]
        for c in ordered
    ]

    # --------------------------------------------------------
    # K SELECTION
    # --------------------------------------------------------

    plt.figure(figsize=(7, 4))

    plt.plot(
        list(scores.keys()),
        list(scores.values()),
        marker="o"
    )

    plt.xlabel("K")
    plt.ylabel("Silhouette score")
    plt.title(
        f"K selection - "
        f"{name.replace('_', ' ').title()}"
    )

    plt.xticks(
        list(scores.keys())
    )

    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        os.path.join(
            PLOTS,
            f"{name}_k_selection.png"
        ),
        dpi=300
    )

    plt.close()


    # --------------------------------------------------------
    # GROUP SIZE
    # --------------------------------------------------------

    counts = (
        df[group_col]
        .value_counts()
        .reindex(ordered_names)
        .fillna(0)
    )

    group_numbers = [
        f"Group {i+1}"
        for i in range(len(counts))
    ]

    plt.figure(figsize=(7, 4))

    bars = plt.bar(
        group_numbers,
        counts.values
    )

    # Number directly on bars
    for bar, value in zip(
        bars,
        counts.values
    ):

        plt.text(
            bar.get_x()
            + bar.get_width()/2,
            bar.get_height() + 1,
            str(int(value)),
            ha="center",
            va="bottom",
            fontsize=10
        )

    plt.xlabel("Activity group")
    plt.ylabel("Number of students")

    plt.title(
        f"Students by activity group - "
        f"{name.replace('_', ' ').title()}"
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            PLOTS,
            f"{name}_group_size.png"
        ),
        dpi=300
    )

    plt.close()


    # --------------------------------------------------------
    # ACTIVITY LEVEL
    # --------------------------------------------------------

    activity = [
        means[c]
        for c in ordered
    ]

    plt.figure(figsize=(7, 4))

    bars = plt.bar(
        group_numbers,
        activity
    )

    for bar, value in zip(
        bars,
        activity
    ):

        plt.text(
            bar.get_x()
            + bar.get_width()/2,
            value,
            f"{value:.2f}",
            ha="center",
            va="bottom"
        )

    plt.xlabel("Activity group")
    plt.ylabel("Mean standardized activity")

    plt.title(
        f"Activity level - "
        f"{name.replace('_', ' ').title()}"
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            PLOTS,
            f"{name}_activity_level.png"
        ),
        dpi=300
    )

    plt.close()


    # --------------------------------------------------------
    # GENDER
    # --------------------------------------------------------

    gender_plot(
        df,
        group_col,
        ordered_names,
        f"{name}_gender_distribution.png",
        f"Gender distribution - "
        f"{name.replace('_', ' ').title()}"
    )


    results.append({
        "dimension": name,
        "best_k": best_k,
        "silhouette": scores[best_k]
    })


# ============================================================
# SAVE BEST K
# ============================================================

pd.DataFrame(results).to_csv(
    os.path.join(
        DATA,
        "best_k_results.csv"
    ),
    index=False
)


# ============================================================
# ACTIVITY PROFILE
# ============================================================

group_cols = [
    "digital_school_group",
    "digital_offschool_group",
    "social_school_group",
    "social_offschool_group"
]

# Short codes only for the activity profile
codes = {
    "Very Low": "VL",
    "Low": "L",
    "Medium": "M",
    "High": "H",
    "Very High": "VH"
}

for col in group_cols:

    df[col + "_code"] = df[
        col
    ].map(codes)

profile_cols = [
    col + "_code"
    for col in group_cols
]

df["activity_profile"] = (
    df[profile_cols]
    .agg("-".join, axis=1)
)


# ============================================================
# FINAL PROFILE K-MEANS
# ============================================================

numeric = {
    "Very Low": 0,
    "Low": 1,
    "Medium": 2,
    "High": 3,
    "Very High": 4
}

profile_numeric = (
    df[group_cols]
    .replace(numeric)
)

(
    X_profile,
    profile_labels,
    profile_k,
    profile_scores
) = cluster_data(
    profile_numeric
)

df["profile_cluster"] = profile_labels


# Use Group 1, Group 2, etc. for final clusters
profile_names = {
    c: f"Group {i+1}"
    for i, c in enumerate(
        sorted(
            np.unique(profile_labels)
        )
    )
}

df["profile_group"] = (
    df["profile_cluster"]
    .map(profile_names)
)


# ============================================================
# FINAL PROFILE K SELECTION
# ============================================================

plt.figure(figsize=(7, 4))

plt.plot(
    list(profile_scores.keys()),
    list(profile_scores.values()),
    marker="o"
)

plt.xlabel("K")
plt.ylabel("Silhouette score")
plt.title("K selection - Final activity profiles")

plt.xticks(
    list(profile_scores.keys())
)

plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    os.path.join(
        PLOTS,
        "profile_k_selection.png"
    ),
    dpi=300
)

plt.close()


# ============================================================
# FINAL PROFILE SIZE
# ============================================================

profile_order = list(
    profile_names.values()
)

counts = (
    df["profile_group"]
    .value_counts()
    .reindex(profile_order)
    .fillna(0)
)

plt.figure(figsize=(7, 4))

bars = plt.bar(
    profile_order,
    counts.values
)

for bar, value in zip(
    bars,
    counts.values
):

    plt.text(
        bar.get_x()
        + bar.get_width()/2,
        bar.get_height() + 1,
        str(int(value)),
        ha="center",
        va="bottom"
    )

plt.xlabel("Final profile")
plt.ylabel("Number of students")
plt.title("Students by final activity profile")

plt.tight_layout()

plt.savefig(
    os.path.join(
        PLOTS,
        "profile_cluster_size.png"
    ),
    dpi=300
)

plt.close()


# ============================================================
# PROFILE COMPOSITION
# ============================================================

profile_means = (
    df.groupby("profile_group")[group_cols]
    .agg(
        lambda x:
        x.map(numeric).mean()
    )
    .reindex(profile_order)
)

profile_means.to_csv(
    os.path.join(
        DATA,
        "profile_cluster_composition.csv"
    )
)


# ============================================================
# PROFILE HEATMAP
# ============================================================

plt.figure(figsize=(9, 6))

plt.imshow(
    profile_means,
    aspect="auto"
)

plt.colorbar(
    label="Mean activity level"
)

plt.xticks(
    range(4),
    [
        "Digital School",
        "Digital Off-School",
        "Social School",
        "Social Off-School"
    ],
    rotation=25,
    ha="right"
)

plt.yticks(
    range(len(profile_means)),
    profile_order
)

plt.xlabel("Activity dimension")
plt.ylabel("Final profile")
plt.title(
    "Activity composition of final profiles"
)

for i in range(len(profile_means)):

    for j in range(4):

        plt.text(
            j,
            i,
            f"{profile_means.iloc[i, j]:.1f}",
            ha="center",
            va="center"
        )

plt.tight_layout()

plt.savefig(
    os.path.join(
        PLOTS,
        "profile_heatmap.png"
    ),
    dpi=300
)

plt.close()


# ACTIVITY PROFILE FREQUENCY

freq = (
    df["activity_profile"]
    .value_counts()
    .sort_index()
)

# Save frequency table
freq.to_csv(
    os.path.join(DATA, "activity_profile_frequency.csv"),
    header=["students"]
)

# Plot
plt.figure(figsize=(14, 6))

bars = plt.bar(
    freq.index,
    freq.values
)

# Add number above each bar
for bar, value in zip(bars, freq.values):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 1,
        str(int(value)),
        ha="center",
        va="bottom",
        fontsize=8
    )

plt.xlabel("Activity pattern")
plt.ylabel("Number of students")
plt.title("Digital and social activity patterns")

plt.xticks(
    rotation=45,
    ha="right"
)

plt.tight_layout()

plt.savefig(
    os.path.join(PLOTS, "activity_profile_frequency.png"),
    dpi=300
)

plt.close()

# ============================================================
# FINAL PROFILE GENDER
# ============================================================

gender_plot(
    df,
    "profile_group",
    profile_order,
    "profile_cluster_gender.png",
    "Gender distribution by final profile"
)


# ============================================================
# SAVE FINAL DATA
# ============================================================

df.to_csv(
    os.path.join(
        DATA,
        "student_activity_clusters.csv"
    ),
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

print("\n================================")
print("CLUSTERING COMPLETE")
print("================================")

for r in results:

    print(
        f"{r['dimension']:25s} "
        f"K={r['best_k']}  "
        f"silhouette={r['silhouette']:.3f}"
    )

print(
    f"\nFinal profiles: "
    f"K={profile_k}, "
    f"silhouette="
    f"{profile_scores[profile_k]:.3f}"
)

print("\nExample profiles:")

print(
    df[
        [
            "user",
            "activity_profile",
            "profile_group"
        ]
    ].head(10)
)

print(
    f"\nResults saved in: {OUT}/"
)