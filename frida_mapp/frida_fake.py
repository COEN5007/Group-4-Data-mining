import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from scipy.stats import chi2_contingency

totallist = "total_list.csv"
folder = "clustering_plots"
os.makedirs(folder, exist_ok=True)
df_original = pd.read_csv(totallist)

feature_groups = {

    "Digital School": [
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

    "Digital Off-School": [
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

    "Social School": [
        "bt_interactions_S-H",
        "bt_avg_duration_S-H",
        "bt_unq_people_S-H",
        "bt_outside_interactions_S-H"
    ],

    "Social Off-School": [
        "bt_interactions_O-S-H",
        "bt_avg_duration_O-S-H",
        "bt_unq_people_O-S-H",
        "bt_outside_interactions_O-S-H"
    ]
}


analysis_features = list(dict.fromkeys(
    sum(feature_groups.values(), [])
)) #the features that are prepared for analysis

columns_to_keep = ["user", "gender"] + analysis_features #save the user and gender for later

df = df_original[columns_to_keep].copy()


for col in analysis_features:

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

    #-1 represents missing data
    df.loc[df[col] == -1, col] = np.nan

#Replace remaining missing values with median, the nan values
for col in analysis_features:
    df[col] = df[col].fillna(
        df[col].median()
    )

#Helping functions for later
def dbscan_silhouette(X, labels):

    mask = labels != -1

    X_clean = X[mask]
    labels_clean = labels[mask]

    if len(set(labels_clean)) < 2:
        return np.nan

    return silhouette_score(
        X_clean,
        labels_clean
    )


def filename(group, suffix):

    return (
        group.lower()
        .replace(" ", "_")
        .replace("-", "")
        + suffix
    )

#for k-means
def run_kmeans(X, group):

    results = []

    for k in range(
        2,
        min(10, len(X) - 1) + 1
    ): #test different k, from 2 to 10

        model = KMeans(
            n_clusters=k,
            random_state=42,
            n_init=20
        )

        labels = model.fit_predict(X)

        results.append({
            "method": "K-Means",
            "k": k,
            "silhouette": silhouette_score(
                X,
                labels
            )
        })

    results = pd.DataFrame(results)

    best = results.loc[
        results.silhouette.idxmax()
    ]

    best_k = int(best.k)

    #final K-Means
    labels = KMeans(
        n_clusters=best_k,
        random_state=42,
        n_init=20
    ).fit_predict(X)

    print(
        f"{group} - K-Means: "
        f"{best_k} clusters"
    )

    plot_silhouette(
        results,
        group,
        "K-Means",
        filename(
            group,
            "_kmeans_silhouette.png"
        )
    )

    return results, labels, best_k

#Kind of same, but for hierarchial, with same testing for k
def run_hierarchical(X, group):

    results = []

    for k in range(
        2,
        min(10, len(X) - 1) + 1
    ):

        labels = AgglomerativeClustering(
            n_clusters=k
        ).fit_predict(X)

        results.append({
            "method": "Hierarchical",
            "k": k,
            "silhouette": silhouette_score(
                X,
                labels
            )
        })

    results = pd.DataFrame(results)

    best = results.loc[
        results.silhouette.idxmax()
    ]

    best_k = int(best.k)

    # Final hierarchical clustering
    labels = AgglomerativeClustering(
        n_clusters=best_k
    ).fit_predict(X)

    print(
        f"{group} - Hierarchical: "
        f"{best_k} clusters"
    )

    plot_silhouette(
        results,
        group,
        "Hierarchical",
        filename(
            group,
            "_hierarchical_silhouette.png"
        )
    )

    return results, labels, best_k


#same for dbscan, but with differents eps to decide distance
def run_dbscan(X, group):

    results = []

    for eps in np.arange(
        0.2,
        3.1,
        0.1
    ):

        for min_samples in range(
            3,
            16
        ):

            labels = DBSCAN(
                eps=eps,
                min_samples=min_samples
            ).fit_predict(X)

            n_clusters = len(
                set(labels) - {-1}
            )

            if n_clusters < 2:
                continue

            score = dbscan_silhouette(
                X,
                labels
            )

            if np.isnan(score):
                continue

            results.append({
                "method": "DBSCAN",
                "eps": round(eps, 2),
                "min_samples": min_samples,
                "n_clusters": n_clusters,
                "silhouette": score
            })

    results = pd.DataFrame(results)

    best = results.loc[
        results.silhouette.idxmax()
    ]

    eps = float(best.eps)

    min_samples = int(
        best.min_samples
    )

    #Final DBSCAN with best eps
    labels = DBSCAN(
        eps=eps,
        min_samples=min_samples
    ).fit_predict(X)

    n_clusters = len(
        set(labels) - {-1}
    )

    n_noise = np.sum(
        labels == -1
    )

    print(
        f"{group} - DBSCAN: "
        f"{n_clusters} clusters, "
        f"{n_noise} noise points "
        f"(eps={eps}, "
        f"min_samples={min_samples})"
    )

    # Plot best DBSCAN parameters
    top = results.nlargest(
        20,
        "silhouette"
    )

    plt.figure(
        figsize=(10, 6)
    )

    plt.scatter(
        top.eps,
        top.min_samples,
        s=80,
        c=top.silhouette,
        cmap="viridis"
    )

    plt.colorbar(
        label="Silhouette score"
    )

    plt.xlabel("eps")
    plt.ylabel("min_samples")

    plt.title(
        f"DBSCAN parameter search - {group}"
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            folder,
            filename(
                group,
                "_dbscan_parameters.png"
            )
        ),
        dpi=300
    )

    plt.show()

    return (
        results,
        labels,
        (eps, min_samples)
    )

#for the silhouette plots
def plot_silhouette(
    results,
    group,
    method,
    file
):

    plt.figure(
        figsize=(8, 5)
    )

    plt.plot(
        results["k"],
        results["silhouette"],
        marker="o"
    )

    plt.xlabel(
        "Number of clusters"
    )

    plt.ylabel(
        "Silhouette score"
    )

    plt.title(
        f"{method} - {group}"
    )

    plt.grid(
        True,
        alpha=0.3
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            folder,
            file
        ),
        dpi=300
    )

    plt.show()

#Running all the clusters
all_results = []

best_cluster_labels = {}


for group, features in feature_groups.items(): #on all features choosen

    X = StandardScaler().fit_transform(
        df[features]
    )

    #k-means, on best result

    km_results, km_labels, km_k = run_kmeans(
        X,
        group
    )

    all_results.append(
        km_results.assign(
            group=group
        )
    )

    best_cluster_labels[
        (group, "K-Means")
    ] = km_labels


    #Hierarchial, on best result

    hc_results, hc_labels, hc_k = run_hierarchical(
        X,
        group
    )

    all_results.append(
        hc_results.assign(
            group=group
        )
    )

    best_cluster_labels[
        (group, "Hierarchical")
    ] = hc_labels


    #db-scan, on best result

    db_results, db_labels, db_params = run_dbscan(
        X,
        group
    )

    if db_params is not None:

        all_results.append(
            db_results.assign(
                group=group
            )
        )

        best_cluster_labels[
            (group, "DBSCAN")
        ] = db_labels


#Combining results
clustering_results = pd.concat(
    all_results,
    ignore_index=True
)


clustering_results.to_csv(
    "clustering_summary.csv",
    index=False
)


#all best results
best_results_df = (
    clustering_results
    .loc[
        clustering_results
        .groupby(
            ["group", "method"]
        )["silhouette"]
        .idxmax()
    ]
    .reset_index(drop=True)
)


print("\n")
print("BEST CLUSTERING RESULTS")


print(
    best_results_df[
        [
            "group",
            "method",
            "silhouette"
        ]
    ].to_string(
        index=False
    )
)


#plots and so and so
for group, features in feature_groups.items():

    X = StandardScaler().fit_transform(
        df[features]
    )

    X_pca = PCA(
        n_components=2
    ).fit_transform(X)

    #PCA currently visualizes K-Means
    labels = best_cluster_labels[
        (group, "K-Means")
    ]

    plt.figure(
        figsize=(8, 6)
    )

    scatter = plt.scatter(
        X_pca[:, 0],
        X_pca[:, 1],
        c=labels,
        cmap="tab10",
        alpha=0.7
    )

    plt.xlabel("PC1")
    plt.ylabel("PC2")

    plt.title(
        f"PCA - K-Means - {group}"
    )

    plt.colorbar(
        scatter,
        label="Cluster"
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            folder,
            filename(
                group,
                "_pca.png"
            )
        ),
        dpi=300
    )

    plt.show()


#moving on to second clustering
score_columns = []


for group, features in feature_groups.items():

    score_name = (
        group.lower()
        .replace(" ", "_")
        + "_score"
    )

    df[score_name] = (
        StandardScaler()
        .fit_transform(
            df[features]
        )
        .mean(axis=1)
    )

    score_columns.append(
        score_name
    )

#score correlation, showing the map
score_correlations = (
    df[score_columns]
    .corr()
)

print("\n")
print("Score correlations")

print(
    score_correlations.round(3)
)


score_correlations.to_csv(
    "four_dimension_correlations.csv"
)


plt.figure(
    figsize=(8, 6)
)

sns.heatmap(
    score_correlations,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    center=0
)

plt.title(
    "Correlation between four activity scores"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        folder,
        "four_dimension_correlation_heatmap.png"
    ),
    dpi=300
)

plt.show()


#User profile clustering
X_profile = StandardScaler().fit_transform(
    df[score_columns]
)

profile_results = []


for k in range(2, 9):

    labels = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=20
    ).fit_predict(
        X_profile
    )

    profile_results.append({
        "k": k,
        "silhouette": silhouette_score(
            X_profile,
            labels
        )
    })


profile_results = pd.DataFrame(
    profile_results
)


#select best k of profiles
best_profile_k = int(
    profile_results.loc[
        profile_results.silhouette.idxmax(),
        "k"
    ]
)


#Create the final profiles
df["user_profile_cluster"] = KMeans(
    n_clusters=best_profile_k,
    random_state=42,
    n_init=20
).fit_predict(
    X_profile
)


print("\n")
print("User profile clusters")

print(
    f"Number of profiles: {best_profile_k}"
)


#User profile table for visalization
profile_summary = (
    df.groupby(
        "user_profile_cluster"
    )[score_columns]
    .mean()
)


profile_counts = (
    df["user_profile_cluster"]
    .value_counts()
    .sort_index()
)


profile_percentages = (
    profile_counts /
    len(df) *
    100
)


profile_table = profile_summary.copy()


profile_table.insert(
    0,
    "Number of students",
    profile_counts
)


profile_table.insert(
    1,
    "Percentage",
    profile_percentages.round(2)
)


#Describing the user profiles
def describe_profile(row):

    high = row[score_columns] >= 0

    digital_school = (
        high["digital_school_score"]
    )

    digital_off = (
        high["digital_off-school_score"]
    )

    social_school = (
        high["social_school_score"]
    )

    social_off = (
        high["social_off-school_score"]
    )


    if (
        digital_school
        and digital_off
        and social_school
        and social_off
    ):
        return "High in all dimensions"


    if (
        not digital_school
        and not digital_off
        and not social_school
        and not social_off
    ):
        return "Low in all dimensions"


    if (
        digital_school
        and digital_off
        and not social_school
        and not social_off
    ):
        return "High digital, low social"


    if (
        not digital_school
        and not digital_off
        and social_school
        and social_off
    ):
        return "Low digital, high social"


    if (
        digital_school
        and digital_off
    ):
        return "High digital activity"


    if (
        social_school
        and social_off
    ):
        return "High social activity"


    return "Mixed activity"


profile_table[
    "Profile description"
] = profile_table.apply(
    describe_profile,
    axis=1
)


print("\n")
print("User profile table")

print(
    profile_table.round(2)
    .to_string()
)


profile_table.to_csv(
    "user_profile_table.csv"
)


#Activity patterns
level_columns = []

for score in score_columns:

    level = score.replace(
        "_score",
        "_level"
    )

    df[level] = np.where(
        df[score] >= 0,
        "High",
        "Low"
    )

    level_columns.append(
        level
    )


df["activity_pattern"] = (
    df[level_columns]
    .replace({
        "High": "H",
        "Low": "L"
    })
    .agg(
        "".join,
        axis=1
    )
)


df["number_of_high_dimensions"] = (
    df[level_columns]
    .eq("High")
    .sum(axis=1)
)


print("\n")
print("Patterns for the activity")

print(
    df.activity_pattern
    .value_counts()
    .sort_index()
)

#Analysing the genders in the clusters
gender_summary = pd.DataFrame({

    "count":
        df.gender
        .value_counts()
        .sort_index(),

    "percentage":
        (
            df.gender
            .value_counts(
                normalize=True
            )
            .sort_index()
            * 100
        )
})


gender_results = []


for group, level in zip(
    feature_groups,
    level_columns
):

    counts = pd.crosstab(
        df.gender,
        df[level]
    )

    percentages = (
        pd.crosstab(
            df.gender,
            df[level],
            normalize="index"
        )
        * 100
    )

    chi2, p, _, _ = (
        chi2_contingency(
            counts
        )
    )

    gender_results.append({

        "dimension": group,

        "chi_square": chi2,

        "p_value": p,
    })


    print("\n")
    print(group)

    print(
        "Counts:\n",
        counts
    )

    print(
        "Percentages:\n",
        percentages.round(2)
    )

    print(
        f"Chi-square={chi2:.4f}, "
        f"p={p:.6f}, "
    )


gender_results_df = pd.DataFrame(
    gender_results
)


gender_pattern_table = (
    pd.crosstab(
        df.gender,
        df.activity_pattern,
        normalize="index"
    )
    * 100
)


high_dimension_gender = (
    pd.crosstab(
        df.gender,
        df.number_of_high_dimensions,
        normalize="index"
    )
    * 100
)

#Saving the data
four_dimension_columns = (

    ["user", "gender"]

    + score_columns

    + level_columns

    + [
        "activity_pattern",
        "number_of_high_dimensions",
        "user_profile_cluster"
    ]
)


four_dimension_df = df[
    four_dimension_columns
].copy()


four_dimension_df.to_csv(
    "four_dimension_scores.csv",
    index=False
)


df.to_csv(
    "final_clean_clustering_results.csv",
    index=False
)

with pd.ExcelWriter(
    "complete_clustering_analysis.xlsx",
    engine="openpyxl"
) as writer:

    sheets = {

        "Final Results":
            df,

        "Clustering":
            clustering_results,

        "Best Clusters":
            best_results_df,

        "Four Dimensions":
            four_dimension_df,

        "Score Correlations":
            score_correlations,

        "User Profiles":
            profile_table,

        "Gender Summary":
            gender_summary,

        "Gender Statistics":
            gender_results_df,

        "Gender Patterns":
            gender_pattern_table,

        "Gender High Dimensions":
            high_dimension_gender
    }


    for sheet, data in sheets.items():

        data.to_excel(
            writer,
            sheet_name=sheet,
            index=True
        )




print("\n")
print("For the analysis")

print(
    "Rows:",
    len(df)
)

print(
    "Columns:",
    len(df.columns)
)


print("\nDimensions:")

for group in feature_groups:

    print("-",group)

print("\nDONE!")