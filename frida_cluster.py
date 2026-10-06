from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

# Excel-filen
file_path = "cluster_mpunkt.xlsx"

# Mapp där alla resultat sparas
output_folder = Path("fäääärdig")
output_folder.mkdir(exist_ok=True)

# För att få samma resultat varje gång
RANDOM_STATE = 42


df = pd.read_excel(file_path)

# Rensa eventuella extra mellanslag i kolumnnamnen
df.columns = (
    df.columns.astype(str)
    .str.strip()
    .str.replace(r"\s+", " ", regex=True)
)

print(f"Antal rader: {len(df)}")
print(f"Antal kolumner: {len(df)}")


digital_school = [
    "Calls_Duration S-H",
    "Calls Made S-H",
    "Calls Received S-H",
    "Clls Unique People S-H",
    "sms sent S-H",
    "sms received S-H",
    "unique people S-H",
    "conversation S-H"
]


digital_outside = [
    "Calls Duration O-S-H",
    "Calls Made O-S-H",
    "Calls Received O-S-H",
    "Calls Unique People O-S-H",
    "sms sent O-S-H",
    "sms received O-S-H",
    "unique people O-S-H",
    "conversation O-S-H"
]

bluetooth_school = [
    "BT_Interactions S-H",
    "BT_Average Duration S-H",
    "BT_Unique people S-H",
    "BT_Outsiders S-H"
]

bluetooth_outside = [
    "BT_Interactions O-S-H",
    "BT_Average Duration O-S-H",
    "BT_Unique people O-S-H",
    "BT_Outsiders O-S-H"
]


# Alla variabler som används i K-Means
feature_blocks = {
    "Digital school": digital_school,
    "Digital outside": digital_outside,
    "Bluetooth school": bluetooth_school,
    "Bluetooth outside": bluetooth_outside
}

features = (
    digital_school
    + digital_outside
    + bluetooth_school
    + bluetooth_outside
)

missing_columns = [
    col for col in features
    if col not in df.columns
]

if missing_columns:
    print("\nFEL: Följande kolumner saknas i Excel-filen:")
    for col in missing_columns:
        print(" -", col)

    print("\nKolumner som faktiskt finns i filen:")
    for col in df.columns:
        print(" -", col)

    raise KeyError(
        "Några av K-Means-kolumnerna saknas. "
        "Kontrollera kolumnnamnen ovan."
    )

print("\nAlla K-Means-kolumner finns.")


X = df[features].copy()

# Försök göra alla variabler numeriska
for col in X.columns:
    X[col] = pd.to_numeric(X[col], errors="coerce")

print("\nAntal saknade värden före imputation:")
print(X.isna().sum().sum())

# Medianimputation
# Varje saknat värde ersätts med medianen för den kolumnen
X = X.fillna(X.median())

print("Saknade värden efter imputation:")
print(X.isna().sum().sum())


X_log = np.log1p(X)

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X_log)

X_scaled = pd.DataFrame(
    X_scaled,
    columns=features,
    index=df.index
)


X_balanced = pd.DataFrame(
    index=X_scaled.index
)

for block_name, block_features in feature_blocks.items():

    n_variables = len(block_features)

    X_balanced[block_features] = (
        X_scaled[block_features]
        / np.sqrt(n_variables)
    )

print("\nTestar antal kluster...")

k_values = range(2, 9)

silhouette_results = []

for k in k_values:

    kmeans = KMeans(
        n_clusters=k,
        random_state=RANDOM_STATE,
        n_init=20
    )

    labels = kmeans.fit_predict(X_balanced)

    score = silhouette_score(
        X_balanced,
        labels
    )

    silhouette_results.append({
        "k": k,
        "silhouette_score": score
    })

    print(
        f"k = {k}: "
        f"silhouette score = {score:.4f}"
    )


# Gör DataFrame
silhouette_df = pd.DataFrame(
    silhouette_results
)


silhouette_df.to_csv(
    output_folder / "01_silhouette_scores.csv",
    index=False
)


plt.figure(figsize=(8, 5))

plt.plot(
    silhouette_df["k"],
    silhouette_df["silhouette_score"],
    marker="o"
)

plt.xlabel("Antal kluster (k)")
plt.ylabel("Silhouette score")
plt.title("Silhouette score för olika antal kluster")

plt.xticks(list(k_values))
plt.grid(True, alpha=0.3)

plt.tight_layout()

plt.savefig(
    output_folder / "01_silhouette_scores.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


best_k = int(
    silhouette_df.loc[
        silhouette_df["silhouette_score"].idxmax(),
        "k"
    ]
)

best_score = float(
    silhouette_df.loc[
        silhouette_df["silhouette_score"].idxmax(),
        "silhouette_score"
    ]
)

print(f"Bästa antal kluster: {best_k}")
print(f"Bästa silhouette score: {best_score:.4f}")

final_kmeans = KMeans(
    n_clusters=best_k,
    random_state=RANDOM_STATE,
    n_init=20
)

cluster_labels = final_kmeans.fit_predict(
    X_balanced
)

# Lägg till kluster i originaldata
df["cluster"] = cluster_labels

cluster_sizes = (
    df["cluster"]
    .value_counts()
    .sort_index()
    .reset_index()
)

cluster_sizes.columns = [
    "cluster",
    "number_of_students"
]

cluster_sizes["percentage"] = (
    cluster_sizes["number_of_students"]
    / len(df)
    * 100
)

cluster_sizes.to_csv(
    output_folder / "02_cluster_sizes.csv",
    index=False
)

print("\nAntal studenter per kluster:")
print(cluster_sizes)


profile_data = X_scaled.copy()
profile_data["cluster"] = cluster_labels

cluster_profiles = (
    profile_data
    .groupby("cluster")[features]
    .mean()
)


# Lägg till blocknamn för varje variabel
block_lookup = {}

for block_name, block_features in feature_blocks.items():
    for feature in block_features:
        block_lookup[feature] = block_name


# Gör en tabell där blocket anges
profile_long = (
    cluster_profiles
    .reset_index()
    .melt(
        id_vars="cluster",
        var_name="variable",
        value_name="mean_standardized_value"
    )
)

profile_long["block"] = (
    profile_long["variable"]
    .map(block_lookup)
)

# Ordna kolumnerna
profile_long = profile_long[
    [
        "cluster",
        "block",
        "variable",
        "mean_standardized_value"
    ]
]

profile_long.to_csv(
    output_folder / "03_cluster_profiles.csv",
    index=False
)

plt.figure(
    figsize=(14, 7)
)

sns.heatmap(
    cluster_profiles,
    cmap="coolwarm",
    center=0,
    annot=False
)

plt.title(
    "Klusterprofiler – standardiserade medelvärden"
)

plt.xlabel("Variabel")
plt.ylabel("Kluster")

plt.xticks(
    rotation=70,
    ha="right"
)

plt.tight_layout()

plt.savefig(
    output_folder / "04_cluster_profiles_heatmap.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

pca = PCA(
    n_components=2,
    random_state=RANDOM_STATE
)

X_pca = pca.fit_transform(
    X_balanced
)

pca_df = pd.DataFrame(
    X_pca,
    columns=["PC1", "PC2"]
)

pca_df["cluster"] = cluster_labels

pca_df.to_csv(
    output_folder / "05_pca_coordinates.csv",
    index=False
)


plt.figure(figsize=(9, 7))

sns.scatterplot(
    data=pca_df,
    x="PC1",
    y="PC2",
    hue="cluster",
    palette="tab10",
    s=60,
    alpha=0.8
)

plt.title(
    "K-Means-kluster visualiserade med PCA"
)

plt.xlabel(
    f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f} %)"
)

plt.ylabel(
    f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f} %)"
)

plt.legend(
    title="Kluster"
)

plt.tight_layout()

plt.savefig(
    output_folder / "06_pca_clusters.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


gender_distribution = pd.crosstab(
    df["gender"],
    df["cluster"],
    normalize="index"
) * 100

# Gör kolumnnamnen tydligare
gender_distribution.columns = [
    f"cluster_{col}"
    for col in gender_distribution.columns
]

gender_distribution.index.name = "gender"

gender_distribution.to_csv(
    output_folder / "07_gender_distribution.csv"
)

print("\nGender-fördelning (% inom varje gender):")
print(gender_distribution)


ax = gender_distribution.plot(
    kind="bar",
    stacked=True,
    figsize=(9, 6)
)

plt.title(
    "Klusterfördelning inom varje gender"
)

plt.xlabel("Gender")
plt.ylabel("Andel (%)")

plt.xticks(
    rotation=0
)

plt.legend(
    title="Kluster",
    bbox_to_anchor=(1.02, 1),
    loc="upper left"
)

plt.ylim(0, 100)

plt.tight_layout()

plt.savefig(
    output_folder / "08_gender_distribution.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


facebook_by_cluster = (
    df.groupby("cluster")["ammount_fbf_school"]
    .agg(
        mean="mean",
        median="median",
        min="min",
        max="max",
        count="count"
    )
    .reset_index()
)

facebook_by_cluster.to_csv(
    output_folder / "09_facebook_by_cluster.csv",
    index=False
)


df.to_csv(
    output_folder / "10_students_with_clusters.csv",
    index=False
)


print(f"Antal studenter: {len(df)}")
print(f"Antal variabler i K-Means: {len(features)}")
print(f"Valt antal kluster: {best_k}")
print(f"Silhouette score: {best_score:.4f}")

print("\nKlusterstorlekar:")
print(cluster_sizes)

print("\nFiler sparade i:")
print(output_folder.resolve())

print("\nSkapade filer:")
print("01_silhouette_scores.png")
print("01_silhouette_scores.csv")
print("02_cluster_sizes.csv")
print("03_cluster_profiles.csv")
print("04_cluster_profiles_heatmap.png")
print("05_pca_coordinates.csv")
print("06_pca_clusters.png")
print("07_gender_distribution.csv")
print("08_gender_distribution.png")
print("09_facebook_by_cluster.csv")
print("10_students_with_clusters.csv")

print("\nKörningen är färdig!")
