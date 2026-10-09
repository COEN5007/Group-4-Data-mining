import pandas as pd
from pathlib import Path
import numpy as np
from sklearn.cluster import KMeans #Gemma lade till för att kunna göra kmeans clustering
from sklearn.metrics import silhouette_score #Gemma lade till

folder = Path(__file__).resolve().parent #Finding the right folder
input_file = folder / "testdatatest.xlsx"
output_file = folder / "färdigskit.xlsx"

df = pd.read_excel(input_file)

# Kontrollera exakt vilka kolumnnamn som finns i testdatatest.xlsx
print("\nCOLUMN NAMES IN testdatatest.xlsx:")
print(df.columns.tolist())

# (Gemma) lägger till för att kontrollera att FAMILIES stämmer
#delar upp clusters på naturliga sätt 

corr = df.select_dtypes("number").corr(method="spearman")
pairs = corr.where(np.triu(np.ones(corr.shape), 1).astype(bool)).stack()
print(pairs[pairs.abs() > 0.8].sort_values(ascending=False))

print("Original rows:", len(df))
print("Original columns:", len(df.columns))

if "user" in df.columns:
    df.drop(columns=["user"], inplace=True) #removes the user since the id is not important


df["gender"] = df["gender"].map({0: "boy", 1: "girl"}) #converts the gender into texts where 0=boy och 1=girl

#=============================== Gemma la till för att kunna göra kmeans clustering
COUNT_LABELS = {
    2: ["few", "many"],
    3: ["few", "middle", "many"],
    4: ["veryfew", "few", "many", "verymany"],
    5: ["veryfew", "few", "middle", "many", "verymany"],
}
LEVEL_LABELS = {
    2: ["low", "high"],
    3: ["low", "middle", "high"],
    4: ["verylow", "low", "high", "veryhigh"],
    5: ["verylow", "low", "middle", "high", "veryhigh"],
}

def natural_cut(positive_values, prefix, label_sets,       
                k_range=(3, 4), log=True, min_share=0.10):
    # Delar positiva värden i naturliga bins med 1D k-means (på log1p).
    # k väljs sedan via silhouette, men bara bland k där minsta bin >= min_share.
    x = np.log1p(positive_values.astype(float)) if log else positive_values.astype(float)
    X = x.to_numpy().reshape(-1, 1)

    best = None
    for k in range(k_range[0], k_range[1] + 1):
        if len(np.unique(X)) <= k:
            break
        km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X)
        shares = np.bincount(km.labels_, minlength=k) / len(X)
        if shares.min() < min_share:
            continue
        score = silhouette_score(X, km.labels_)
        if best is None or score > best[0]:
            best = (score, k, km)

    if best is None:  # fallback: tertiler (som kvantiler men tre grupper och med rangordning)
        cats = pd.qcut(positive_values.rank(method="first"), q=3, labels=label_sets[3])
        return cats.astype(str) + prefix

    _, k, km = best
    order = np.argsort(km.cluster_centers_.ravel())   # sortera bins låg -> hög
    rank = np.empty(k, dtype=int)
    rank[order] = np.arange(k)
    names = np.array(label_sets[k])
    return pd.Series(names[rank[km.labels_]], index=positive_values.index) + prefix
#=============================== 

# (Gemma) Jag kommenterar ut den gamla kategoriseringsfunktionen eftersom jag har gjort en ny som använder kmeans clustering istället för qcut.
"""def categorize_count(series, prefix):

    values = series.copy()

    result = pd.Series(index=series.index, dtype="object")

    missing = values.isna() #handling missing values, if there are any they will be marked as unknown
    zero = values == 0 #handling zero values
    result.loc[zero] = ("no" + prefix) #if it is a zero, mark it as no so we can remove the 0-values

    #Find positive values
    positive = values > 0
    positive_values = values.loc[positive]

    #Divide positive values into five groups, based on the quantiles of the positive values
    try:
        categories = pd.qcut(
            positive_values,
            q=5,
            labels=[
                "veryfew" + prefix,
                "few" + prefix,
                "middle" + prefix,
                "many" + prefix,
                "verymany" + prefix
            ],
            duplicates="drop")

        result.loc[positive] = (categories.astype(str))


    #If qcut cannot create five groups, use ranking
    except ValueError:

        ranks = positive_values.rank(method="first", pct=True)

        result.loc[positive] = ranks.apply(

            lambda x:

            "veryfew" + prefix
            if x <= 0.20

            else "few" + prefix
            if x <= 0.40

            else "middle" + prefix
            if x <= 0.60

            else "many" + prefix
            if x <= 0.80

            else "verymany" + prefix
        )

    #mark missing values as unknown
    result.loc[missing] = ("unknown" + prefix)

    return result"""

COLUMNS = {

    "amount_fbf_school",

    "BT_Interactions S-H",

    "BT_Average Duration S-H",

    "BT_Interactions O-S-H",

    "BT_Average Duration O-S-H",

    "Calls_Duration S-H",

    "Calls Duration O-S-H",

    "BT_Unique people S-H",

    "BT_Outsiders S-H",

    "BT_Unique people O-S-H",

    "BT_Outsiders O-S-H",

    "Calls Made S-H",

    "Calls Received S-H",

    "Calls Missed As Caller S-H",

    "Calls Missed As Callee S-H",

    "Calls Unique People S-H",

    "Calls Made O-S-H",

    "Calls Received O-S-H",

    "Calls Missed As Caller O-S-H",

    "Calls Missed As Callee O-S-H",

    "Calls Unique People O-S-H",

    "SMS sent S-H",

    "SMS received S-H",

    "SMS unique people S-H",

    "SMS conversation S-H",

    "SMS sent O-S-H",

    "SMS received O-S-H",
    
    "SMS unique people O-S-H",

    "SMS conversation O-S-H",
}

# (Gemma) kommenterar ut den undre delen och gör en ny
"""#Categorize all variables
for column, prefix in COLUMNS.items():
    df[column] = categorize_count(df[column], prefix)
"""

# (Gemma) Nedan följer ny kod för kmeans clustering istället för qcut.
def categorize_count(series, prefix, label_sets=COUNT_LABELS):
    values = series.copy()
    result = pd.Series(index=series.index, dtype="object")

    missing = values.isna() | (values < 0)     # -1 = ingen data
    zero = values == 0
    positive = values > 0

    result.loc[zero] = "no" + prefix
    result.loc[missing] = "unknown" + prefix
    result.loc[positive] = natural_cut(values.loc[positive], prefix, label_sets)
    return result

for column, prefix in COLUMNS.items():
    label_sets = LEVEL_LABELS if "Duration" in column else COUNT_LABELS
    df[column] = categorize_count(df[column], prefix, label_sets)

#Rename Facebook column
df.rename(
    columns={
        "amount_fbf_school": "fb"
    },
    inplace=True
)


df.to_excel(
    output_file,
    index=False
)

print("\nFile saved:")
print(output_file)

print("\nRows:", len(df))
print("Columns:", len(df.columns))