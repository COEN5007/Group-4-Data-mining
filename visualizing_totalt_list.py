import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv("FinalTotalList2oct.csv")


print(data.shape)
print(data.head(10))
print(data.describe())


#Make the bluetooth columns duration to float64, so now all columns are float and i'm able to plot them! 

cols = [
    "BT_Average Duration S-H",
    "BT_Average Duration O-S-H"
]

for col in cols:
    data[col] = pd.to_numeric(
        data[col].str.replace(",", ".", regex=False)
    )

print(data.dtypes)




"""boxplot_columns = data.columns.drop(["user", "gender"])

for col in boxplot_columns:

    plt.figure(figsize=(8, 4))

    plt.boxplot(
        data[col],
        vert=False
    )

    plt.title(f"Boxplot of {col}")
    plt.xlabel(col)

    plt.tight_layout()
    plt.show()"""



heatmap_data = data.drop(columns=["user", "gender"])

# Calculate correlation matrix
corr_matrix = heatmap_data.corr()

# Plot heatmap
plt.figure(figsize=(20, 16))

sns.heatmap(
    corr_matrix,
    cmap="coolwarm",
    center=0,
    vmin=-1,
    vmax=1,
    annot=True,
    fmt=".2f",
    linewidths=0.5
)

plt.title("Correlation Heatmap of Social Interaction Features")

plt.tight_layout()
plt.show()




