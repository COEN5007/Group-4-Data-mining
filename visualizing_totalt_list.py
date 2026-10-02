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





# Don't plot ID or categorical variable
plot_columns = data.columns.drop(["user", "gender"])

hist_columns = data.columns.drop(["user", "gender"])

for col in hist_columns:

    plt.figure(figsize=(8, 5))

    plt.hist(
        data[col],
        bins=30,
        edgecolor="black"
    )

    plt.title(f"Distribution of {col}")
    plt.xlabel(col)
    plt.ylabel("Number of users")

    plt.tight_layout()
    plt.show()


