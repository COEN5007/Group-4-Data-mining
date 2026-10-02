import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv("FINALFINALcleaned2oktober.csv")


print(data.shape)
print(data.head(10))
print(data.describe())


#Make the blutooth columns Duration to float 64 

cols = [
    "BT_Average Duration S-H",
    "BT_Average Duration O-S-H"
]

for col in cols:
    data[col] = pd.to_numeric(
        data[col].str.replace(",", ".", regex=False)
    )

print(data.dtypes)



