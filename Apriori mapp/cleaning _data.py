import pandas as pd
from pathlib import Path
import numpy as np

folder = Path(__file__).resolve().parent #Finding the right folder
input_file = folder / "testdatatest.xlsx"
output_file = folder / "färdigskit.xlsx"

df = pd.read_excel(input_file)

print("Original rows:", len(df))
print("Original columns:", len(df.columns))

if "user" in df.columns:
    df.drop(columns=["user"], inplace=True) #removes the user since the id is not important


df["gender"] = df["gender"].map({0: "boy", 1: "girl"}) #converts the gender into texts where 0=boy och 1=girl


def categorize_count(series, prefix):

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

    return result

COLUMNS = {

    "amount_fbf_school":
        "fb",

    "BT_Interactions S-H":
        "btint_S-H",

    "BT_Average Duration S-H":
        "btavgdur_S-H",

    "BT_Interactions O-S-H":
        "btint_O-S-H",

    "BT_Average Duration O-S-H":
        "btavgdur_O-S-H",

    "Calls_Duration S-H":
        "calldur_S-H",

    "Calls Duration O-S-H":
        "calldur_O-S-H",

    "BT_Unique people S-H":
        "btunq_S-H",

    "BT_Outsiders S-H":
        "btout_S-H",

    "BT_Unique people O-S-H":
        "btunq_O-S-H",

    "BT_Outsiders O-S-H":
        "btout_O-S-H",

    "Calls Made S-H":
        "callsmade_S-H",

    "Calls Received S-H":
        "callsrec_S-H",

    "Calls Missed As Caller S-H":
        "misscall_S-H",

    "Calls Missed As Callee S-H":
        "misscallee_S-H",

    "Calls Unique People S-H":
        "callsunq_S-H",

    "Calls Made O-S-H":
        "callsmade_O-S-H",

    "Calls Received O-S-H":
        "callsrec_O-S-H",

    "Calls Missed As Caller O-S-H":
        "misscall_O-S-H",

    "Calls Missed As Callee O-S-H":
        "misscallee_O-S-H",

    "Calls Unique People O-S-H":
        "callsunq_O-S-H",

    "SMS sent S-H":
        "smssent_S-H",

    "SMS received S-H":
        "smsrec_S-H",

    "SMS unique people S-H":
        "smsunq_S-H",

    "SMS conversation S-H":
        "smsconv_S-H",

    "SMS sent O-S-H":
        "smssent_O-S-H",

    "SMS received O-S-H":
        "smsrec_O-S-H",

    "SMS unique people O-S-H":
        "smsunq_O-S-H",

    "SMS conversation O-S-H":
        "smsconv_O-S-H"
}


#Categorize all variables
for column, prefix in COLUMNS.items():
    df[column] = categorize_count(df[column], prefix)


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