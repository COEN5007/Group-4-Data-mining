import pandas as pd
from pathlib import Path
import numpy as np

folder = Path(__file__).resolve().parent

input_file = folder / "testdatatest.xlsx"
output_file = folder / "färdigskit.xlsx"

df = pd.read_excel(input_file)

print("Antal rader:", len(df))
print("Antal kolumner:", len(df.columns))

df.columns = (
    df.columns
    .astype(str)
    .str.strip()
)

column_fixes = {

    "Clls Unique People S-H":
        "Calls Unique People S-H"
}

df.rename(
    columns=column_fixes,
    inplace=True
)

if "user" in df.columns:

    df.drop(
        columns=["user"],
        inplace=True
    )

def clean_numeric(series):

    # Om det redan är numeriskt
    if pd.api.types.is_numeric_dtype(series):

        return pd.to_numeric(
            series,
            errors="coerce"
        )


    # Gör om till text
    cleaned = (
        series
        .astype(str)
        .str.strip()
    )


    # Hantera vanliga Excel-format
    cleaned = (
        cleaned
        .str.replace(",", ".", regex=False)
        .str.replace(" ", "", regex=False)
    )


    # Tomma värden
    cleaned = cleaned.replace(
        ["", "nan", "None", "null", "NULL"],
        np.nan
    )


    return pd.to_numeric(
        cleaned,
        errors="coerce"
    )

gender = clean_numeric(
    df["gender"]
)


df["gender"] = gender.map({

    0: "boy",

    1: "girl"
})

def categorize_count(series, prefix):

    values = clean_numeric(series)


    result = pd.Series(
        index=series.index,
        dtype="object"
    )

    missing = values.isna()


    zero = values == 0

    result.loc[zero] = (
        "no" + prefix
    )

    positive = values > 0

    positive_values = values.loc[positive]


    if len(positive_values) == 0:

        result.loc[missing] = (
            "unknown" + prefix
        )

        return result

    if positive_values.nunique() == 1:

        result.loc[positive] = (
            "middle" + prefix
        )

        result.loc[missing] = (
            "unknown" + prefix
        )

        return result

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

            duplicates="drop"
        )


        result.loc[positive] = (
            categories.astype(str)
        )


    except ValueError:

        ranks = positive_values.rank(
            method="first",
            pct=True
        )


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

    result.loc[missing] = (
        "unknown" + prefix
    )


    return result

def categorize_level(series, prefix):

    values = clean_numeric(series)


    result = pd.Series(
        index=series.index,
        dtype="object"
    )

    zero = values == 0

    result.loc[zero] = (
        "no" + prefix
    )

    positive = values > 0

    positive_values = values.loc[positive]

    if len(positive_values) == 0:

        result.loc[values.isna()] = (
            "unknown" + prefix
        )

        return result

    try:

        categories = pd.qcut(

            positive_values,

            q=5,

            labels=[
                "verylow" + prefix,
                "low" + prefix,
                "middle" + prefix,
                "high" + prefix,
                "veryhigh" + prefix
            ],

            duplicates="drop"
        )


        result.loc[positive] = (
            categories.astype(str)
        )


    except ValueError:

        ranks = positive_values.rank(
            method="first",
            pct=True
        )


        result.loc[positive_values.index] = ranks.apply(

            lambda x:

            "verylow" + prefix
            if x <= 0.20

            else "low" + prefix
            if x <= 0.40

            else "middle" + prefix
            if x <= 0.60

            else "high" + prefix
            if x <= 0.80

            else "veryhigh" + prefix
        )

    result.loc[values.isna()] = (
        "unknown" + prefix
    )


    return result


df["amount_fbf_school"] = categorize_count(

    df["amount_fbf_school"],

    "fb"
)


df.rename(

    columns={
        "amount_fbf_school": "fb"
    },

    inplace=True
)


LEVEL_COLUMNS = {

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
        "calldur_O-S-H"
}


for column, prefix in LEVEL_COLUMNS.items():

    print(
        "Kategoriserar:",
        column
    )

    df[column] = categorize_level(

        df[column],

        prefix
    )


COUNT_COLUMNS = {

    # Bluetooth
    "BT_Unique people S-H":
        "btunq_S-H",

    "BT_Outsiders S-H":
        "btout_S-H",

    "BT_Unique people O-S-H":
        "btunq_O-S-H",

    "BT_Outsiders O-S-H":
        "btout_O-S-H",


    # Calls S-H
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


    # Calls O-S-H
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


    # SMS S-H
    "SMS sent S-H":
        "smssent_S-H",

    "SMS received S-H":
        "smsrec_S-H",

    "SMS unique people S-H":
        "smsunq_S-H",

    "SMS conversation S-H":
        "smsconv_S-H",


    # SMS O-S-H
    "SMS sent O-S-H":
        "smssent_O-S-H",

    "SMS received O-S-H":
        "smsrec_O-S-H",

    "SMS unique people O-S-H":
        "smsunq_O-S-H",

    "SMS conversation O-S-H":
        "smsconv_O-S-H"
}


for column, prefix in COUNT_COLUMNS.items():

    print(
        "Kategoriserar:",
        column
    )

    df[column] = categorize_count(

        df[column],

        prefix
    )

for column in df.columns:

    print(
        f"\n{column}"
    )

    print(
        df[column]
        .value_counts(dropna=False)
        .head(10)
        .to_string()
    )

numeric_columns = df.select_dtypes(
    include="number"
).columns.tolist()



if numeric_columns:

    print(
        "Följande är fortfarande numeriska:"
    )

    for column in numeric_columns:
        print("-", column)

else:

    print(
        "Alla numeriska värden är konverterade."
    )


for column in df.columns:

    unknown_count = (
        df[column]
        .astype(str)
        .str.startswith("unknown")
        .sum()
    )


    if unknown_count > 0:

        print(
            column,
            ":",
            unknown_count,
            "unknown"
        )

df.to_excel(

    output_file,

    index=False
)

print(
    "Fil sparad:"
)

print(
    output_file
)

print(
    "\nAntal rader:",
    len(df)
)

print(
    "Antal kolumner:",
    len(df.columns)
)