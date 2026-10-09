import pandas as pd
from pathlib import Path

from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

folder = Path(__file__).resolve().parent #Added to find the right folder 
file_path = folder / "färdigskit.xlsx" #The finnished file with all of the results
df = pd.read_excel(file_path)

# (Gemma) Kontrollera exakt vilka kolumner Apriori får från färdigskit.xlsx
print("\nCOLUMN NAMES IN färdigskit.xlsx:")
print(df.columns.tolist())

"""df = df.iloc[:, 1:] #"User" is not of any importance for algorithm, removed here""" # (Gemma) Kommenterar ut, since we already removed the user in cleaning_data.py
df.columns = df.columns.astype(str).str.strip()

#Remove unnecessary spaces from text values, only cleaning the format
for col in df.columns:
    if df[col].dtype == "object":
        df[col] = df[col].str.strip()

transactions = [] #Creating list of all transactions so the rows are "tied" together, the algorithm can thenn read it

for _, row in df.iterrows():

    transaction = []

    for col in df.columns:

        value = row[col]

        if pd.notna(value): #To remove the values that we do not want

            value = str(value).strip()
            if value.lower().startswith(("no", "unknown")): #Remove the empty values, all that starts with "no", (Gemma): also "unknown" will be removed, since they are not important for the algorithm
                continue

            transaction.append(f"{col}={value}") #Adds the value that we want in a new column with name and value

    transactions.append(transaction) #Creates finnished list of transactions


print("Transactions created:", len(transactions)) #Checking ammount of transactions, should be same ammount as rows in og

te = TransactionEncoder()

encoded = te.fit(transactions).transform(transactions) #Codes the columns into true/false format to make it easier to read

transaction_df = pd.DataFrame(encoded,columns=te.columns_)

print("Number of items:", len(te.columns_)) #Checking ammount of items, should be same ammount as columns in og

# ===============================
# (Gemma) FAMILIES: Kolumner med samma typ av aktivitet
FAMILIES = {

    "BT_Interactions": [
        "BT_Interactions S-H",
        "BT_Interactions O-S-H",
    ],

    "BT_Average_Duration": [
        "BT_Average Duration S-H",
        "BT_Average Duration O-S-H",
    ],

    "BT_Unique_People": [
        "BT_Unique people S-H",
        "BT_Unique people O-S-H",
    ],

    "BT_Outsiders": [
        "BT_Outsiders S-H",
        "BT_Outsiders O-S-H",
    ],

    "Calls_Duration": [
        "Calls_Duration S-H",
        "Calls Duration O-S-H",
    ],

    "Calls_Made": [
        "Calls Made S-H",
        "Calls Made O-S-H",
    ],

    "Calls_Received": [
        "Calls Received S-H",
        "Calls Received O-S-H",
    ],

    "Calls_Missed_As_Caller": [
        "Calls Missed As Caller S-H",
        "Calls Missed As Caller O-S-H",
    ],

    "Calls_Missed_As_Callee": [
        "Calls Missed As Callee S-H",
        "Calls Missed As Callee O-S-H",
    ],

    "Calls_Unique_People": [
        "Calls Unique People S-H",
        "Calls Unique People O-S-H",
    ],

    "SMS_Sent": [
        "SMS sent S-H",
        "SMS sent O-S-H",
    ],

    "SMS_Received": [
        "SMS received S-H",
        "SMS received O-S-H",
    ],

    "SMS_Unique_People": [
        "SMS unique people S-H",
        "SMS unique people O-S-H",
    ],

    "SMS_Conversation": [
        "SMS conversation S-H",
        "SMS conversation O-S-H",
    ],
}

#=============================== (Gemma) Lade till för att kunna göra kmeans clustering
COL_TO_FAMILY = {col: fam for fam, cols in FAMILIES.items() for col in cols}

def family_of(item):
    col = item.split("=", 1)[0]
    return COL_TO_FAMILY.get(col, col)       # okänd kolumn = egen familj

def one_per_family(itemset):
    fams = [family_of(i) for i in itemset]
    return len(fams) == len(set(fams))

# (Gemma) för stabilitet
# ===============================
# STABILITY TEST
# ===============================

def make_rule_keys(rules_df):
    """
    Skapar en jämförbar textnyckel för varje regel.
    Ordningen på items spelar ingen roll.
    """
    if rules_df.empty:
        return set()

    return {
        (
            tuple(sorted(map(str, row["antecedents"]))),
            tuple(sorted(map(str, row["consequents"])))
        )
        for _, row in rules_df.iterrows()
    }


def run_apriori_on_transactions(transactions_subset):
    """
    Kör samma Apriori-regler på en delmängd av transaktionerna.
    """

    te_half = TransactionEncoder()

    encoded_half = te_half.fit(transactions_subset).transform(
        transactions_subset
    )

    transaction_df_half = pd.DataFrame(
        encoded_half,
        columns=te_half.columns_
    )

    if transaction_df_half.empty:
        return pd.DataFrame()

    frequent_itemsets_half = apriori(
        transaction_df_half,
        min_support=MIN_SUPPORT,
        use_colnames=True
    )

    if frequent_itemsets_half.empty:
        return pd.DataFrame()

    frequent_itemsets_half = frequent_itemsets_half[
        frequent_itemsets_half["itemsets"].apply(one_per_family)
    ]

    if frequent_itemsets_half.empty:
        return pd.DataFrame()

    rules_half = association_rules(
        frequent_itemsets_half,
        metric="confidence",
        min_threshold=MIN_CONFIDENCE
    )

    rules_half = rules_half[
        (rules_half["support"] >= MIN_SUPPORT)
        & (rules_half["confidence"] >= MIN_CONFIDENCE)
        & (rules_half["lift"] >= MIN_LIFT)
    ].copy()

    if rules_half.empty:
        return pd.DataFrame()

    rules_half["rule_length"] = (
        rules_half["antecedents"].apply(len)
        + rules_half["consequents"].apply(len)
    )

    rules_half = rules_half[
        rules_half["rule_length"] >= MIN_RULE_LENGTH
    ]

    number_of_transactions_half = len(transaction_df_half)

    rules_half["support_count"] = (
        rules_half["support"] * number_of_transactions_half
    ).round().astype(int)

    rules_half = rules_half[
        rules_half["support_count"] >= MIN_SUPPORT_COUNT
    ]

    rules_half = rules_half.drop_duplicates(
        subset=["antecedents", "consequents"]
    )

    return rules_half
#==========================================

#Values that we play around with, keep lift at 1.00 at least
MIN_SUPPORT = 0.05 # (Gemma) Changed from 0.01 to 0.05, since support favours rules that are more common
MIN_CONFIDENCE = 0.50
MIN_LIFT = 1.2 # (Gemma) Changed from 1.0 to 1.2, since Lift favours rules that are less common

MIN_RULE_LENGTH = 3 #Minimum total to make a rule, must be a rule with at least 2 ifs 

MIN_SUPPORT_COUNT = 5 #Minimal ammount to support the rule, might be removed

#======================== (Gemma) Added for stability test
import random

random.seed(42)

transactions_copy = transactions.copy()
random.shuffle(transactions_copy)

half = len(transactions_copy) // 2

transactions_a = transactions_copy[:half]
transactions_b = transactions_copy[half:]

print("\nRunning stability test...")
print("Half A transactions:", len(transactions_a))
print("Half B transactions:", len(transactions_b))

rules_a = run_apriori_on_transactions(transactions_a)
rules_b = run_apriori_on_transactions(transactions_b)

keys_a = make_rule_keys(rules_a)
keys_b = make_rule_keys(rules_b)

stable_keys = keys_a.intersection(keys_b)

print("\nSTABLE RULES (found in BOTH halves):")

for number, key in enumerate(
    sorted(stable_keys),
    start=1
):
    antecedents, consequents = key

    print(f"\nSTABLE RULE {number}")
    print("IF")
    print("        " + "\n AND ".join(antecedents))

    print("\nTHEN")
    print("        " + "\n AND ".join(consequents))

print("\nSTABILITY RESULTS")
print("Rules in half A:", len(keys_a))
print("Rules in half B:", len(keys_b))
print("Rules in both halves:", len(stable_keys))
# =========================

#Creates a list of all itemsets that  have higher than the minimum support 
frequent_itemsets = apriori(transaction_df, min_support=MIN_SUPPORT, use_colnames=True,)

# (Gemma) Behåller bara itemsets som har max 1 item per family, annars blir det bara massa rules med samma family
frequent_itemsets = frequent_itemsets[frequent_itemsets["itemsets"].apply(one_per_family)]

print("Frequent itemsets found:", len(frequent_itemsets))

if not frequent_itemsets.empty: #If we have found itemsets, will here create the rules, otherwise skip

    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=MIN_CONFIDENCE) #Applying the minimum confidence to the rules already created

    #Keep only rules that meet our requirements
    rules = rules[(rules["support"] >= MIN_SUPPORT)
        & (rules["confidence"] >= MIN_CONFIDENCE)
        & (rules["lift"] >= MIN_LIFT)
    ].copy()

    #Calculate the total number of items in each rule, just a summary of whats been created
    rules["rule_length"] = (
        rules["antecedents"].apply(len)
        + rules["consequents"].apply(len)
    )

    #Keep only rules longer than the minimum length
    rules = rules[rules["rule_length"] >= MIN_RULE_LENGTH]

    #Calculate how many transactions support each rule
    number_of_transactions = len(transaction_df)

    rules["support_count"] = (rules["support"] * number_of_transactions).round().astype(int)

    #Remove rules not meeting minimum support
    rules = rules[rules["support_count"] >= MIN_SUPPORT_COUNT]

    #Remove exact duplicate rules, since time no matter, remove if the if and then are same but different order
    rules = rules.drop_duplicates(subset=["antecedents", "consequents"])

    # (Gemma) högre positivt leverage-värde betyder att regeln
    # omfattar fler personer än vad som skulle förväntas av slumpen.
    rules["leverage_count"] = (
        rules["leverage"] * number_of_transactions
    ).round().astype(int)

    rules = rules.sort_values(
        ["lift", "confidence"], # (Gemma) Changed from ["lift", "confidence", "support"] to ["lift", "confidence"], since support favours rules that are more common
        ascending=False
    ) #Sorting the rules so best rules are on top

    #Keep the 10 best rules, can be removed but gets more effective so so
    result = rules.head(10).copy()

#Checking with the previous list where it is wrong if there are no results, since always a copy we can go back
if frequent_itemsets.empty:

    print("\nNo frequent itemsets were found.")

elif result.empty:

    print("\nNo association rules were found.")

else:

    #Convert antecedents into readable text
    result["antecedents_text"] = result[
        "antecedents"
    ].apply(
        lambda x: "\n AND ".join(
            sorted(map(str, x))
        )
    )

    #Convert consequents into readable text
    result["consequents_text"] = result[
        "consequents"
    ].apply(
        lambda x: "\n AND ".join(
            sorted(map(str, x))
        )
    )

    print("\n")
    print("LONG ASSOCIATION RULES")

    number_of_transactions = len(transaction_df)

    #Print the rules
    for number, (_, row) in enumerate(
        result.iterrows(),
        start=1
    ):

        print(f"\nRULE {number}")

        print("IF")
        print(
            f"        {row['antecedents_text']}"
        )

        print("\nTHEN")
        print(
            f"        {row['consequents_text']}"
        )

        print("\nStatistics:")

        print(
            f"        Support:     "
            f"{row['support']:.1%} " 
            f"({int(row['support_count'])} "
            f"of {number_of_transactions})"
        )

        print(
            f"        Confidence:  "
            f"{row['confidence']:.1%}"
        )

        print(
            f"        Lift:        "
            f"{row['lift']:.2f}"
        )
        print(
            f"        Leverage:    " #Gemma
            f"{row['leverage']:.4f} "
            f"(ca {int(row['leverage_count'])} personer)"
        )
        print(
            f"        Rule length: "
            f"{int(row['rule_length'])} items"
        )

    print(
        "\nNUMBER OF RULES SHOWN:",
        len(result)
    )

    #Create the output file path, se name above
    output_path = folder / "apriori_results.xlsx"

    #Select the columns we want to save, all of the ones that meets the rules
    result_to_save = result[
        [
            "antecedents_text",
            "consequents_text",
            "support",
            "confidence",
            "lift",
            "leverage", #Gemma
            "leverage_count", #Gemma
            "rule_length",
            "support_count"
        ]
    ].copy()

    #Rename the columns to make the Excel file easier to read
    result_to_save = result_to_save.rename(
        columns={
            "antecedents_text": "IF",
            "consequents_text": "THEN",
            "support": "Support",
            "confidence": "Confidence",
            "lift": "Lift",
            "leverage": "Leverage", #Gemma
            "leverage_count": "Leverage count", #Gemma
            "rule_length": "Rule length",
            "support_count": "Support count"})




    
    #Save the results, take away if not gonna save, also output in the beginning
    result_to_save.to_excel(output_path, index=False)

    print("RESULTS SAVED")
    print(f"File: {output_path}")
