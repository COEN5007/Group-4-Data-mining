import pandas as pd
from pathlib import Path

from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

folder = Path(__file__).resolve().parent #Added to find the right folder 
file_path = folder / "färdigskit.xlsx" #The finnished file with all of the results
df = pd.read_excel(file_path)

df = df.iloc[:, 1:] #"User" is not of any importance for algorithm, removed here
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
            if value.lower().startswith("no"): #Remove the empty values, all that starts with "no"
                continue

            transaction.append(f"{col}={value}") #Adds the value that we want in a new column with name and value

    transactions.append(transaction) #Creates finnished list of transactions


print("Transactions created:", len(transactions)) #Checking ammount of transactions, should be same ammount as rows in og


te = TransactionEncoder()

encoded = te.fit(transactions).transform(transactions) #Codes the columns into true/false format to make it easier to read

transaction_df = pd.DataFrame(encoded,columns=te.columns_)

print("Number of items:", len(te.columns_)) #Checking ammount of items, should be same ammount as columns in og

#Values that we play around with, keep lift at 1.00 at least
MIN_SUPPORT = 0.10
MIN_CONFIDENCE = 0.50
MIN_LIFT = 1.00

MIN_RULE_LENGTH = 3 #Minimum total to make a rule, must be a rule with at least 2 ifs 

MIN_SUPPORT_COUNT = 5 #Minimal ammount to support the rule, might be removed

#Creates a list of all itemsets that  have higher than the minimum support 
frequent_itemsets = apriori(transaction_df, min_support=MIN_SUPPORT, use_colnames=True,)

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


    rules = rules.sort_values(
        ["lift", "confidence", "support"],
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
        lambda x: "\nAND ".join(
            sorted(map(str, x))
        )
    )

    #Convert consequents into readable text
    result["consequents_text"] = result[
        "consequents"
    ].apply(
        lambda x: "\nAND ".join(
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
            "rule_length": "Rule length",
            "support_count": "Support count"})

    #Save the results, take away if not gonna save, also output in the beginning
    result_to_save.to_excel(output_path, index=False)

    print("RESULTS SAVED")
    print(f"File: {output_path}")
