import pandas as pd
from pathlib import Path

from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules


# ============================================================
# 1. Hitta Excel-filen
# ============================================================

folder = Path(__file__).resolve().parent
file_path = folder / "färdigskit.xlsx"

df = pd.read_excel(file_path)


# ============================================================
# 2. Förbered data
# ============================================================

df = df.iloc[:, 1:-1]

df.columns = df.columns.astype(str).str.strip()

for col in df.columns:
    if df[col].dtype == "object":
        df[col] = df[col].str.strip()

if "user" in df.columns:
    df = df.drop(columns=["user"])

transactions = []

for _, row in df.iterrows():

    transaction = []

    for col in df.columns:

        value = row[col]

        if pd.notna(value):

            value = str(value).strip()

            # Ta inte med värden som börjar med "no"
            if value.lower().startswith("no"):
                continue

            transaction.append(
                f"{col}={value}"
            )

    transactions.append(transaction)


print(
    "Transactions created:",
    len(transactions)
)


te = TransactionEncoder()

encoded = te.fit(
    transactions
).transform(
    transactions
)

transaction_df = pd.DataFrame(
    encoded,
    columns=te.columns_
)

print(
    "Number of items:",
    len(te.columns_)
)


MIN_SUPPORT = 0.10
MIN_CONFIDENCE = 0.60
MIN_LIFT = 1.00

MAX_ITEMSET_LENGTH = 8
MIN_RULE_LENGTH = 2

MIN_SUPPORT_COUNT = 5


# ============================================================
# 6. Apriori
# ============================================================

frequent_itemsets = apriori(
    transaction_df,
    min_support=MIN_SUPPORT,
    use_colnames=True,
    max_len=MAX_ITEMSET_LENGTH
)

print(
    "Frequent itemsets found:",
    len(frequent_itemsets)
)


# ============================================================
# 7. Kontrollera att itemsets hittades
# ============================================================

if frequent_itemsets.empty:

    print("\nNo frequent itemsets were found.")
    print(
        "Try lowering MIN_SUPPORT."
    )

else:

    # ========================================================
    # 8. Association rules
    # ========================================================

    rules = association_rules(
        frequent_itemsets,
        metric="confidence",
        min_threshold=MIN_CONFIDENCE
    )


    # ========================================================
    # 9. Filtrera regler
    # ========================================================

    rules = rules[
        (rules["support"] >= MIN_SUPPORT)
        & (rules["confidence"] >= MIN_CONFIDENCE)
        & (rules["lift"] >= MIN_LIFT)
    ].copy()


    # ========================================================
    # 10. Regelns längd
    # ========================================================

    rules["rule_length"] = (
        rules["antecedents"].apply(len)
        +
        rules["consequents"].apply(len)
    )

    rules = rules[
        rules["rule_length"] >= MIN_RULE_LENGTH
    ]


    # ========================================================
    # 11. Support count
    # ========================================================

    number_of_transactions = len(transaction_df)

    rules["support_count"] = (
        rules["support"] * number_of_transactions
    ).round().astype(int)

    rules = rules[
        rules["support_count"] >= MIN_SUPPORT_COUNT
    ]


    # ========================================================
    # 12. Ta bort dubbletter
    # ========================================================

    rules = rules.drop_duplicates(
        subset=[
            "antecedents",
            "consequents"
        ]
    )


    # ========================================================
    # 13. Sortera
    # ========================================================

    rules = rules.sort_values(
        ["lift", "confidence", "support"],
        ascending=False
    )


    # ========================================================
    # 14. Ta de 10 bästa
    # ========================================================

    result = rules.head(10).copy()


    # ========================================================
    # 15. Gör reglerna läsbara
    # ========================================================

    result["antecedents_text"] = result[
        "antecedents"
    ].apply(
        lambda x: "\nAND ".join(
            sorted(map(str, x))
        )
    )

    result["consequents_text"] = result[
        "consequents"
    ].apply(
        lambda x: "\nAND ".join(
            sorted(map(str, x))
        )
    )


    # ========================================================
    # 16. Skriv ut resultat
    # ========================================================

    print("\n")
    print("LONG ASSOCIATION RULES")


    if result.empty:

        print(
            "\nNo association rules were found "
            "after filtering."
        )

    else:

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


        # ====================================================
        # 17. Spara till Excel
        # ====================================================

        output_path = folder / "apriori_results.xlsx"

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

        result_to_save = result_to_save.rename(
            columns={
                "antecedents_text": "IF",
                "consequents_text": "THEN",
                "support": "Support",
                "confidence": "Confidence",
                "lift": "Lift",
                "rule_length": "Rule length",
                "support_count": "Support count"
            }
        )

        result_to_save.to_excel(
            output_path,
            index=False
        )

        print("\n")
        print("RESULTS SAVED")
        print(f"File: {output_path}")