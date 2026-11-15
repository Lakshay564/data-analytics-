# Experiment 9: Association Rule Mining using Apriori
# Aim: Find frequent itemsets and evaluate rules with support, confidence and lift.
# Association rules are not normally evaluated with classification accuracy.

%pip -q install mlxtend

import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

transactions = [
    ["Milk", "Bread", "Butter"],
    ["Bread", "Butter"],
    ["Milk", "Bread"],
    ["Milk", "Diapers", "Beer", "Bread"],
    ["Bread", "Butter", "Diapers"],
    ["Milk", "Diapers", "Beer"],
    ["Milk", "Bread", "Butter", "Diapers"],
    ["Bread", "Butter", "Beer"]
]

encoder = TransactionEncoder()
encoded = encoder.fit(transactions).transform(transactions)
basket = pd.DataFrame(encoded, columns=encoder.columns_)

print("Transaction dataset:")
print(basket)

frequent_itemsets = apriori(basket, min_support=0.25, use_colnames=True)
print("\nFrequent itemsets:")
print(frequent_itemsets)

rules = association_rules(
    frequent_itemsets, metric="confidence", min_threshold=0.5
)

if rules.empty:
    print("No rules found. Try lowering the confidence threshold.")
else:
    rules = rules.sort_values(["confidence", "lift"], ascending=False)
    print("\nAssociation rules:")
    print(rules[["antecedents", "consequents", "support", "confidence", "lift"]])
    print("\nRule evaluation:")
    print("Number of rules:", len(rules))
    print("Average support:", round(rules["support"].mean(), 3))
    print("Average confidence:", round(rules["confidence"].mean(), 3))
    print("Average lift:", round(rules["lift"].mean(), 3))
