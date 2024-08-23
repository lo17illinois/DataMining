import numpy as np 
import pandas as pd 
import mlxtend
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
#categorytest
#categories
with open('categories.txt', 'r') as file:
    data = [line.strip().split(';') for line in file]
    te = TransactionEncoder()
    te_ary = te.fit(data).transform(data)
    df = pd.DataFrame(te_ary, columns=te.columns_)
    min_support = 0.01  # Adjust the minimum support as needed
    frequent_itemsets = apriori(df, min_support=min_support, use_colnames=True, max_len=None)
    
print(data)
frequent_itemsets['index_col'] = frequent_itemsets.index

with open('patterns.txt', 'w') as patterns_file:
    print("hi")
    for _, row in frequent_itemsets.iterrows():
        print(row['index_col'])
        print(row)
        support_count = int(row['support'] * len(data))
        pattern_names = ';'.join(map(str, row['itemsets']))
        patterns_file.write(f"{support_count}:{pattern_names};\n")