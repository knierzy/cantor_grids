# Compares two Excel datasets based on the first four columns to identify identical entries.
# Each row is treated as a 4-component tuple (as strings) and exact matches are calculated.
# The script computes the number and relative frequency of identical rows in each dataset.
# Useful for identifying data overlaps or redundancies between two datasets.
# In this example, the input files refer to greenschist and blueschist compositions, respectively.

import pandas as pd

# File paths
file1 = "data/D5_constrained_cartesian_product_greenschists.xlsx"
file2 = "data/D3_constrained_cartesian_product_blueschists.xlsx"

# read Excel files
df1 = pd.read_excel(file1, usecols=[0, 1, 2, 3])
df2 = pd.read_excel(file2, usecols=[0, 1, 2, 3])

# number of rows read per file
num_rows_file1 = len(df1)
num_rows_file2 = len(df2)

# convert columns to strings to ensure exact matches
df1 = df1.astype(str)
df2 = df2.astype(str)

# convert the first 4 columns of each file into tuples for comparison
set1 = set([tuple(row) for row in df1.values])
set2 = set([tuple(row) for row in df2.values])

# find identical rows
identical_rows = set1.intersection(set2)
identical_count = len(identical_rows)

# calculate relative frequencies
relative_freq_file1 = (identical_count / num_rows_file1) * 100 if num_rows_file1 > 0 else 0
relative_freq_file2 = (identical_count / num_rows_file2) * 100 if num_rows_file2 > 0 else 0

# output results
print(f"Number of rows read from greenschists_test.xlsx: {num_rows_file1}")
print(f"Number of rows read from blueschists_test.xlsx: {num_rows_file2}")
print(f"Number of identical rows: {identical_count}")
print(f"Relative frequency in greenschists_test.xlsx: {relative_freq_file1:.2f}%")
print(f"Relative frequency in blueschists_test.xlsx: {relative_freq_file2:.2f}%")

# show the first 10 rows from each file
print("\nFirst 10 rows from greenschists_test.xlsx:")
print(df1.head(10))

print("\nFirst 10 rows from blueschists_test.xlsx:")
print(df2.head(10))

# if there are identical rows, show the first 5
if identical_count > 0:
    identical_rows_list = list(identical_rows)[:5]  # Take the first 5 entries
    identical_df = pd.DataFrame(identical_rows_list, columns=df1.columns)

    print("\nFirst 5 identical rows between both files:")
    print(identical_df)
else:
    print("\nNo identical rows found.")
