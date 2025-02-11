import pandas as pd

# Load the CSV file
df = pd.read_csv("preprocessed_data.csv")

# print total rows 
print(len(df), " rows")

# print the name columns 
print(df.columns)

# Print 10 random rows
print(df.sample(20))
# print(df.sample(10, random_state=42)) 


