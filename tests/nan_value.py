
import pandas as pd

# Load dataset (modify the path accordingly)
df = pd.read_csv("preprocessed_data.csv")

# Count NaN rows per class
nan_per_class = df[df.isna().any(axis=1)].groupby("class").size()

# Count total rows per class
total_per_class = df["class"].value_counts()

# Calculate percentage of NaN rows per class
nan_percentage_per_class = (nan_per_class / total_per_class) * 100

# Combine results into a DataFrame
nan_class_summary = pd.DataFrame({
    "Total Rows": total_per_class,
    "NaN Rows": nan_per_class,
    "Percentage (%)": nan_percentage_per_class
}).fillna(0)  # Fill NaN with 0 for classes that have no missing values

# Add the total row at the bottom
nan_class_summary.loc["Total"] = nan_class_summary.sum()
nan_class_summary.loc["Total", "Percentage (%)"] = (nan_class_summary.loc["Total", "NaN Rows"] / nan_class_summary.loc["Total", "Total Rows"]) * 100

# Print results
print("NaN distribution per class:")
print(nan_class_summary)
