import pandas as pd
import numpy as np
from tqdm import tqdm
import Levenshtein

def extract_data_from_file():
    print("\nStarting data extraction...")

    # File paths
    polluters_user_file = "Datasets/content_polluters.txt"
    polluters_followings_file = "Datasets/content_polluters_followings.txt"
    polluters_tweets_file = "Datasets/content_polluters_tweets.txt"

    legitimate_user_file = "Datasets/legitimate_users.txt"
    legitimate_followings_file = "Datasets/legitimate_users_followings.txt"
    legitimate_tweets_file = "Datasets/legitimate_users_tweets.txt"

    # Extract data for polluters and legitimate users
    polluters_data = extract_data(polluters_user_file, polluters_followings_file, polluters_tweets_file, "polluter", 1)
    legitimate_data = extract_data(legitimate_user_file, legitimate_followings_file, legitimate_tweets_file, "legitimate", 0)

    # Concatenate both datasets
    final_df = pd.concat([polluters_data, legitimate_data], ignore_index=True)

    print("Data extraction completed!\n")
    return final_df

def extract_data(user_file, followings_file, tweets_file, source, class_label): 
    print(f"Extracting data for {source} users...")
    
    # Read the user file (polluters or legitimate users)
    user_columns = ["user_id", "created_at", "collected_at", "num_followings", "num_followers",
                    "num_tweets", "length_screen_name", "length_description"]
    user_df = pd.read_csv(user_file, sep="\t", names=user_columns)

    # Read the followings file
    followings_df = pd.read_csv(followings_file, sep="\t", names=["user_id", "followings"])
    followings_df["followings"] = followings_df["followings"].fillna("").astype(str)

    # Read the tweets file
    tweets_df = pd.read_csv(tweets_file, sep="\t", names=["user_id", "tweet_id", "tweet", "tweet_created_at"])

    # Convert timestamps to datetime format
    tweets_df["tweet_created_at"] = pd.to_datetime(tweets_df["tweet_created_at"], errors="coerce")

    # Aggregate tweets per user
    tweets_grouped = tweets_df.groupby("user_id")["tweet"].apply(list).reset_index()

    # Merge all dataframes on user_id
    merged_df = pd.merge(user_df, followings_df, on="user_id", how="left")
    merged_df = pd.merge(merged_df, tweets_grouped, on="user_id", how="left")

    # Add numerical class column
    merged_df["class"] = class_label

    print(f"Extraction complete for {source} users.")
    return merged_df

def calculate_min_levenshtein_distance(df):
    print("\nCalculating min Levenshtein distances...")

    def min_distance(tweets):
        """Calculates the minimum Levenshtein distance between tweets of a user."""
        if not isinstance(tweets, list) or len(tweets) < 2:
            return np.nan  # Not enough tweets to compute distance
        
        tweets = [str(t) for t in tweets if isinstance(t, str)]  # Ensure all tweets are strings
        if len(tweets) < 2:
            return np.nan  # If filtering results in <2 tweets, return NaN
        
        min_dist = float("inf")
        for i in range(len(tweets)):
            for j in range(i + 1, len(tweets)):
                dist = Levenshtein.distance(tweets[i], tweets[j])
                min_dist = min(min_dist, dist)
        
        return min_dist

    tqdm.pandas()
    df["tweet"] = df["tweet"].apply(lambda x: x if isinstance(x, list) else [])  # Ensure tweets are lists
    df["min_levenshteinDistance"] = df["tweet"].progress_apply(min_distance)

    print("Levenshtein distance calculation complete!\n")
    return df

def add_number_of_followings_variance(df):
    print("\nCalculating variance of followings...")

    # Convert followings from comma-separated string to a list of integers
    df["followings_list"] = df["followings"].apply(lambda x: list(map(int, x.split(','))) if isinstance(x, str) and x else [])

    # Compute variance for each user
    df["variance_of_followings"] = df["followings_list"].progress_apply(lambda x: np.var(x) if len(x) > 1 else np.nan)

    # Drop intermediate list column
    df.drop(columns=["followings_list"], inplace=True)

    print("Variance of followings calculation complete!\n")
    return df

def clean_data(df):
    print("\nCleaning data (dropping missing values)...")
    df = df.dropna()
    print("Data cleaning complete!\n")
    return df

def save_to_csv(df, output_file):
    print(f"\nSaving merged data to {output_file}...")

    # Ensure 'class' is the last column
    cols = [col for col in df.columns if col != "class"] + ["class"]
    df = df[cols]  # Reorder columns

    df.to_csv(output_file, index=False)
    print(f"Data saved successfully to {output_file}!\n")


# Enable tqdm for pandas operations
tqdm.pandas()

# Extract data
df = extract_data_from_file()

# Clean data
df = clean_data(df)

# Add the variance of followings
df = add_number_of_followings_variance(df)

# Add the "min_levenshteinDistance" field
df = calculate_min_levenshtein_distance(df) # this one takes around 30 min!

# Save final dataset
save_to_csv(df, "preprocessed_data.csv")
