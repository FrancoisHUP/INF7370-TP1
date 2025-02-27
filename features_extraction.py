import pandas as pd
import numpy as np
from tqdm import tqdm
import Levenshtein
from scipy.stats import zscore 
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

    user_df["created_at"] = pd.to_datetime(user_df["created_at"], errors="coerce")
    user_df["collected_at"] = pd.to_datetime(user_df["collected_at"], errors="coerce")

    # Read the followings file
    followings_df = pd.read_csv(followings_file, sep="\t", names=["user_id", "followings"])
    followings_df["followings"] = followings_df["followings"].fillna("").astype(str)

    # Read the tweets file
    tweets_df = pd.read_csv(tweets_file, sep="\t", names=["user_id", "tweet_id", "tweet", "tweet_created_at"])
    # Convert timestamps to datetime format
    tweets_df["tweet_created_at"] = pd.to_datetime(tweets_df["tweet_created_at"], errors="coerce")
    
    # Dataframe to calculate stats from tweets
    tweets_stats = tweets_df.groupby("user_id").agg({"tweet": list, "tweet_created_at": list}).reset_index()
    print(tweets_stats.dtypes)
    print(tweets_stats.head(10))

    # Calculate mentions (@)
    tweets_stats = calculate_mentions(tweets_stats)

    # Calculate URLs (http)
    tweets_stats = calculate_url(tweets_stats)

    # Calculate average and max time between tweets
    tweets_stats[["avg_time_between_tweets", "max_time_between_tweets"]] = tweets_stats["tweet_created_at"].apply(calculate_mean_and_max_time)

    print("tweets stats")
    print(tweets_stats.dtypes)
    print(tweets_stats.head(10))

    # Merge all dataframes on user_id
    merged_df = pd.merge(user_df, followings_df, on="user_id", how="left")
    merged_df = pd.merge(merged_df, tweets_stats, on="user_id", how="left")

    # Add numerical class column
    merged_df["class"] = class_label

    print(f"Extraction complete for {source} users.")
    return merged_df

def calculate_mean_and_max_time(timestamps):
        valid_times = [pd.to_datetime(t) for t in timestamps if pd.notna(t)]
        if len(valid_times) < 2:
            return pd.Series([np.nan, np.nan])
        valid_times.sort()
        diffs = [(valid_times[i] - valid_times[i - 1]).total_seconds() for i in range(1, len(valid_times))]
        return pd.Series([np.mean(diffs), max(diffs)])

def compute_normalized_levenshtein(tweet1, tweet2):
    """Computes Levenshtein distance normalized by max length of tweets."""
    if tweet1 == tweet2:
        return 0.0  # Identical tweets
    
    dist = Levenshtein.distance(tweet1, tweet2)
    max_len = max(len(tweet1), len(tweet2))
    return dist / max_len if max_len > 0 else 0  # Avoid division by zero

def calculate_avg_normalized_levenshtein_distance(df):
    print("\nCalculating average normalized Levenshtein distances...")

    def avg_normalized_distance(tweets):
        """Calculates the average normalized Levenshtein distance between tweets of a user."""
        if not isinstance(tweets, list) or len(tweets) < 2:
            return np.nan  # Not enough tweets to compute distance
        
        tweets = [str(t) for t in tweets if isinstance(t, str)]  # Ensure all tweets are strings
        if len(tweets) < 2:
            return np.nan  # If filtering results in <2 tweets, return NaN
        
        distances = []

        for i in range(len(tweets)):
            for j in range(i + 1, len(tweets)):
                norm_dist = compute_normalized_levenshtein(tweets[i], tweets[j])
                distances.append(norm_dist)

        return np.mean(distances) if distances else np.nan  # Compute average

    tqdm.pandas()
    df["tweet"] = df["tweet"].apply(lambda x: x if isinstance(x, list) else [])  # Ensure tweets are lists
    df["avg_norm_levenshtein_distance"] = df["tweet"].progress_apply(avg_normalized_distance)

    print("Levenshtein distance calculation complete!\n")
    return df

def calculate_account_lifetime(df):
    print("Calculating account lifetime...")
    df["account_lifetime_days"] = (df["collected_at"] - df["created_at"]).dt.days
    return df

def calculate_mentions(df):
    print("\nCounting @ mentions in tweets...")

    df["tweet"] = df["tweet"].apply(lambda x: x if isinstance(x, list) else [])

    df["num_mentions"] = df["tweet"].apply(lambda tweets: sum(str(tweet).count("@") for tweet in tweets if isinstance(tweet, str)))

    print("Counting @ mentions complete!\n")
    return df

def calculate_url(df):
    print("\nCounting URLs in tweets...")

    df["tweet"] = df["tweet"].apply(lambda x: x if isinstance(x, list) else [])

    df["num_urls"] = df["tweet"].apply(lambda tweets: sum(tweet.count("http") for tweet in tweets if isinstance(tweet, str)))

    print("Counting URLs complete!\n")
    return df

def calculate_mentions_urls_ratio(df):
    df["mentions_ratio"] = df.apply(lambda row: row["num_mentions"] / row["num_tweets"] if row["num_tweets"] > 0 else np.nan, axis=1)
    df["url_ratio"] = df.apply(lambda row: row["num_urls"] / row["num_tweets"] if row["num_tweets"] > 0 else np.nan, axis=1)
    return df

def calculate_following_follower_ratio(df):
    print("\nCalculating following/follower ratio...")

    df["following_follower_ratio"] = df.apply(
        lambda row: row["num_followings"] / row["num_followers"]
        if row["num_followers"] > 0 else np.nan, axis=1
    )

    print("Following/Follower ratio calculation complete!\n")
    return df


def calculate_tweets_per_day(df):
    df["num_tweets"] = pd.to_numeric(df["num_tweets"], errors="coerce")
    
    df["tweets_per_day"] = df.apply(
        lambda row: row["num_tweets"] / row["account_lifetime_days"]
        if row["account_lifetime_days"] > 0 else np.nan, axis=1
    )

    print("Tweets per day calculation complete!\n")
    return df

def calculate_avg_and_max_time_between_tweets(df):
    
    df["tweet"] = df["tweet"].apply(lambda x: x if isinstance(x, list) else [])

    def avg_time_between_tweets(tweets):
        timestamps = [pd.to_datetime(t) for t in tweets if pd.notna(t)] 
        timestamps.sort()  

        if len(timestamps) < 2:
            return np.nan  

        diffs = [(timestamps[i] - timestamps[i - 1]).total_seconds() for i in range(1, len(timestamps))]
        return np.mean(diffs) if diffs else np.nan 
    
    def max_time_between_tweets(tweets):
        timestamps = [pd.to_datetime(t) for t in tweets if pd.notna(t)]  
        timestamps.sort() 

        if len(timestamps) < 2:
            return np.nan  

        diffs = [(timestamps[i] - timestamps[i - 1]).total_seconds() for i in range(1, len(timestamps))]
        return max(diffs) if diffs else np.nan  
    
    print("\nCalculating average time between tweets...")
    df["avg_time_between_tweets"] = df["tweet"].apply(avg_time_between_tweets)
    print("Average time between tweets calculation complete!\n")
    print("\nCalculating maximum time between tweets...")
    df["max_time_between_tweets"] = df["tweet"].apply(max_time_between_tweets)
    print("Maximum time between tweets calculation complete!\n")
    return df

def calculate_z_score(df):
    print("\nCalculating Z-score for similarity...")

    # Ensure avg_norm_levenshtein_distance is numeric
    df["avg_norm_levenshtein_distance"] = pd.to_numeric(df["avg_norm_levenshtein_distance"], errors="coerce")

    # Compute Z-score only for valid rows (ignore NaN values)
    valid_rows = df["avg_norm_levenshtein_distance"].notna()
    df.loc[valid_rows, "z_score_similarity"] = zscore(df.loc[valid_rows, "avg_norm_levenshtein_distance"])

    print("Z-score calculation complete!\n")
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

def remove_uneccessary_columns(df):
    print("\nRemoving all but relevant columns...")
    df = df.drop(columns=["user_id", "created_at", "collected_at", "num_tweets", "tweet", "tweet_created_at", "num_mentions", "num_urls"])
    print("All but relevant columns removed!\n")

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

# Compute average normalized Levenshtein distance
df = calculate_avg_normalized_levenshtein_distance(df)

# Compute Z-score for similarity comparison
df = calculate_z_score(df)

# Calculate account lifetime
df = calculate_account_lifetime(df)

# Calculate Following/Follower ratio
df = calculate_following_follower_ratio(df)

# Calculate number of tweets per day
df = calculate_tweets_per_day(df)

# Calculate the ratio of mentions and URLS
df = calculate_mentions_urls_ratio(df)

# Removing all but necessary columns 
df = remove_uneccessary_columns(df)

print(df.dtypes)
print(df.head(10))
# Save final dataset
save_to_csv(df, "preprocessed_data.csv")
