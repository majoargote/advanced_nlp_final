"""
Data setup for Part 3: Full data set 1%, 10%, 25%, 50%, 75%, 100%
"""
import pandas as pd


OUTPUT_DIR = "../data/"


def filter_data(df, categories: list):
    """Filter the dataframe to include only the specified categories."""
    filtered_df = df[df['category'].isin(categories)]
    return filtered_df

def cut_and_save_data(df, percentage):
    # Calculate the number of rows to sample based on the percentage
    num_rows = int(len(df) * (percentage / 100))
    
    # Randomly sample the specified percentage of rows
    sampled_df = df.sample(n=num_rows, random_state=42)
    
    # Save the sampled dataframe to a CSV file
    output_file = f"{OUTPUT_DIR}/filtered_reviews_{percentage}percent.csv"
    sampled_df.to_csv(output_file, index=False)
    print(f"Saved {percentage}% of data to {output_file}")
    return sampled_df


def quick_data_check(df):
    print("Dataset size:", len(df))
    print("Unique companies:", df['company'].nunique())
    print("Unique categories:", df['category'].nunique())
    print("===================================")
    print("Star rating distribution:\n", df.groupby('category')['stars'].value_counts())
    print("===================================")
    print("Sample of data:")
    print(df.info())


def main():
    df = pd.read_pickle(f"{OUTPUT_DIR}/trustpilot_reviews.pkl")
    categories = ["Travel & Vacation", "Media & Publishing"]
    filtered_df = filter_data(df, categories)
    for percentage in [1, 10, 25, 50, 75, 100]:
        print("==================================="*2)
        print(f"\nProcessing {percentage}% of data...")
        cut_df = cut_and_save_data(filtered_df, percentage)
        ## perform a quick data check on the filtered dataframe
        print("===================================")
        print(f"\nQuick data check for {percentage}% of data:")
        quick_data_check(cut_df)

if __name__ == "__main__":
    main()

