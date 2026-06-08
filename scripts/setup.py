
"""
Use: Run this script to load the dataset from Hugging Face, filter it for the two categories of interest (Travel & Vacation and Media & Publishing), and save the filtered data as a CSV file for use in the challenge.

Functions:

load_dataset_and_save_as_csv(): Purpose of this script is to pull the data from hugging face and save it as a csv file.

filter_and_save_data(df, category): Filter the data for the two categories we are interested in: Travel & Vacation and Media & Publishing. 
Keep only 16 labels ['stars'] for each category, and save the filtered/ partially unlabebeled data as a csv file.

quick_data_check(df): Quick data check to ensure the data is in the expected format and that we have only 16 labeled stars per category. 
Also prints out some basic statistics about the dataset.
"""


import pandas as pd
from datasets import load_dataset

def load_dataset_and_save_as_csv():
    # Load the dataset from Hugging Face
    ds = load_dataset("Kerassy/trustpilot-reviews-123k")

    # format data as pandas dataframe for easier EDA
    df = pd.DataFrame(ds["train"])
    df.head()

    #save data as pickle file for easier loading in future
    df.to_pickle("../data/trustpilot_reviews.pkl")

    return df


def filter_and_save_data(df, category):
    # Filter the data for the specified category
    filtered_df = df[df['category'] == category]

    ## randomly sample 16 rows per category
    sampled_df = filtered_df.sample(n=16, random_state=42)

    ## drop the sampled rows from the original filtered dataframe to create a partially unlabeled dataset for the challenge
    filtered_df = filtered_df.drop(sampled_df.index)
    ## null all star ratings to create partially unlabeled data for the challenge
    filtered_df['stars'] = None

    #concatenate the sampled_df with the filtered_df to create a combined dataframe for the challenge
    combined_df = pd.concat([sampled_df, filtered_df], ignore_index=True)

    # # Save the combined_df to a CSV file
    # combined_df.to_csv(output_file, index=False)
    return combined_df


def quick_data_check(df):
    print("Dataset size:", len(df))
    print("Unique companies:", df['company'].nunique())
    print("Unique categories:", df['category'].nunique())
    print("===================================")
    print("Star rating distribution:\n", df.groupby('category')['stars'].value_counts())
    #ensure only 16 labeled stars per category
    print("===================================")
    print("Labeled stars per category: Ensuring only 16 labeled stars per category")
    print(df[df['stars'].notnull()].groupby('category')['stars'].count()) 
    print("===================================")
    print("Sample of data:")
    print(df.info())

def main():
    df = load_dataset_and_save_as_csv()
    filtered_travel = filter_and_save_data(df, "Travel & Vacation")
    filtered_media = filter_and_save_data(df, "Media & Publishing")
    one_dataframe = pd.concat([filtered_travel, filtered_media], ignore_index=True)
    # Save the combined dataframe to a CSV file
    one_dataframe.to_csv("../data/filtered_reviews.csv", index=False)
    quick_data_check(one_dataframe)



if __name__ == "__main__":
    main()

