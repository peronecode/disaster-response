import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """
    Load and merge messages and categories datasets.
    
    Args:
    messages_filepath: str, path to messages CSV file.
    categories_filepath: str, path to categories CSV file.
    
    Returns:
    df: DataFrame, merged messages and categories datasets.
    """
    df_messages = pd.read_csv(messages_filepath)
    df_categories = pd.read_csv(categories_filepath)
    df = pd.merge(df_messages, df_categories, on='id')
    
    return df


def split_categories(df):
    """
    Split categories into separate category columns.
    
    Args:
    df: DataFrame, DataFrame containing the 'categories' column to be split.
    
    Returns:
    categories: DataFrame, DataFrame with each category as a separate column.
    """
    categories = df['categories'].str.split(';', expand=True)
    row = categories.iloc[0]
    category_colnames = [cn.split('-')[0] for cn in row.unique()]
    categories.columns = category_colnames
    
    return categories


def convert_category_values(categories):
    """
    Convert category values to binary (0 or 1).
    
    Args:
    categories: DataFrame, DataFrame with the category columns to be converted.
    
    Returns:
    categories: DataFrame, DataFrame with binary category values.
    """
    for column in categories:
        categories[column] = categories[column].apply(lambda v: v.split('-')[1])
        categories[column] = categories[column].astype(int)
        
    return categories


def clean_data(df, categories):
    """
    Clean dataframe by removing duplicates and merging categories.
    
    Args:
    df: DataFrame, original DataFrame containing the 'categories' column.
    categories: DataFrame, DataFrame with binary category values.
    
    Returns:
    df: DataFrame, cleaned DataFrame with merged categories and no duplicates.
    """
    df = df.drop(['categories'], axis=1)
    df = pd.merge(df, categories, left_index=True, right_index=True)
    df = df.drop_duplicates()
    
    return df


def save_data(df, database_filename):
    """
    Save cleaned data to an SQLite database.
    
    Args:
    df: DataFrame, cleaned DataFrame to be saved.
    database_filename: str, filename of the SQLite database.
    """
    engine = create_engine(f'sqlite:///{database_filename}')
    df.to_sql('Messages', engine, if_exists='replace')


def main():
    if len(sys.argv) == 4:
        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print(f'Loading data...\n    MESSAGES: {messages_filepath}\n    CATEGORIES: {categories_filepath}')
        df = load_data(messages_filepath, categories_filepath)

        print('Splitting categories...')
        categories = split_categories(df)

        print('Converting category values...')
        categories = convert_category_values(categories)

        print('Cleaning data...')
        df = clean_data(df, categories)
        
        print(f'Saving data...\n    DATABASE: {database_filepath}')
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()