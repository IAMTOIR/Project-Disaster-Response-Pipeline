import sys
import pandas as pd 
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """
    Combines datasets from two filepaths after loading them.
    
    Parameters:
    messages_filepath: messages csv file
    categories_filepath: categories csv file
    
    Input: Messages and Categories datasets files
    Output: dataframe containing messages_filepath and categories_filepath merged
    """
    
    # Load datasets from filepaths
    messages = pd.read_csv('data/disaster_messages.csv')
    categories = pd.read_csv('data/disaster_categories.csv')
    
    # Merge datasets
    df = messages.merge(categories, on='id' ,how='left')
    
    return df


def clean_data(df):
    """
    Clean the dataframe
    
    Parameters:
    df: input dataframe
    
    Input: Dataframe following messages and categories merger
    Output: The cleaned-up dataframe
    """
    
    # create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(';', expand=True)
   
    # select the first row of the categories dataframe
    row = categories.head(1)
    
    # use this row to extract a list of new column names for categories.
    # one way is to apply a lambda function that takes everything 
    # up to the second to last character of each string with slicing
    category_colnames = row.apply(lambda x: x.str[:-2]).values.tolist()

    # rename the columns of `categories`
    categories.columns = category_colnames
    
    # convert category values to just numbers 0 or 1
    for column in categories:
    # set each value to be the last character of the string
        categories[column] = categories[column].str[-1]
    
    # convert column from string to numeric
        categories[column] = (pd.to_numeric(categories[column])>=1)*1


    # drop the original categories column from `df`
    df.drop(['categories'], axis=1, inplace = True)
   
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1)
  
    # drop duplicates
    df = df.drop_duplicates()
    
    return df


def save_data(df, database_filename):
    """
    Publish dataframes to SQLite databases
    """
    engine = create_engine(f'sqlite:///{database_filename}')
    df.to_sql('disaster_response_table', engine, index=False, if_exists = 'replace')


def main():
    """
    Data is loaded, cleaned, and saved to a database.
    """
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
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
