# **Disaster Response Pipeline Project**
**Link to GitHub Repository**

`https://github.com/IAMTOIR/Project-Disaster-Response-Pipeline`

## **Table of contents**

- [Project Descriptions](#project-descriptions)
- [File Structure](#file-structure)
- [Components](#components)
- [Usage](#usage)


## **Project Descriptions**

The Udacity Data Scientist Nanodegree program includes this project. So that we can respond quickly, it helps to organize the message of a crisis event.

## **File Structure**

~~~~~~~
workspace
    |-- app
        |-- templates
                |-- go.html
                |-- master.html
        |-- run.py
    |-- data
        |-- disaster_response.db
        |-- disaster_message.csv
        |-- disaster_categories.csv
        |-- process_data.py
    |-- models
        |-- classifier.pkl
        |-- train_classifier.py
    |-- notebooks
        |-- ETL Pipeline Preparation.ipynb
        |-- ML Pipeline Preparation.ipynb
    |-- README
~~~~~~~
### Components
There are three components I completed for this project. 

#### 1. ETL Pipeline
A Python script, `process_data.py`, writes a data cleaning pipeline that:

 - Loads the messages and categories datasets
 - Merges the two datasets
 - Cleans the data
 - Stores it in a SQLite database
 
A jupyter notebook `ETL Pipeline Preparation` was used to do EDA to prepare the process_data.py python script. 
 
#### 2. ML Pipeline
A Python script, `train_classifier.py`, writes a machine learning pipeline that:

 - Loads data from the SQLite database
 - Splits the dataset into training and test sets
 - Builds a text processing and machine learning pipeline
 - Trains and tunes a model using GridSearchCV
 - Outputs results on the test set
 - Exports the final model as a pickle file
 
A jupyter notebook `ML Pipeline Preparation` was used to do EDA to prepare the train_classifier.py python script. 

#### 3. Flask Web App
An emergency worker can enter a new message into the project's web interface and receive classification results in a number of categories. Additionally, data visualizations will be shown in the web application.


## **Usage**

1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/disaster_response.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/disaster_response.db models/classifier.pkl`

2. Go to `app` directory: `cd app`

3. Run your web app: `python run.py`

4. Click the `PREVIEW` button to open the homepage
