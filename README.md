Wine Quality Dataset

1. Dataset Description
The Wine Quality dataset contains information about the physicochemical
properties of white wine samples and their corresponding quality
ratings. It can be used to study the relationship between wine
characteristics and wine quality.

2. Dataset Information
- File name: winequality.csv
- Number of records: 4,898
- Number of features: 11 input features
- Target variable: quality
- Total columns: 12
- File format: CSV
- Missing values: No missing values were found during inspection.

3. Objective
The main objective of this dataset is to analyse the chemical properties
of white wine and understand how these properties are related to wine
quality. It can also be used to build machine learning models that
predict the quality rating of wine.

4. Features Description
  Feature                  Description
  fixed acidity          Fixed acids present in the wine
volatile acidity       Volatile acids present in the wine
citric acid            Amount of citric acid
residual sugar         Sugar remaining after fermentation
chlorides              Amount of chlorides
free sulfur dioxide    Free sulfur dioxide present in the wine
total sulfur dioxide   Total sulfur dioxide present in the wine
density                Density of the wine
pH                     Acidity or alkalinity level
sulphates              Amount of sulphates
alcohol                Alcohol content
quality                Quality rating of the wine (target variable)

5. Target Variable
The target variable is quality. It represents the quality score
assigned to each wine sample. In this dataset, the observed ratings
range from 3 to 9.

6. Dataset Structure
The dataset contains 4,898 rows and 12 columns. Each row represents one
white wine sample, and each column represents a measured property or its
quality score.

7. Possible Applications
- Exploratory Data Analysis (EDA)
- Data visualisation
- Correlation analysis
- Classification and regression
- Wine quality prediction using machine learning

8. Data Preprocessing
The CSV file uses a semicolon (;) as its delimiter. When loading the
dataset with Pandas, use:
import pandas as pd

df = pd.read_csv("winequality.csv", sep=";")
Check the data types, distributions and feature relationships before
building a model.

9. Conclusion
The Wine Quality dataset is useful for understanding how physicochemical
properties are associated with white wine quality. It provides a
practical dataset for data analysis and machine learning experiments
