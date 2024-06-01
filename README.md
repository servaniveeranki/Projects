# Projects
# My small projects on Data Science and Analytics

EDA_PROJECT:
# Data Science Job Salaries Analysis

## Objective
This project aims to enhance skills in data manipulation, cleaning, transformation, and visualization using libraries like numpy, pandas, matplotlib, seaborn, and others. Additionally, a comprehensive report is generated using pandas_profiling.

## Dataset
The dataset used in this project is the Data Science Job Salaries dataset, which includes the following columns:
- work_year
- experience_level
- employment_type
- job_title
- salary
- salary_currency
- salary_in_usd
- employee_residence
- remote_ratio
- company_location
- company_size

## Tasks and Questions

### Task 1: Data Loading and Initial Exploration
- Loaded the dataset using pandas.
- Displayed the first 10 rows of the dataset.
- Checked for missing values and data types of each column.

**Question 1:** How many missing values are there in each column?

### Task 2: Data Cleaning and Transformation
- Handled missing values by filling with median values.
- Converted appropriate columns to categorical data types.

**Question 2:** What transformations did you apply to handle missing values and type conversions?

### Task 3: Descriptive Analysis
- Calculated descriptive statistics for numerical columns.
- Performed group-by operations to find meaningful insights.

**Question 3:** What are the mean, median, and standard deviation of the numerical columns? Provide insights from group-by operations.

### Task 4: Data Visualization (EDA)
- Created visualizations using matplotlib and seaborn.
- Analyzed yearly trends of jobs.
- Plotted salary distribution and employee residency trends.
- Examined company size trends and remote work trends.
- Generated a comprehensive report using pandas_profiling.

**Question 4:** Provide and interpret the visualizations you created. What insights do they offer about the data?

## Visualizations
- Yearly Trend of Jobs: Shows the distribution of job postings over different years.
- Salary Distribution: Highlights the distribution and spread of salaries.
- Salary Box Plot: Displays the salary distribution with outliers.
- Employee Residency Trends: Shows the distribution of employee residences.
- Company Size Trends: Visualizes the distribution of company sizes.
- Remote Work Trends: Displays the trend of remote work adoption.

## Report
A comprehensive profiling report has been generated and saved as `data_science_job_salaries_report.html`.

## Requirements
- pandas
- numpy
- matplotlib
- seaborn
- pandas_profiling

## Usage
1. Ensure all required libraries are installed.
2. Load the dataset and run the analysis as shown in the provided code.
3. Open the `data_science_job_salaries_report.html` to view the detailed profiling report.

## Conclusion
This project provided insights into the Data Science Job Salaries dataset, including salary trends, job distributions, and company characteristics. The visualizations and report generated offer a comprehensive understanding of the data.

-----------------------------------------------------------------------------------------------------------------------------------------------------------
ML_DS_PROJECT
# Crop Recommendation Model

## Objective
This project aims to build a machine learning model to recommend crops based on various features. The dataset used for this project contains several features such as N, P, K, temperature, humidity, pH, and rainfall, and the target variable is the crop label.

## Dataset
The dataset includes the following columns:
- N: Ratio of Nitrogen content in soil
- P: Ratio of Phosphorous content in soil
- K: Ratio of Potassium content in soil
- temperature: Temperature in degree Celsius
- humidity: Relative humidity in %
- pH: pH value of the soil
- rainfall: Rainfall in mm
- Label: Type of crop

## Steps Performed

### 1. Data Loading and Initial Exploration
- Loaded the dataset using pandas.
- Displayed the first 10 rows of the dataset.
- Checked for missing values and data types of each column.

### 2. Exploratory Data Analysis (EDA)
- Displayed the distribution of the target variable.
- Used pairplots to visualize relationships between features.
- Created a correlation heatmap to identify feature correlations.

### 3. Data Preprocessing
- Split the data into features and target variable.
- Split the data into training and testing sets.
- Standardized the features using StandardScaler.

### 4. Model Training and Evaluation
- Trained multiple models: Logistic Regression, Naive Bayes, Support Vector Machine, K-Nearest Neighbors, Decision Tree, Random Forest, Bagging, and Gradient Boosting.
- Evaluated the models using accuracy score.
- Selected the best model based on accuracy.

### 5. Model Saving
- Saved the best model to a file named `model.pkl`.

### 6. Model Usage Demonstration
- Loaded the saved model.
- Demonstrated making predictions on the test set using the loaded model.

## Best Model
The best model was selected based on the highest accuracy score.

## Usage

1. **Load the Dataset**:
    ```python
    import pandas as pd
    df = pd.read_csv('crop_recommendation.csv')
    ```

2. **Preprocess the Data**:
    ```python
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    X = df.drop(columns=['Label'])
    y = df['Label']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    ```

3. **Load the Model**:
    ```python
    import pickle

    with open('model.pkl', 'rb') as file:
        loaded_model = pickle.load(file)
    ```

4. **Make Predictions**:
    ```python
    y_pred = loaded_model.predict(X_test_scaled)
    ```

5. **Evaluate the Model**:
    ```python
    from sklearn.metrics import accuracy_score

    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy:.4f}')
    ```

## Requirements
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- pickle

## Conclusion
This project provided a comprehensive solution for recommending crops based on various soil and weather features. The best model was saved for future predictions, and its usage was demonstrated with the test dataset.



