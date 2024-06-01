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

