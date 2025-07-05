# Project-1: Sales Forecasting with Linear-Regression
Businesses often face challenges in estimating future sales. Accurate sales forecasting enables better inventory planning, marketing strategies, and resource management. This project aims to predict future sales using simple yet effective machine learning techniques
This project performs time series sales forecasting using linear regression on historical store sales data. It visualizes both actual and predicted sales trends, and forecasts sales for the next 30 days.

🧾 Dataset Overview
File: stores_sales_forecasting.csv
Key Column(s):

Order Date: Date of the sales transaction

Sales: Sales amount on that day

⚠️ Ensure the file is encoded in ISO-8859-1 and the column names match the code.

🧠 Project Objectives
Preprocess date-based sales data for regression modeling.

Use Linear Regression to learn from past sales trends.

Forecast future sales for 30 days.

Visualize both actual and predicted sales using Matplotlib.

🔧 Steps Performed
1. 📅 Date Handling
Converted Order Date to datetime format.

Aggregated total sales per day.

2. 🧮 Feature Engineering
Transformed dates into ordinal values for numerical modeling.

3. 🧠 Model Training
Used Linear Regression from sklearn.

Performed an 80-20 train-test split.

4. 📊 Visualization
Styled Matplotlib chart with dark theme.

Compared actual vs. predicted sales trends.

5. 🔮 Forecasting
Predicted sales for the next 30 days using the trained model.

Displayed forecasted results in tabular form.

📉 Sample Output (Forecast Preview)
Date	Forecasted Sales
2024-07-01	13450.77
2024-07-02	13498.32
...	...

🛠️ Tech Stack
Python 3

Pandas – data manipulation

NumPy – numerical operations

Matplotlib – visualization

Scikit-learn – linear regression model
