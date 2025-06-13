#!/usr/bin/env python
# coding: utf-8

# In[21]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Load your CSV (change path if needed)
df=pd.read_csv("C:\\Users\\ANSHIKA\\Downloads\\stores_sales_forecasting.csv", encoding='ISO-8859-1')
print(df.head())

df['Order Date']=pd.to_datetime(df['Order Date'])

# Group by date and sum Sales
df_sales = df.groupby('Order Date')['Sales'].sum().reset_index()

# Convert date to numeric (ordinal) for regression
df_sales['Date_Ordinal'] = df_sales['Order Date'].map(pd.Timestamp.toordinal)

# Prepare features and target
X = df_sales[['Date_Ordinal']]
y = df_sales['Sales']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Apply Linear Regression
model = LinearRegression()
model.fit(X_train, y_train)

# Predict on full data
df_sales['Predicted Sales'] = model.predict(X)

# Plot actual vs predicted
# Enhanced Plot: Dark background and styled lines
plt.style.use('dark_background')
plt.figure(figsize=(14, 7))

# Plot actual sales
plt.plot(df_sales['Order Date'], df_sales['Sales'], label='Actual Sales',
         color='#00FFFF', linewidth=2.5, marker='o', markersize=4)

# Plot predicted sales
plt.plot(df_sales['Order Date'], df_sales['Predicted Sales'], label='Predicted Sales',
         color='#FF4500', linewidth=2.5, linestyle='--', marker='x', markersize=4)

# Customize the plot
plt.title('Actual vs Predicted Sales Over Time', fontsize=16, fontweight='bold', color='white')
plt.xlabel('Date', fontsize=20, color='white')
plt.ylabel('Sales', fontsize=20, color='white')
plt.legend(facecolor='black', edgecolor='white', fontsize=12)
plt.tight_layout()

# Show the plot
plt.show()

# Forecast future sales (next 30 days)
future_dates = pd.date_range(df_sales['Order Date'].max() + pd.Timedelta(days=1), periods=30)
future_ordinal = future_dates.map(pd.Timestamp.toordinal).values.reshape(-1, 1)
future_df = pd.DataFrame(future_ordinal, columns=['Date_Ordinal'])  # Match training column name
future_sales = model.predict(future_df)


# Display forecast
forecast_df = pd.DataFrame({
    'Date': future_dates,
    'Forecasted Sales': future_sales
})

print(forecast_df.head(10))


# In[ ]:





# In[ ]:




