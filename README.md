# Ex.No: 1B                     CONVERSION OF NON STATIONARY TO STATIONARY DATA
# Date:01/11/2026

### AIM:
To develop a Python program to convert non-stationary time series data into stationary data using Regular Differencing, Seasonal Adjustment, and Log Transformation on the Tesla Vehicle Sales Performance dataset.
### Dataset Name:
tesla_vehicle_sales_performance.csv
### ALGORITHM:
1. Import the required libraries such as Pandas, NumPy, Matplotlib, and Statsmodels.
2. Read the Tesla Vehicle Sales Performance dataset using Pandas.
3. Convert the order_date column into datetime format and set it as the index.
4. Group the net_revenue_usd column by month to calculate total monthly revenue.
5. Perform Regular Differencing, Seasonal Adjustment, and Log Transformation.
6. Perform Log Transformation with Regular Differencing and Seasonal Adjustment.
7. Plot the original data and all transformed data using six subplots.
8. Display the graphs and observe the changes.
### PROGRAM:
```python

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

# Read the Tesla Vehicle Sales dataset
data = pd.read_csv('/content/tesla_vehicle_sales_performance.csv')

# Convert order_date into datetime format
data['order_date'] = pd.to_datetime(
    data['order_date'], format='mixed', errors='coerce'
)

data['net_revenue_usd'] = pd.to_numeric(
    data['net_revenue_usd'], errors='coerce'
)

data = data.dropna(subset=['order_date', 'net_revenue_usd'])
data = data.sort_values('order_date')

# Group net revenue by month
monthly_revenue = data.groupby(
    data['order_date'].dt.to_period('M')
)['net_revenue_usd'].sum()

# Set monthly date as index
monthly_revenue.index = monthly_revenue.index.to_timestamp()

# Regular Differencing
revenue_diff = monthly_revenue.diff()

# Seasonal Adjustment
result = seasonal_decompose(
    monthly_revenue,
    model='additive',
    period=12
)

revenue_seasonal_diff = result.resid

# Log Transformation
revenue_log = np.log(monthly_revenue)

# Log + Regular Differencing
revenue_log_diff = revenue_log.diff()

# Log + Regular + Seasonal Differencing
result_log = seasonal_decompose(
    revenue_log_diff.dropna(),
    model='additive',
    period=12
)

revenue_log_seasonal_diff = result_log.resid

# Plot all six graphs
plt.figure(figsize=(16, 16))

# Original Data
plt.subplot(6, 1, 1)
plt.plot(monthly_revenue, label='Tesla Monthly Revenue')
plt.legend(loc='best')
plt.title('Original Tesla Revenue Data')
plt.xlabel('Year')
plt.ylabel('Net Revenue (USD)')

# Regular Differencing
plt.subplot(6, 1, 2)
plt.plot(revenue_diff, label='revenue_diff')
plt.legend(loc='best')
plt.title('Regular Differencing')
plt.xlabel('Year')
plt.ylabel('Differenced Revenue')

# Seasonal Adjustment
plt.subplot(6, 1, 3)
plt.plot(revenue_seasonal_diff, label='revenue_seasonal_diff')
plt.legend(loc='best')
plt.title('Seasonal Adjustment')
plt.xlabel('Year')
plt.ylabel('Seasonally Adjusted Revenue')

# Log Transformation
plt.subplot(6, 1, 4)
plt.plot(revenue_log, label='revenue_log')
plt.legend(loc='best')
plt.title('Log Transformation')
plt.xlabel('Year')
plt.ylabel('Log Revenue')

# Log + Regular Differencing
plt.subplot(6, 1, 5)
plt.plot(revenue_log_diff, label='revenue_log_diff')
plt.legend(loc='best')
plt.title('Log Transformation + Regular Differencing')
plt.xlabel('Year')
plt.ylabel('Differenced Log Revenue')

# Log + Regular + Seasonal Differencing
plt.subplot(6, 1, 6)
plt.plot(
    revenue_log_seasonal_diff,
    label='revenue_log_seasonal_diff'
)
plt.legend(loc='best')
plt.title('Log + Regular Difference + Seasonal Adjustment')
plt.xlabel('Year')
plt.ylabel('Adjusted Log Revenue')

plt.tight_layout()
plt.show()

# Plot all transformed columns in one combined graph
combined_data = pd.DataFrame({
    'Original Revenue': monthly_revenue,
    'Regular Differencing': revenue_diff,
    'Seasonal Adjustment': revenue_seasonal_diff,
    'Log Transformation': revenue_log,
    'Log + Regular Differencing': revenue_log_diff,
    'Log + Regular + Seasonal Adjustment': revenue_log_seasonal_diff
})

combined_data.plot(
    kind='line',
    figsize=(14, 6),
    title='Tesla Vehicle Sales - Time Series Transformations'
)

plt.xlabel('Year')
plt.ylabel('Revenue / Transformed Values')
plt.legend(loc='best')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


```

### OUTPUT:


Original Tesla Revenue Data:

<img width="1302" height="222" alt="image" src="https://github.com/user-attachments/assets/3dbecc3c-e0f4-4b1c-9e7a-838d69f61079" />


Regular Differencing:

<img width="1304" height="226" alt="image" src="https://github.com/user-attachments/assets/668e7e0e-b08a-41c7-8cf6-62a70186d146" />


Seasonal Adjustment:

<img width="1306" height="232" alt="image" src="https://github.com/user-attachments/assets/edd84cd9-050b-446b-9064-8e1f2a380fbb" />


Log Transformation:

<img width="1305" height="217" alt="image" src="https://github.com/user-attachments/assets/aaf14911-e100-4102-b329-76fbc730ae5d" />


Log Transformation + Regular Differencing :

<img width="994" height="169" alt="image" src="https://github.com/user-attachments/assets/ed49c856-f464-4e77-8f6f-344d03d0d984" />

Log + Regular Difference + Seasonal Adjustment :

<img width="995" height="170" alt="image" src="https://github.com/user-attachments/assets/08712501-3288-48a1-bc61-f2f3bc79a828" />



Tesla Vehicle Sales-Time Series Transformations

<img width="993" height="428" alt="image" src="https://github.com/user-attachments/assets/0434b7fb-c05b-4101-9dfe-ad537df544b8" />





### RESULT:
Thus, the Python program for converting non-stationary time series data into stationary data using Regular Differencing, Seasonal Adjustment, and Log Transformation on the Tesla Vehicle Sales Performance dataset was successfully developed and executed.
