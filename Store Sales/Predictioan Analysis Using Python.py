import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# 1) Load data from Sheet6
file_path = 'SuperStore5.xlsx'
df = pd.read_excel(
    file_path,
    sheet_name='Sheet6',
    engine='openpyxl'
)
original_rows = df.shape[0]


# 2) Preprocess
df['Order Date'] = pd.to_datetime(df['Order Date'], errors='coerce')
df = df.dropna(subset=['Order Date', 'Sales'])
df['Year'] = df['Order Date'].dt.year

print("---------------")
cleaned_rows = df.shape[0]
dropped_rows = original_rows - cleaned_rows
print(f"Original rows: {original_rows}, Cleaned rows: {cleaned_rows}, Dropped rows: {dropped_rows}")
print("---------------")

# 3) Aggregate yearly sales
yearly_sales = (
    df
    .groupby('Year')['Sales']
    .sum()
    .reset_index()
)
print(yearly_sales)


# 4) Prepare data for modeling
X = pd.DataFrame(yearly_sales['Year'])
y = yearly_sales['Sales']

# 5) Fit Linear Regression model
model = LinearRegression().fit(X, y)

# 6) Forecast next 3 years
last_year = yearly_sales['Year'].max()
future_years = np.arange(last_year + 1, last_year + 4).reshape(-1, 1)
future_preds = model.predict(future_years)

# 7) Plot Actual vs. Forecast
plt.figure(figsize=(10,6))
plt.plot(yearly_sales['Year'], yearly_sales['Sales'], 'o-', label='Actual Sales')
plt.plot(future_years.flatten(), future_preds, 'o--', label='Forecasted Sales')
for yr, val in zip(future_years.flatten(), future_preds):
    plt.text(yr, val, f"{val:,.0f}", ha='center', va='bottom', fontsize=9, color='red')
plt.title("Yearly Sales Forecast (Actual + Next 3 Years)")
plt.xlabel("Year")
plt.ylabel("Sales Amount")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# 8) Year-over-Year Growth (%)
yearly_sales['GrowthPct'] = yearly_sales['Sales'].pct_change() * 100
plt.figure(figsize=(8,5))
plt.bar(yearly_sales['Year'], yearly_sales['GrowthPct'])
plt.title("Year-over-Year Sales Growth (%)")
plt.xlabel("Year")
plt.ylabel("Growth Rate (%)")
plt.grid(axis='y')
plt.tight_layout()
plt.show()

# 9) Sales by Region per Year
if 'Region' in df.columns:
    region_year_sales = df.groupby(['Year', 'Region'])['Sales'].sum().unstack()
    region_year_sales.plot(kind='bar', stacked=True, figsize=(10,6))
    plt.title("Sales per Region by Year")
    plt.xlabel("Year")
    plt.ylabel("Sales Amount")
    plt.legend(title="Region")
    plt.tight_layout()
    plt.grid(axis='y')
    plt.show()
else:
    print("No 'Region' column found in dataset.")


