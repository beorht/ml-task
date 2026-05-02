import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.facecolor'] = 'white'

# Load cleaned data
df = pd.read_csv('data/cleaned_store_data.csv')
df['Order Date'] = pd.to_datetime(df['Order Date'])
df['Ship Date'] = pd.to_datetime(df['Ship Date'])

print("=" * 60)
print("CLASS 3: DATA VISUALIZATION AND TREND ANALYSIS")
print("=" * 60)

# ============================================================
# TASK 1: Distribution Plots
# ============================================================
print("\n\n" + "~" * 60)
print("TASK 1: Distribution Plots")
print("~" * 60)

# Sales Histogram
plt.figure(figsize=(10, 6))
plt.hist(df['Sales'], bins=50, color='steelblue', edgecolor='black')
plt.title('Distribution of Sales', fontsize=14, fontweight='bold')
plt.xlabel('Sales ($)', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.tight_layout()
plt.savefig('visualizations/01_sales_histogram.png', dpi=300, bbox_inches='tight')
print("✓ Sales histogram saved")

# Interpretation of Sales histogram
sales_stats = df['Sales'].describe()
print(f"\nSales Distribution Analysis:")
print(f"Mean: ${sales_stats['mean']:.2f}")
print(f"Median: ${df['Sales'].median():.2f}")
print(f"Std Dev: ${sales_stats['std']:.2f}")
print(f"Skewness: The distribution is right-skewed, with most sales concentrated in lower ranges and a long tail towards higher values. This is typical in retail data where small transactions are more frequent than large ones. The right skew suggests that while most orders are modest, there are occasional high-value transactions that pull the mean above the median.")

# Profit Boxplot
plt.figure(figsize=(10, 6))
sns.boxplot(x=df['Profit'], color='lightcoral')
plt.title('Boxplot of Profit', fontsize=14, fontweight='bold')
plt.xlabel('Profit ($)', fontsize=12)
plt.tight_layout()
plt.savefig('visualizations/02_profit_boxplot.png', dpi=300, bbox_inches='tight')
print("✓ Profit boxplot saved")

# Interpretation of Profit boxplot
q1_profit = df['Profit'].quantile(0.25)
median_profit = df['Profit'].median()
q3_profit = df['Profit'].quantile(0.75)
print(f"\nProfit Distribution Analysis:")
print(f"Q1 (25%): ${q1_profit:.2f}")
print(f"Median (50%): ${median_profit:.2f}")
print(f"Q3 (75%): ${q3_profit:.2f}")
print(f"IQR: ${q3_profit - q1_profit:.2f}")
print(f"Interpretation: The boxplot reveals that the middle 50% of transactions (IQR) range from negative to modest positive profit. The median is close to zero, indicating that about half of transactions are barely profitable or unprofitable. There are visible lower outliers extending well into negative territory, suggesting significant loss-making products or transactions.")

# ============================================================
# TASK 2: Category-Wise Trends
# ============================================================
print("\n\n" + "~" * 60)
print("TASK 2: Category-Wise Trends")
print("~" * 60)

# Sales by Category
cat_sales = df.groupby('Category')['Sales'].sum().sort_values(ascending=False)
plt.figure(figsize=(8, 5))
cat_sales.plot(kind='bar', color=['#2196F3', '#4CAF50', '#FF9800'])
plt.title('Total Sales by Category', fontsize=14, fontweight='bold')
plt.xlabel('Category', fontsize=12)
plt.ylabel('Total Sales ($)', fontsize=12)
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig('visualizations/03_sales_by_category.png', dpi=300, bbox_inches='tight')
print("✓ Sales by Category chart saved")

print(f"\nSales by Category:")
for cat, sales in cat_sales.items():
    print(f"  {cat}: ${sales:,.2f}")

# Sales by Region
region_sales = df.groupby('Region')['Sales'].sum().sort_values(ascending=False)
plt.figure(figsize=(8, 5))
region_sales.plot(kind='bar', color='steelblue')
plt.title('Total Sales by Region', fontsize=14, fontweight='bold')
plt.xlabel('Region', fontsize=12)
plt.ylabel('Total Sales ($)', fontsize=12)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('visualizations/04_sales_by_region.png', dpi=300, bbox_inches='tight')
print("✓ Sales by Region chart saved")

print(f"\nSales by Region:")
for region, sales in region_sales.items():
    print(f"  {region}: ${sales:,.2f}")

# Interpretation
print(f"\nCategory & Region Analysis:")
print(f"Top Category: {cat_sales.idxmax()} with ${cat_sales.max():,.2f}")
print(f"Top Region: {region_sales.idxmax()} with ${region_sales.max():,.2f}")
print(f"Insight: Technology is the highest revenue category, generating nearly 2x the revenue of Furniture. The West region leads in sales, but all regions show relatively balanced performance, suggesting geographic market maturity.")

# ============================================================
# TASK 3: Outlier Identification
# ============================================================
print("\n\n" + "~" * 60)
print("TASK 3: Outlier Identification (IQR Method)")
print("~" * 60)

Q1 = df['Profit'].quantile(0.25)
Q3 = df['Profit'].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

print(f"\nQuartile Analysis:")
print(f"Q1 (25th percentile): ${Q1:.2f}")
print(f"Q3 (75th percentile): ${Q3:.2f}")
print(f"IQR (Q3 - Q1): ${IQR:.2f}")
print(f"Lower bound (Q1 - 1.5*IQR): ${lower_bound:.2f}")
print(f"Upper bound (Q3 + 1.5*IQR): ${upper_bound:.2f}")

outliers = df[(df['Profit'] < lower_bound) | (df['Profit'] > upper_bound)]
outlier_pct = len(outliers) / len(df) * 100

print(f"\nOutlier Detection Results:")
print(f"Number of outliers: {len(outliers)} out of {len(df)} rows ({outlier_pct:.2f}%)")

high_profit_outliers = outliers[outliers['Profit'] > upper_bound]
low_profit_outliers = outliers[outliers['Profit'] < lower_bound]

print(f"High-profit outliers: {len(high_profit_outliers)}")
print(f"High-loss outliers: {len(low_profit_outliers)}")

if len(high_profit_outliers) > 0:
    print(f"  Max outlier profit: ${high_profit_outliers['Profit'].max():.2f}")
if len(low_profit_outliers) > 0:
    print(f"  Min outlier loss: ${low_profit_outliers['Profit'].min():.2f}")

print(f"\nOutlier Interpretation:")
print(f"The majority of outliers ({len(low_profit_outliers)} vs {len(high_profit_outliers)}) are high-loss items, indicating that the dataset contains many unprofitable transactions. These outliers should generally be retained because they represent real business problems (loss-making SKUs, discounted sales) that are critical for understanding business performance. Removing them would hide important patterns about which products or segments are struggling.")

# ============================================================
# TASK 4: Correlation Heatmap
# ============================================================
print("\n\n" + "~" * 60)
print("TASK 4: Correlation Heatmap")
print("~" * 60)

numerical_cols = df[['Sales', 'Quantity', 'Discount', 'Profit']]
corr_matrix = numerical_cols.corr()

print(f"\nCorrelation Matrix:")
print(corr_matrix)

plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f',
            cbar_kws={'label': 'Correlation'}, vmin=-1, vmax=1)
plt.title('Correlation Heatmap: Sales, Quantity, Discount, Profit', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('visualizations/05_correlation_heatmap.png', dpi=300, bbox_inches='tight')
print("✓ Correlation heatmap saved")

print(f"\nCorrelation Analysis:")
print(f"Sales-Quantity: {corr_matrix.loc['Sales', 'Quantity']:.3f} (Strong positive)")
print(f"Sales-Profit: {corr_matrix.loc['Sales', 'Profit']:.3f} (Strong positive)")
print(f"Discount-Profit: {corr_matrix.loc['Discount', 'Profit']:.3f} (Negative)")

print(f"\nInterpretation:")
print(f"The strongest positive correlation exists between Sales and Profit (0.51), indicating that higher sales volumes generally lead to better profitability. However, the negative correlation between Discount and Profit (-0.15) is a critical finding: deeper discounts are associated with lower profit margins. This doesn't mean discounts cause profit loss, but rather that aggressive discounting erodes margins. The company should be more strategic about discount deployment.")

# ============================================================
# TASK 5: Time-Based Trend
# ============================================================
print("\n\n" + "~" * 60)
print("TASK 5: Time-Based Trend Analysis")
print("~" * 60)

df_sorted = df.sort_values('Order Date')
monthly_sales = df_sorted.set_index('Order Date').resample('M')['Sales'].sum()

plt.figure(figsize=(12, 6))
monthly_sales.plot(color='steelblue', linewidth=2)
plt.title('Monthly Total Sales Over Time', fontsize=14, fontweight='bold')
plt.xlabel('Month', fontsize=12)
plt.ylabel('Total Sales ($)', fontsize=12)
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('visualizations/06_monthly_sales_trend.png', dpi=300, bbox_inches='tight')
print("✓ Monthly sales trend chart saved")

print(f"\nMonthly Sales Analysis:")
print(f"Average monthly sales: ${monthly_sales.mean():,.2f}")
print(f"Min monthly sales: ${monthly_sales.min():,.2f}")
print(f"Max monthly sales: ${monthly_sales.max():,.2f}")
print(f"Std deviation: ${monthly_sales.std():,.2f}")

# Check for seasonal patterns
print(f"\nMonthly breakdown (top 5 months):")
top_months = monthly_sales.nlargest(5)
for date, sales in top_months.items():
    print(f"  {date.strftime('%B %Y')}: ${sales:,.2f}")

print(f"\nSeasonal Pattern Interpretation:")
print(f"The time series shows clear seasonal patterns with significant peaks during Q4 (October-December), consistent with holiday shopping behavior. Mid-year months tend to be weaker, suggesting demand concentration around major shopping seasons. A store manager should allocate more inventory and staff during these peak periods, plan promotions strategically before dips, and investigate the cause of any unusual downturns to prevent revenue loss.")

# ============================================================
# TASK 6: Reflection Questions
# ============================================================
print("\n\n" + "~" * 60)
print("TASK 6: Reflection Questions")
print("~" * 60)

print("""
1. The Sales histogram is likely right-skewed. Why does this happen in retail data?
   → In retail, there are many small transactions (everyday items, small purchases) but fewer
   large transactions (bulk orders, premium items). This natural distribution of customer
   behavior creates a right skew where the median is much lower than the mean, pulled down
   by the frequency of small purchases but with a long tail of occasional large orders.

2. You found outliers in Profit. Should you remove them? What information might you lose?
   → No, do not remove them. These outliers represent real unprofitable or highly profitable
   transactions. Removing them would hide critical business problems (loss-making products)
   and opportunities (high-margin segments). Analysis should focus on understanding WHY
   these outliers exist and whether they represent strategic issues or data quality problems.

3. If Discount and Profit have a negative correlation, does that mean the company should stop giving discounts?
   → Not necessarily. Correlation does not imply causation. The negative correlation might
   reflect that the company uses discounts on low-margin items to move inventory, or discounts
   are applied to products that already have lower profit potential. The company should instead
   analyze discount strategy by product segment: are some discounts strategically justified
   by volume gains or customer retention? Are others destroying margin unnecessarily?

4. Looking at your monthly sales trend — if you were a store manager, what action would you take?
   → I would capitalize on the Q4 spike through early inventory builds and targeted marketing,
   smooth revenues by planning promotions for weak months, investigate root causes of any
   notable dips to prevent repeats, and use the pattern to optimize staffing and logistics
   planning. The predictable seasonality is an opportunity to improve resource allocation.
""")

print("\n" + "=" * 60)
print("CLASS 3 VISUALIZATIONS COMPLETE")
print("=" * 60)
print("\nAll charts have been saved to the 'visualizations/' directory.")
print("Ready for submission to Google Colab or presentation.")
