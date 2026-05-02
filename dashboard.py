import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Configure Streamlit page
st.set_page_config(
    page_title="Sales Analysis Report",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main {
        padding: 2rem;
    }
    h1 {
        color: #1f1f1f;
        text-align: left;
        margin-bottom: 1.5rem;
        font-size: 28px;
        font-weight: 600;
    }
    h2 {
        color: #333333;
        border-bottom: 1px solid #cccccc;
        padding-bottom: 0.5rem;
        font-size: 20px;
        margin-top: 2rem;
    }
    h3 {
        color: #555555;
        font-size: 16px;
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.title("Superstore Sales Analysis Report")
st.markdown("Class 3 Assignment: Data Visualization and Trend Analysis")
st.markdown("---")

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv("data/superstore_cleaned.csv")
    df['Order Date'] = pd.to_datetime(df['Order Date'])
    df['Ship Date'] = pd.to_datetime(df['Ship Date'])
    return df

df = load_data()

# ===== SIDEBAR FILTERS =====
st.sidebar.header("Filters")
st.sidebar.markdown("---")

# Date range filter
date_range = st.sidebar.date_input(
    "Select Date Range:",
    value=(df['Order Date'].min().date(), df['Order Date'].max().date()),
    min_value=df['Order Date'].min().date(),
    max_value=df['Order Date'].max().date()
)

# Category filter
selected_categories = st.sidebar.multiselect(
    "Select Categories:",
    options=df['Category'].unique(),
    default=df['Category'].unique()
)

# Region filter
selected_regions = st.sidebar.multiselect(
    "Select Regions:",
    options=df['Region'].unique(),
    default=df['Region'].unique()
)

# Segment filter
selected_segments = st.sidebar.multiselect(
    "Select Segments:",
    options=df['Segment'].unique(),
    default=df['Segment'].unique()
)

st.sidebar.markdown("---")

# Apply filters
df_filtered = df[
    (df['Order Date'].dt.date >= date_range[0]) &
    (df['Order Date'].dt.date <= date_range[1]) &
    (df['Category'].isin(selected_categories)) &
    (df['Region'].isin(selected_regions)) &
    (df['Segment'].isin(selected_segments))
].copy()

# ===== KEY METRICS =====
st.header("Summary Metrics")
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric(
        label="Total Sales",
        value=f"${df_filtered['Sales'].sum():,.0f}"
    )

with col2:
    st.metric(
        label="Total Profit",
        value=f"${df_filtered['Profit'].sum():,.0f}"
    )

with col3:
    st.metric(
        label="Avg Order Value",
        value=f"${df_filtered['Sales'].mean():.2f}"
    )

with col4:
    st.metric(
        label="Transactions",
        value=f"{len(df_filtered)}"
    )

with col5:
    profit_margin = (df_filtered['Profit'].sum() / df_filtered['Sales'].sum() * 100)
    st.metric(
        label="Profit Margin",
        value=f"{profit_margin:.2f}%"
    )

st.markdown("---")

# ===== TASK 1: DISTRIBUTION PLOTS =====
st.header("Task 1: Distribution Analysis")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Sales Distribution (Histogram)")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(df_filtered['Sales'], bins=50, color='steelblue', edgecolor='black', alpha=0.7)
    ax.set_title('Distribution of Sales', fontsize=14, fontweight='bold')
    ax.set_xlabel('Sales ($)', fontsize=11)
    ax.set_ylabel('Frequency', fontsize=11)
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)

    st.write("""
    The Sales distribution shows a right-skewed pattern with most transactions
    clustered in the lower price range ($0-$500). This is typical in retail,
    where small purchases are frequent but large transactions are rare.
    """)

with col2:
    st.subheader("Profit Distribution (Boxplot)")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(x=df_filtered['Profit'], color='lightcoral', ax=ax)
    ax.set_title('Boxplot of Profit', fontsize=14, fontweight='bold')
    ax.set_xlabel('Profit ($)', fontsize=11)
    ax.grid(True, alpha=0.3, axis='x')
    st.pyplot(fig)

    st.write("""
    The boxplot reveals quartiles, median, and outliers in profit data.
    The median is around $12, with the middle 50% of transactions ranging between
    approximately $2 and $30. Outliers are visible on both sides.
    """)

st.markdown("---")

# ===== TASK 2: CATEGORY-WISE TRENDS =====
st.header("Task 2: Category and Region Analysis")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Total Sales by Category")
    cat_sales = df_filtered.groupby('Category')['Sales'].sum().sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['#2196F3', '#4CAF50', '#FF9800']
    cat_sales.plot(kind='bar', ax=ax, color=colors[:len(cat_sales)], edgecolor='black')
    ax.set_title('Sales by Category', fontsize=14, fontweight='bold')
    ax.set_xlabel('Category', fontsize=11)
    ax.set_ylabel('Total Sales ($)', fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    plt.xticks(rotation=0)
    st.pyplot(fig)

    st.write(f"""
    {cat_sales.index[0]} leads with ${cat_sales.iloc[0]:,.0f}, followed by
    {cat_sales.index[1]} and {cat_sales.index[2]}. {cat_sales.index[0]} generates the highest revenue.
    """)

with col2:
    st.subheader("Total Sales by Region")
    region_sales = df_filtered.groupby('Region')['Sales'].sum().sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(10, 6))
    region_sales.plot(kind='bar', ax=ax, color='steelblue', edgecolor='black')
    ax.set_title('Sales by Region', fontsize=14, fontweight='bold')
    ax.set_xlabel('Region', fontsize=11)
    ax.set_ylabel('Total Sales ($)', fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    plt.xticks(rotation=45)
    st.pyplot(fig)

    st.write(f"""
    The {region_sales.index[0]} region generates the highest sales at ${region_sales.iloc[0]:,.0f}.
    Regional performance variation suggests differences in market dynamics across regions.
    """)

st.markdown("---")

# ===== TASK 3: OUTLIER IDENTIFICATION =====
st.header("Task 3: Outlier Analysis")

col1, col2 = st.columns(2)

with col1:
    # Calculate quartiles
    Q1 = df_filtered['Profit'].quantile(0.25)
    Q3 = df_filtered['Profit'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers = df_filtered[(df_filtered['Profit'] < lower_bound) |
                           (df_filtered['Profit'] > upper_bound)]
    outlier_count = len(outliers)
    outlier_percentage = (outlier_count / len(df_filtered)) * 100
    low_outliers = len(outliers[outliers['Profit'] < lower_bound])
    high_outliers = len(outliers[outliers['Profit'] > upper_bound])

    st.metric("Total Outliers", f"{outlier_count} ({outlier_percentage:.2f}%)")
    st.metric("Low-Profit Outliers", low_outliers)
    st.metric("High-Profit Outliers", high_outliers)

with col2:
    st.subheader("IQR Method Results")
    st.write(f"Q1 (25th percentile): ${Q1:.2f}")
    st.write(f"Q3 (75th percentile): ${Q3:.2f}")
    st.write(f"IQR: ${IQR:.2f}")
    st.write(f"Lower Bound: ${lower_bound:.2f}")
    st.write(f"Upper Bound: ${upper_bound:.2f}")

st.write("""
Outliers identified using the IQR method (1.5 × IQR). Most outliers are on the negative side
(losses), indicating some products or transactions were unprofitable. These merit investigation.
""")

st.markdown("---")

# ===== TASK 4: CORRELATION HEATMAP =====
st.header("Task 4: Correlation Analysis")

st.subheader("Correlation Matrix")

numerical_cols = df_filtered[['Sales', 'Quantity', 'Discount', 'Profit']]
corr_matrix = numerical_cols.corr()

fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f',
            cbar_kws={'label': 'Correlation Coefficient'}, linewidths=2, ax=ax,
            square=True, annot_kws={'size': 12})
ax.set_title('Correlation Matrix', fontsize=14, fontweight='bold')
st.pyplot(fig)

col1, col2 = st.columns(2)

with col1:
    sales_quantity_corr = corr_matrix.loc['Sales', 'Quantity']
    st.write(f"Sales-Quantity: {sales_quantity_corr:.3f}")
    st.write(f"Positive relationship - more items sold indicates higher revenue.")

with col2:
    discount_profit_corr = corr_matrix.loc['Discount', 'Profit']
    st.write(f"Discount-Profit: {discount_profit_corr:.3f}")
    st.write(f"Negative correlation - higher discounts reduce short-term profitability.")

st.markdown("---")

# ===== TASK 5: TIME-BASED TREND =====
st.header("Task 5: Temporal Trends")

st.subheader("Monthly Sales Over Time")

df_sorted = df_filtered.set_index('Order Date')
monthly_sales = df_sorted['Sales'].resample('ME').sum()

fig, ax = plt.subplots(figsize=(14, 6))
ax.plot(monthly_sales.index, monthly_sales.values, color='steelblue', linewidth=2.5, marker='o', markersize=6)
ax.fill_between(monthly_sales.index, monthly_sales.values, alpha=0.3, color='steelblue')
ax.set_title('Monthly Total Sales Over Time', fontsize=14, fontweight='bold')
ax.set_xlabel('Month', fontsize=11)
ax.set_ylabel('Total Sales ($)', fontsize=11)
ax.grid(True, alpha=0.3)
plt.xticks(rotation=45)
st.pyplot(fig)

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Highest Month", f"${monthly_sales.max():,.0f}")
    st.write(monthly_sales.idxmax().strftime('%B %Y'))

with col2:
    st.metric("Lowest Month", f"${monthly_sales.min():,.0f}")
    st.write(monthly_sales.idxmin().strftime('%B %Y'))

with col3:
    st.metric("Average Monthly", f"${monthly_sales.mean():,.0f}")
    st.write(f"Std Dev: ${monthly_sales.std():,.0f}")

st.write("""
The monthly sales trend shows clear seasonal patterns with peaks typically occurring in
November-December (holiday season) and lower activity in summer months. These patterns are
crucial for inventory planning and marketing strategy.
""")

st.markdown("---")

# ===== TASK 6: REFLECTION QUESTIONS =====
st.header("Task 6: Analysis Questions and Findings")

st.subheader("Question 1: Why is the Sales histogram right-skewed?")
st.write("""
Retail data naturally exhibits right-skew because small purchases are common
(customers buying single items) while large bulk purchases are rare. This distribution
reflects typical consumer behavior patterns where most transactions are modest, but there
is a long tail of larger orders.
""")

st.subheader("Question 2: Should we remove the profit outliers?")
st.write("""
No, outliers should not be removed. They may represent important edge cases such as
clearance sales, bulk orders, or operational inefficiencies worth investigating. Removing
them would lose valuable business insights and could mask systemic problems.
""")

st.subheader("Question 3: If Discount and Profit correlate negatively, stop discounts?")
st.write("""
Not necessarily. Negative correlation does not imply causation. Discounts may be given
to move slow inventory or attract new customers, which could increase volume and long-term
profitability despite short-term margin reduction. The relationship is complex.
""")

st.subheader("Question 4: As a store manager, what actions based on the sales trend?")
st.write("""
Key actions:
1. Prepare inventory and staff for the November-December surge
2. Plan promotional campaigns during low seasons (summer) to boost sales
3. Analyze what drives peaks - promotions, holidays, or specific products
4. Replicate success - implement winning strategies year-round
5. Manage cash flow - expect revenue concentration in peak months
""")

st.markdown("---")

# ===== FOOTER =====
st.markdown("""
<div style='text-align: center; color: #999; margin-top: 2rem; font-size: 12px;'>
    <p>Superstore Sales Analysis Report</p>
    <p>Module 2 | Class 3 Assignment: Data Visualization and Trend Analysis</p>
</div>
""", unsafe_allow_html=True)
