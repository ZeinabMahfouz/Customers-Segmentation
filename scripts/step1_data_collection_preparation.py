
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
from datetime import datetime

warnings.filterwarnings('ignore')

# Set visualization style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)

# Create output directories if they don't exist
os.makedirs('data/processed', exist_ok=True)
os.makedirs('outputs/figures', exist_ok=True)
os.makedirs('outputs/reports', exist_ok=True)

# ============================================================================
# CONFIGURATION
# ============================================================================

RAW_DATA_PATH = 'data/raw/marketing_campaign.csv'
OUTPUT_DATA_PATH = 'data/processed/step1_cleaned_data.csv'
CURRENT_YEAR = 2024

print("=" * 80)
print("STEP 1: DATA COLLECTION AND PREPARATION")
print("=" * 80)
print(f"\nStart Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# ============================================================================
# 1.1: LOAD RAW DATA
# ============================================================================

print("\n" + "-" * 80)
print("1.1 LOADING RAW DATA")
print("-" * 80)

try:
    df = pd.read_csv(RAW_DATA_PATH, sep='\t')
    print(f"✓ Data loaded successfully from: {RAW_DATA_PATH}")
    print(f"  Initial shape: {df.shape[0]} rows, {df.shape[1]} columns")
except FileNotFoundError:
    print(f"✗ Error: File not found at {RAW_DATA_PATH}")
    print(r"  Please download the dataset from Kaggle and place it in data/raw/")
    exit(1)

# ============================================================================
# 1.2: INITIAL DATA EXPLORATION
# ============================================================================

print("\n" + "-" * 80)
print("1.2 INITIAL DATA EXPLORATION")
print("-" * 80)

print("\nFirst 5 rows:")
print(df.head())

print("\n\nColumn Information:")
print(df.info())

print("\n\nStatistical Summary:")
print(df.describe())

print("\n\nColumn Names:")
for i, col in enumerate(df.columns, 1):
    print(f"  {i:2d}. {col}")

# ============================================================================
# 1.3: DATA QUALITY ASSESSMENT
# ============================================================================

print("\n" + "-" * 80)
print("1.3 DATA QUALITY ASSESSMENT")
print("-" * 80)

# Check for missing values
print("\nMissing Values:")
missing_summary = pd.DataFrame({
    'Column': df.columns,
    'Missing_Count': df.isnull().sum().values,
    'Missing_Percentage': (df.isnull().sum() / len(df) * 100).round(2).values
})
missing_summary = missing_summary[missing_summary['Missing_Count'] > 0]

if len(missing_summary) > 0:
    print(missing_summary.to_string(index=False))
else:
    print("  No missing values found!")

# Check for duplicates
duplicates = df.duplicated().sum()
print(f"\nDuplicate Rows: {duplicates}")

# Check data types
print("\nData Types Distribution:")
print(df.dtypes.value_counts())

# ============================================================================
# 1.4: HANDLE MISSING VALUES
# ============================================================================

print("\n" + "-" * 80)
print("1.4 HANDLING MISSING VALUES")
print("-" * 80)

initial_rows = len(df)

# Handle missing values in Income (fill with median)
if 'Income' in df.columns:
    income_missing = df['Income'].isnull().sum()
    if income_missing > 0:
        median_income = df['Income'].median()
        df['Income'].fillna(median_income, inplace=True)
        print(f"✓ Filled {income_missing} missing Income values with median: ${median_income:,.2f}")

# Remove rows with any remaining missing values
remaining_missing = df.isnull().sum().sum()
if remaining_missing > 0:
    df.dropna(inplace=True)
    print(f"✓ Removed {initial_rows - len(df)} rows with missing values")
else:
    print("✓ No missing values to remove")

print(f"  Remaining rows: {len(df)}")

# ============================================================================
# 1.5: REMOVE DUPLICATES
# ============================================================================

print("\n" + "-" * 80)
print("1.5 REMOVING DUPLICATES")
print("-" * 80)

before_dedup = len(df)
df.drop_duplicates(inplace=True)
duplicates_removed = before_dedup - len(df)

if duplicates_removed > 0:
    print(f"✓ Removed {duplicates_removed} duplicate rows")
else:
    print("✓ No duplicates found")

print(f"  Remaining rows: {len(df)}")

# ============================================================================
# 1.6: OUTLIER DETECTION
# ============================================================================

print("\n" + "-" * 80)
print("1.6 OUTLIER DETECTION AND HANDLING")
print("-" * 80)

def detect_outliers_iqr(data, column):
    """Detect outliers using IQR method"""
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = data[(data[column] < lower_bound) | (data[column] > upper_bound)]
    return len(outliers), lower_bound, upper_bound

# Create Age from Year_Birth if not exists
if 'Year_Birth' in df.columns and 'Age' not in df.columns:
    df['Age'] = CURRENT_YEAR - df['Year_Birth']

# Check for outliers in key numerical columns
outlier_columns = ['Age', 'Income']
if 'Age' in df.columns:
    outlier_columns.append('Age')

print("\nOutlier Analysis (IQR Method):")
for col in outlier_columns:
    if col in df.columns:
        n_outliers, lower, upper = detect_outliers_iqr(df, col)
        print(f"  {col}: {n_outliers} outliers (bounds: {lower:.2f} - {upper:.2f})")

# Remove unrealistic ages
before_age_filter = len(df)
if 'Age' in df.columns:
    df = df[(df['Age'] >= 18) & (df['Age'] <= 100)]
    removed = before_age_filter - len(df)
    if removed > 0:
        print(f"\n✓ Removed {removed} rows with unrealistic ages (< 18 or > 100)")
        print(f"  Remaining rows: {len(df)}")

# Remove extreme income outliers (optional - be careful with this)
if 'Income' in df.columns:
    # Only remove extreme outliers beyond 3 standard deviations
    income_mean = df['Income'].mean()
    income_std = df['Income'].std()
    lower_income = income_mean - 3 * income_std
    upper_income = income_mean + 3 * income_std
    
    before_income_filter = len(df)
    df = df[(df['Income'] >= lower_income) & (df['Income'] <= upper_income)]
    removed = before_income_filter - len(df)
    
    if removed > 0:
        print(f"✓ Removed {removed} rows with extreme income outliers (±3 std)")
        print(f"  Remaining rows: {len(df)}")

# ============================================================================
# 1.7: CREATE BASIC FEATURES
# ============================================================================

print("\n" + "-" * 80)
print("1.7 CREATING BASIC FEATURES")
print("-" * 80)

# Age (if not already created)
if 'Year_Birth' in df.columns and 'Age' not in df.columns:
    df['Age'] = CURRENT_YEAR - df['Year_Birth']
    print("✓ Created: Age")

# Total Spending
if all(col in df.columns for col in ['MntWines', 'MntFruits', 'MntMeatProducts', 
                                       'MntFishProducts', 'MntSweetProducts', 'MntGoldProds']):
    df['Total_Spending'] = df[['MntWines', 'MntFruits', 'MntMeatProducts', 
                                'MntFishProducts', 'MntSweetProducts', 'MntGoldProds']].sum(axis=1)
    print("✓ Created: Total_Spending")

# Total Purchases
if all(col in df.columns for col in ['NumWebPurchases', 'NumCatalogPurchases', 
                                       'NumStorePurchases', 'NumDealsPurchases']):
    df['Total_Purchases'] = df[['NumWebPurchases', 'NumCatalogPurchases', 
                                 'NumStorePurchases', 'NumDealsPurchases']].sum(axis=1)
    print("✓ Created: Total_Purchases")

# Total Children
if all(col in df.columns for col in ['Kidhome', 'Teenhome']):
    df['Children'] = df['Kidhome'] + df['Teenhome']
    print("✓ Created: Children")

# Customer Tenure
if 'Dt_Customer' in df.columns:
    df['Dt_Customer'] = pd.to_datetime(df['Dt_Customer'],dayfirst=True)
    reference_date = df['Dt_Customer'].max()
    df['Customer_Tenure_Days'] = (reference_date - df['Dt_Customer']).dt.days
    print("✓ Created: Customer_Tenure_Days")

# ============================================================================
# 1.8: DATA VALIDATION
# ============================================================================

print("\n" + "-" * 80)
print("1.8 DATA VALIDATION")
print("-" * 80)

# Check for negative values in numerical columns
numerical_cols = df.select_dtypes(include=[np.number]).columns
negative_values = {}

for col in numerical_cols:
    if col != 'ID':  # Skip ID column
        neg_count = (df[col] < 0).sum()
        if neg_count > 0:
            negative_values[col] = neg_count

if negative_values:
    print("⚠ Warning: Found negative values in:")
    for col, count in negative_values.items():
        print(f"  {col}: {count} negative values")
else:
    print("✓ No negative values found in numerical columns")

# Check data ranges
print("\nData Range Validation:")
if 'Age' in df.columns:
    print(f"  Age: {df['Age'].min():.0f} - {df['Age'].max():.0f} years")
if 'Income' in df.columns:
    print(f"  Income: ${df['Income'].min():,.2f} - ${df['Income'].max():,.2f}")
if 'Total_Spending' in df.columns:
    print(f"  Total Spending: ${df['Total_Spending'].min():,.2f} - ${df['Total_Spending'].max():,.2f}")

# ============================================================================
# 1.9: VISUALIZATION - DATA DISTRIBUTION
# ============================================================================

print("\n" + "-" * 80)
print("1.9 CREATING DATA DISTRIBUTION VISUALIZATIONS")
print("-" * 80)

# Create distribution plots for key variables
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('Data Distribution After Cleaning', fontsize=16, fontweight='bold')

# Age distribution
if 'Age' in df.columns:
    axes[0, 0].hist(df['Age'], bins=30, color='steelblue', alpha=0.7, edgecolor='black')
    axes[0, 0].set_title('Age Distribution')
    axes[0, 0].set_xlabel('Age')
    axes[0, 0].set_ylabel('Frequency')

# Income distribution
if 'Income' in df.columns:
    axes[0, 1].hist(df['Income'], bins=30, color='green', alpha=0.7, edgecolor='black')
    axes[0, 1].set_title('Income Distribution')
    axes[0, 1].set_xlabel('Income ($)')
    axes[0, 1].set_ylabel('Frequency')

# Total Spending distribution
if 'Total_Spending' in df.columns:
    axes[0, 2].hist(df['Total_Spending'], bins=30, color='coral', alpha=0.7, edgecolor='black')
    axes[0, 2].set_title('Total Spending Distribution')
    axes[0, 2].set_xlabel('Total Spending ($)')
    axes[0, 2].set_ylabel('Frequency')

# Education distribution
if 'Education' in df.columns:
    education_counts = df['Education'].value_counts()
    axes[1, 0].bar(range(len(education_counts)), education_counts.values, color='purple', alpha=0.7)
    axes[1, 0].set_xticks(range(len(education_counts)))
    axes[1, 0].set_xticklabels(education_counts.index, rotation=45, ha='right')
    axes[1, 0].set_title('Education Level Distribution')
    axes[1, 0].set_ylabel('Count')

# Marital Status distribution
if 'Marital_Status' in df.columns:
    marital_counts = df['Marital_Status'].value_counts()
    axes[1, 1].bar(range(len(marital_counts)), marital_counts.values, color='orange', alpha=0.7)
    axes[1, 1].set_xticks(range(len(marital_counts)))
    axes[1, 1].set_xticklabels(marital_counts.index, rotation=45, ha='right')
    axes[1, 1].set_title('Marital Status Distribution')
    axes[1, 1].set_ylabel('Count')

# Total Purchases distribution
if 'Total_Purchases' in df.columns:
    axes[1, 2].hist(df['Total_Purchases'], bins=30, color='teal', alpha=0.7, edgecolor='black')
    axes[1, 2].set_title('Total Purchases Distribution')
    axes[1, 2].set_xlabel('Total Purchases')
    axes[1, 2].set_ylabel('Frequency')

plt.tight_layout()
plt.savefig('outputs/figures/step1_data_distributions.png', dpi=300, bbox_inches='tight')
print("✓ Saved visualization: outputs/figures/step1_data_distributions.png")
plt.close()

# ============================================================================
# 1.10: SAVE CLEANED DATA
# ============================================================================

print("\n" + "-" * 80)
print("1.10 SAVING CLEANED DATA")
print("-" * 80)

# Save cleaned dataset
df.to_csv(OUTPUT_DATA_PATH, index=False)
print(f"✓ Cleaned data saved to: {OUTPUT_DATA_PATH}")
print(f"  Final shape: {df.shape[0]} rows, {df.shape[1]} columns")

# Create data quality report
quality_report = pd.DataFrame({
    'Column': df.columns,
    'Data_Type': df.dtypes.values,
    'Non_Null_Count': df.count().values,
    'Null_Count': df.isnull().sum().values,
    'Unique_Values': [df[col].nunique() for col in df.columns],
    'Sample_Value': [df[col].iloc[0] if len(df) > 0 else None for col in df.columns]
})

quality_report_path = 'outputs/reports/step1_data_quality_report.csv'
quality_report.to_csv(quality_report_path, index=False)
print(f"✓ Data quality report saved to: {quality_report_path}")

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "=" * 80)
print("STEP 1 COMPLETED SUCCESSFULLY")
print("=" * 80)

print(f"\nProcessing Summary:")
print(f"  Initial rows: {initial_rows}")
print(f"  Final rows: {len(df)}")
print(f"  Rows removed: {initial_rows - len(df)} ({(initial_rows - len(df))/initial_rows*100:.2f}%)")
print(f"  Final columns: {len(df.columns)}")

print(f"\nKey Statistics:")
if 'Age' in df.columns:
    print(f"  Age range: {df['Age'].min():.0f} - {df['Age'].max():.0f} years")
if 'Income' in df.columns:
    print(f"  Income range: ${df['Income'].min():,.2f} - ${df['Income'].max():,.2f}")
if 'Total_Spending' in df.columns:
    print(f"  Spending range: ${df['Total_Spending'].min():,.2f} - ${df['Total_Spending'].max():,.2f}")

print(f"\nOutput Files:")
print(f"  1. {OUTPUT_DATA_PATH}")
print(f"  2. {quality_report_path}")
print(f"  3. outputs/figures/step1_data_distributions.png")

print(f"\nNext Step: Run step2_rfm_analysis.py")
print(f"\nEnd Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 80)