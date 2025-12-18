
import pandas as pd
import numpy as np
from datetime import datetime
import os
import pickle

print("="*70)
print("STEP 10: TABLEAU EXPORT")
print("="*70)

# =============================================================================
# 1. LOAD FINAL DATA AND MODEL INFO
# =============================================================================

print("\n1. Loading final processed data and model information...")

# Load the final evaluated data
df = pd.read_csv('D://My projects/customer_segmentation_project/data/processed/step6_evaluated_clusters.csv')
print(f"✓ Loaded {len(df)} customer records with {len(df.columns)} columns")

# Load model configuration
try:
    with open('D://My projects/customer_segmentation_project/outputs/models/model_config.pkl', 'rb') as f:
        model_config = pickle.load(f)
    print(f"✓ Loaded model configuration")
    print(f"  - Best Model: {model_config['best_model']}")
    print(f"  - Model Accuracy: {model_config['accuracy']*100:.2f}%")
    print(f"  - Number of Clusters: {model_config['n_clusters']}")
except Exception as e:
    print(f"⚠ Could not load model config: {e}")
    model_config = {'n_clusters': len(df['Cluster'].unique())}

n_clusters = model_config['n_clusters']
print(df.columns)

# =============================================================================
# 2. CREATE MAIN TABLEAU DATASET
# =============================================================================

print("\n2. Creating main Tableau dataset...")

# Select all relevant columns for Tableau
tableau_columns = [
    # Identifiers
    'ID',
    
    # Demographics
    'Age',
    'Education',
    'Marital_Status',
    'Income',
    'Children',
    'Kidhome',
    'Teenhome',
    'Family_Size',
    'Has_Children',
    
    # RFM Metrics
    'Recency',
    'R_Score',
    'R_Value',
    'F_Score',
    'F_Value',
    'M_Score',
    'M_Value',
    'RFM_Score',
    'RFM_Segment',
    
    # Spending
    'MntWines',
    'MntFruits',
    'MntMeatProducts',
    'MntFishProducts',
    'MntSweetProducts',
    'MntGoldProds',
    'Total_Spending',
    
    # Purchase Behavior
    'NumWebPurchases',
    'NumCatalogPurchases',
    'NumStorePurchases',
    'NumDealsPurchases',
    'Total_Purchases',
    'Avg_Order_Value',
    'Purchase_Frequency_Rate',
    
    # Engagement
    'NumWebVisitsMonth',
    'Campaign_Acceptance_Rate',
    'AcceptedCmp1',
    'AcceptedCmp2',
    'AcceptedCmp3',
    'AcceptedCmp4',
    'AcceptedCmp5',
    'Response',
    'Complain',
    'Has_Complained',
    'Engagement_Score',
    
    # Product Preferences
    'Wine_Ratio',
    'Meat_Ratio',
    'Product_Diversity',
    'Premium_Product_Ratio',
    
    # Channel Preferences
    'Web_Purchase_Ratio',
    'Store_Purchase_Ratio',
    
    # Customer Value
    'CLV_Estimate',
    'Customer_Value_Score',
    
    # Customer Segments
    'Customer_Segment',
    'Cluster',
    
    # Flags
    'Is_High_Spender',
    'Is_Active',
    'Is_Campaign_Responder',
    'Is_Web_Shopper',
    'Is_Deal_Seeker',
    
    # Tenure
    'Customer_Tenure_Days',
    'Dt_Customer',
    
    # Quality Metrics
    'Stability_Score',
    
    # Dimensionality Reduction (for scatter plots)
    'PCA1',
    'PCA2'
]

# Add 3D PCA if available
if 'PCA1_3D' in df.columns:
    tableau_columns.extend(['PCA1_3D', 'PCA2_3D', 'PCA3_3D'])

# Filter to only columns that exist
available_columns = [col for col in tableau_columns if col in df.columns]
tableau_df = df[available_columns].copy()
tableau_df['Cluster'] = tableau_df['Customer_Segment']
#tableau_df['Final_ML_Cluster'] = tableau_df['Customer_Segment']

print(f"✓ Created main dataset with {len(available_columns)} columns")

# =============================================================================
# 3. ADD DERIVED COLUMNS FOR TABLEAU
# =============================================================================

print("\n3. Adding derived columns for better Tableau experience...")

# Convert date if exists
if 'Dt_Customer' in tableau_df.columns:
    tableau_df['Customer_Since'] = pd.to_datetime(tableau_df['Dt_Customer'], errors='coerce')
    tableau_df['Enrollment_Year'] = tableau_df['Customer_Since'].dt.year
    tableau_df['Enrollment_Month'] = tableau_df['Customer_Since'].dt.month
    tableau_df['Enrollment_Quarter'] = tableau_df['Customer_Since'].dt.quarter
    tableau_df['Enrollment_Month_Name'] = tableau_df['Customer_Since'].dt.strftime('%B')
    print("  ✓ Added date-based features")

# Age groups
if 'Age' in tableau_df.columns:
    tableau_df['Age_Group'] = pd.cut(tableau_df['Age'], 
                                      bins=[0, 25, 35, 45, 55, 65, 100],
                                      labels=['18-25', '26-35', '36-45', '46-55', '56-65', '65+'])
    print("  ✓ Added age groups")

# Income groups
if 'Income' in tableau_df.columns:
    tableau_df['Income_Group'] = pd.cut(tableau_df['Income'],
                                         bins=[0, 30000, 50000, 70000, 100000, np.inf],
                                         labels=['<$30K', '$30-50K', '$50-70K', '$70-100K', '$100K+'])
    print("  ✓ Added income groups")

# Spending groups
if 'Total_Spending' in tableau_df.columns:
    tableau_df['Spending_Group'] = pd.cut(tableau_df['Total_Spending'],
                                           bins=[0, 100, 500, 1000, 2000, np.inf],
                                           labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])
    print("  ✓ Added spending groups")

# Tenure groups (in years)
if 'Customer_Tenure_Days' in tableau_df.columns:
    tableau_df['Tenure_Years'] = (tableau_df['Customer_Tenure_Days'] / 365).round(1)
    tableau_df['Tenure_Group'] = pd.cut(tableau_df['Tenure_Years'],
                                         bins=[0, 1, 2, 3, 5, np.inf],
                                         labels=['<1 Year', '1-2 Years', '2-3 Years', '3-5 Years', '5+ Years'])
    print("  ✓ Added tenure groups")

# Cluster names (descriptive)
if 'Cluster' in tableau_df.columns:
    tableau_df['Cluster_ID'] = tableau_df['Cluster']
    tableau_df['Cluster_Name'] = 'Segment ' + tableau_df['Cluster'].astype(str)
    print("  ✓ Added cluster names")

# Value tier
if 'CLV_Estimate' in tableau_df.columns:
    tableau_df['Value_Tier'] = pd.qcut(tableau_df['CLV_Estimate'], 
                                        q=4, 
                                        labels=['Bronze', 'Silver', 'Gold', 'Platinum'],
                                        duplicates='drop')
    print("  ✓ Added value tiers")

# Engagement level
if 'Engagement_Score' in tableau_df.columns:
    tableau_df['Engagement_Level'] = pd.cut(tableau_df['Engagement_Score'],
                                            bins=[0, 3, 6, 10],
                                            labels=['Low', 'Medium', 'High'])
    print("  ✓ Added engagement levels")

# Recency category
if 'Recency' in tableau_df.columns:
    tableau_df['Recency_Category'] = pd.cut(tableau_df['Recency'],
                                            bins=[0, 30, 90, 180, np.inf],
                                            labels=['Last Month', 'Last Quarter', 'Last 6 Months', '6+ Months'])
    print("  ✓ Added recency categories")

# Purchase frequency category
if 'Total_Purchases' in tableau_df.columns:
    tableau_df['Purchase_Frequency'] = pd.cut(tableau_df['Total_Purchases'],
                                              bins=[0, 5, 15, 30, np.inf],
                                              labels=['Rare', 'Occasional', 'Regular', 'Frequent'])
    print("  ✓ Added purchase frequency categories")

# Preferred channel
if all(col in tableau_df.columns for col in ['NumWebPurchases', 'NumStorePurchases', 'NumCatalogPurchases']):
    tableau_df['Preferred_Channel'] = tableau_df[['NumWebPurchases', 'NumStorePurchases', 'NumCatalogPurchases']].idxmax(axis=1)
    tableau_df['Preferred_Channel'] = tableau_df['Preferred_Channel'].map({
        'NumWebPurchases': 'Web',
        'NumStorePurchases': 'Store',
        'NumCatalogPurchases': 'Catalog'
    })
    print("  ✓ Added preferred channel")

# Top product category
product_cols = ['MntWines', 'MntMeatProducts', 'MntFishProducts', 'MntFruits', 'MntSweetProducts', 'MntGoldProds']
available_products = [col for col in product_cols if col in tableau_df.columns]
if available_products:
    tableau_df['Top_Product_Category'] = tableau_df[available_products].idxmax(axis=1)
    tableau_df['Top_Product_Category'] = tableau_df['Top_Product_Category'].str.replace('Mnt', '').str.replace('Products', '')
    print("  ✓ Added top product category")

print(f"\n✓ Final dataset has {len(tableau_df.columns)} columns")

# =============================================================================
# 4. CREATE CLUSTER SUMMARY TABLE
# =============================================================================

print("\n4. Creating cluster summary table...")

cluster_summary = tableau_df.groupby('Cluster').agg({
    'ID': 'count',
    'Age': 'mean',
    'Income': 'mean',
    'Total_Spending': ['mean', 'median', 'sum'],
    'Total_Purchases': 'mean',
    'Avg_Order_Value': 'mean',
    'Customer_Tenure_Days': 'mean',
    'Campaign_Acceptance_Rate': 'mean',
    'Engagement_Score': 'mean',
    'CLV_Estimate': ['mean', 'sum'],
    'Stability_Score': 'mean',
    'Is_High_Spender': 'mean',
    'Is_Active': 'mean',
    'Is_Campaign_Responder': 'mean',
    'Has_Children': 'mean'
}).round(2)

# Flatten column names
cluster_summary.columns = ['_'.join(col).strip() if col[1] else col[0] 
                           for col in cluster_summary.columns.values]
cluster_summary = cluster_summary.reset_index()

# Rename columns for clarity
cluster_summary.columns = [
    'Cluster', 'Customer_Count', 'Avg_Age', 'Avg_Income',
    'Avg_Spending', 'Median_Spending', 'Total_Revenue',
    'Avg_Purchases', 'Avg_Order_Value', 'Avg_Tenure_Days',
    'Avg_Campaign_Acceptance', 'Avg_Engagement_Score', 
    'Avg_CLV', 'Total_CLV', 'Avg_Stability',
    'Pct_High_Spenders', 'Pct_Active', 'Pct_Campaign_Responders',
    'Pct_Has_Children'
]

# Add cluster name
cluster_summary['Cluster_Name'] = 'Segment ' + cluster_summary['Cluster'].astype(str)

# Calculate percentage of total
cluster_summary['Pct_of_Total'] = (cluster_summary['Customer_Count'] / len(tableau_df) * 100).round(2)

print(f"✓ Created cluster summary table: {len(cluster_summary)} clusters")

# =============================================================================
# 5. CREATE RFM SEGMENT SUMMARY
# =============================================================================

print("\n5. Creating RFM segment summary...")

if 'RFM_Segment' in tableau_df.columns:
    rfm_summary = tableau_df.groupby('RFM_Segment').agg({
        'ID': 'count',
        'Total_Spending': ['mean', 'sum'],
        'Recency': 'mean',
        'Total_Purchases': 'mean',
        'M_Value': 'mean',
        'CLV_Estimate': 'mean'
    }).round(2)
    
    rfm_summary.columns = ['Customer_Count', 'Avg_Spending', 'Total_Revenue',
                           'Avg_Recency', 'Avg_Frequency', 'Avg_Monetary', 'Avg_CLV']
    rfm_summary = rfm_summary.reset_index()
    
    # Add percentage
    rfm_summary['Pct_of_Total'] = (rfm_summary['Customer_Count'] / len(tableau_df) * 100).round(2)
    
    print(f"✓ Created RFM summary table: {len(rfm_summary)} segments")
else:
    rfm_summary = None
    print("⚠ RFM_Segment not found, skipping RFM summary")

# =============================================================================
# 6. CREATE PRODUCT CATEGORY ANALYSIS
# =============================================================================

print("\n6. Creating product category analysis...")

if available_products:
    product_summary = tableau_df.groupby('Cluster')[available_products].mean().round(2)
    product_summary = product_summary.reset_index()
    
    # Rename columns
    product_summary.columns = ['Cluster'] + [col.replace('Mnt', 'Avg_').replace('Products', '') 
                                             for col in available_products]
    
    # Add cluster name
    product_summary['Cluster_Name'] = 'Segment ' + product_summary['Cluster'].astype(str)
    
    print(f"✓ Created product category analysis")
else:
    product_summary = None
    print("⚠ Product columns not found")

# =============================================================================
# 7. CREATE CHANNEL PREFERENCE ANALYSIS
# =============================================================================

print("\n7. Creating channel preference analysis...")

channel_cols = ['NumWebPurchases', 'NumStorePurchases', 'NumCatalogPurchases']
if all(col in tableau_df.columns for col in channel_cols):
    channel_summary = tableau_df.groupby('Cluster')[channel_cols].mean().round(2)
    channel_summary = channel_summary.reset_index()
    channel_summary['Cluster_Name'] = 'Segment ' + channel_summary['Cluster'].astype(str)
    
    # Add total purchases per cluster
    channel_summary['Total_Avg_Purchases'] = channel_summary[channel_cols].sum(axis=1)
    
    print(f"✓ Created channel preference analysis")
else:
    channel_summary = None
    print("⚠ Channel columns not found")

# =============================================================================
# 8. CREATE CAMPAIGN RESPONSE ANALYSIS
# =============================================================================

print("\n8. Creating campaign response analysis...")

campaign_cols = ['AcceptedCmp1', 'AcceptedCmp2', 'AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5', 'Response']
available_campaigns = [col for col in campaign_cols if col in tableau_df.columns]

if available_campaigns:
    campaign_summary = tableau_df.groupby('Cluster')[available_campaigns].sum()
    campaign_summary = campaign_summary.reset_index()
    campaign_summary['Cluster_Name'] = 'Segment ' + campaign_summary['Cluster'].astype(str)
    
    # Add total responses
    campaign_summary['Total_Responses'] = campaign_summary[available_campaigns].sum(axis=1)
    
    # Add acceptance rate
    if 'Campaign_Acceptance_Rate' in tableau_df.columns:
        campaign_summary['Avg_Acceptance_Rate'] = tableau_df.groupby('Cluster')['Campaign_Acceptance_Rate'].mean().values
    
    print(f"✓ Created campaign response analysis")
else:
    campaign_summary = None
    print("⚠ Campaign columns not found")

# =============================================================================
# 9. CREATE TIME-BASED ANALYSIS
# =============================================================================

print("\n9. Creating time-based analysis...")

if 'Enrollment_Year' in tableau_df.columns and 'Enrollment_Quarter' in tableau_df.columns:
    time_summary = tableau_df.groupby(['Enrollment_Year', 'Enrollment_Quarter', 'Cluster']).agg({
        'ID': 'count',
        'Total_Spending': 'sum',
        'CLV_Estimate': 'mean'
    }).round(2)
    
    time_summary.columns = ['New_Customers', 'Total_Revenue', 'Avg_CLV']
    time_summary = time_summary.reset_index()
    time_summary['Cluster_Name'] = 'Segment ' + time_summary['Cluster'].astype(str)
    
    print(f"✓ Created time-based analysis")
else:
    time_summary = None
    print("⚠ Time columns not found")

# =============================================================================
# 10. SAVE ALL DATASETS
# =============================================================================

print("\n10. Saving Tableau exports...")

# Create tableau exports directory
os.makedirs('tableau_exports', exist_ok=True)

# Save main dataset
tableau_df.to_csv('tableau_exports/01_customer_segmentation_main.csv', index=False)
print(f"✓ Saved main dataset: tableau_exports/01_customer_segmentation_main.csv")
print(f"  - Records: {len(tableau_df):,}")
print(f"  - Columns: {len(tableau_df.columns)}")

# Save cluster summary
cluster_summary.to_csv('tableau_exports/02_cluster_summary.csv', index=False)
print(f"✓ Saved cluster summary: tableau_exports/02_cluster_summary.csv")

# Save RFM summary
if rfm_summary is not None:
    rfm_summary.to_csv('tableau_exports/03_rfm_segment_summary.csv', index=False)
    print(f"✓ Saved RFM summary: tableau_exports/03_rfm_segment_summary.csv")

# Save product analysis
if product_summary is not None:
    product_summary.to_csv('tableau_exports/04_product_category_analysis.csv', index=False)
    print(f"✓ Saved product analysis: tableau_exports/04_product_category_analysis.csv")

# Save channel analysis
if channel_summary is not None:
    channel_summary.to_csv('tableau_exports/05_channel_preference_analysis.csv', index=False)
    print(f"✓ Saved channel analysis: tableau_exports/05_channel_preference_analysis.csv")

# Save campaign analysis
if campaign_summary is not None:
    campaign_summary.to_csv('tableau_exports/06_campaign_response_analysis.csv', index=False)
    print(f"✓ Saved campaign analysis: tableau_exports/06_campaign_response_analysis.csv")

# Save time-based analysis
if time_summary is not None:
    time_summary.to_csv('tableau_exports/07_time_based_analysis.csv', index=False)
    print(f"✓ Saved time analysis: tableau_exports/07_time_based_analysis.csv")

# =============================================================================
# 11. CREATE DATA DICTIONARY
# =============================================================================

print("\n11. Creating comprehensive data dictionary...")

data_dict = {
    'Column_Name': [],
    'Description': [],
    'Data_Type': [],
    'Category': [],
    'Sample_Values': []
}

# Define column descriptions
descriptions = {
    # Identifiers
    'ID': ('Customer unique identifier', 'Integer', 'Identifier', 'e.g., 1001, 1002'),
    'Cluster': ('Assigned cluster/segment number', 'Integer', 'Segmentation', 'e.g., 0, 1, 2, 3'),
    'Cluster_Name': ('Descriptive cluster name', 'Text', 'Segmentation', 'e.g., Segment 0, Segment 1'),
    
    # Demographics
    'Age': ('Customer age in years', 'Integer', 'Demographics', 'e.g., 25, 40, 65'),
    'Age_Group': ('Categorized age range', 'Text', 'Demographics', 'e.g., 26-35, 36-45'),
    'Education': ('Education level detail', 'Text', 'Demographics', 'e.g., Graduation, PhD'),
    'Marital_Status': ('Marital status', 'Text', 'Demographics', 'e.g., Married, Single'),
    'Income': ('Annual household income in dollars', 'Float', 'Demographics', 'e.g., 50000, 75000'),
    'Income_Group': ('Categorized income range', 'Text', 'Demographics', 'e.g., $50-70K, $70-100K'),
    'Family_Size': ('Total family size', 'Integer', 'Demographics', 'e.g., 2, 4'),
    'Has_Children': ('Whether customer has children', 'Binary', 'Demographics', '0=No, 1=Yes'),
    'Children': ('Total number of children', 'Integer', 'Demographics', 'e.g., 0, 1, 2'),
    
    # RFM
    'Recency': ('Days since last purchase', 'Integer', 'RFM', 'e.g., 10, 45, 90'),
    'Recency_Category': ('Categorized recency', 'Text', 'RFM', 'e.g., Last Month, Last Quarter'),
    'R_Score': ('Recency score (1-5, 5=recent)', 'Integer', 'RFM', 'e.g., 1, 3, 5'),
    'F_Score': ('Frequency score (1-5, 5=frequent)', 'Integer', 'RFM', 'e.g., 1, 3, 5'),
    'M_Score': ('Monetary score (1-5, 5=high spender)', 'Integer', 'RFM', 'e.g., 1, 3, 5'),
    'RFM_Score': ('Combined RFM score', 'Float', 'RFM', 'e.g., 3.5, 4.2'),
    'RFM_Segment': ('RFM segment classification', 'Text', 'Segmentation', 'e.g., Champions, At Risk'),
    
    # Spending
    'Total_Spending': ('Total amount spent', 'Float', 'Spending', 'e.g., 500, 1500, 3000'),
    'Spending_Group': ('Categorized spending level', 'Text', 'Spending', 'e.g., Low, Medium, High'),
    'Avg_Order_Value': ('Average order value', 'Float', 'Spending', 'e.g., 50, 100, 200'),
    'MntWines': ('Amount spent on wine products', 'Float', 'Product Spending', 'e.g., 100, 300'),
    'MntMeatProducts': ('Amount spent on meat products', 'Float', 'Product Spending', 'e.g., 150, 400'),
    
    # Purchases
    'Total_Purchases': ('Total number of purchases', 'Integer', 'Purchase Behavior', 'e.g., 5, 15, 30'),
    'Purchase_Frequency': ('Categorized purchase frequency', 'Text', 'Purchase Behavior', 'e.g., Rare, Regular'),
    'NumWebPurchases': ('Number of web purchases', 'Integer', 'Channel', 'e.g., 2, 5, 10'),
    'NumStorePurchases': ('Number of store purchases', 'Integer', 'Channel', 'e.g., 3, 8, 15'),
    'Preferred_Channel': ('Primary purchase channel', 'Text', 'Channel', 'e.g., Web, Store'),
    
    # Engagement
    'Campaign_Acceptance_Rate': ('Rate of campaign acceptance', 'Float', 'Engagement', 'e.g., 0.2, 0.5, 0.8'),
    'Engagement_Score': ('Overall engagement score (0-10)', 'Float', 'Engagement', 'e.g., 3.5, 6.2, 8.7'),
    'Engagement_Level': ('Categorized engagement', 'Text', 'Engagement', 'e.g., Low, Medium, High'),
    'NumWebVisitsMonth': ('Web visits per month', 'Integer', 'Engagement', 'e.g., 3, 7, 12'),
    
    # Value
    'CLV_Estimate': ('Customer lifetime value estimate', 'Float', 'Value', 'e.g., 1000, 3000, 5000'),
    'Customer_Value_Score': ('Overall customer value score', 'Float', 'Value', 'e.g., 10, 30, 50'),
    'Value_Tier': ('Customer value tier', 'Text', 'Value', 'e.g., Bronze, Silver, Gold, Platinum'),
    
    # Tenure
    'Customer_Tenure_Days': ('Days as customer', 'Integer', 'Tenure', 'e.g., 365, 730, 1095'),
    'Tenure_Years': ('Years as customer', 'Float', 'Tenure', 'e.g., 1.0, 2.5, 5.2'),
    'Tenure_Group': ('Categorized tenure', 'Text', 'Tenure', 'e.g., 1-2 Years, 3-5 Years'),
    
    # Flags
    'Is_High_Spender': ('High spender flag', 'Binary', 'Flags', '0=No, 1=Yes'),
    'Is_Active': ('Active customer flag', 'Binary', 'Flags', '0=No, 1=Yes'),
    'Is_Campaign_Responder': ('Campaign responder flag', 'Binary', 'Flags', '0=No, 1=Yes'),
    
    # Quality
    'Stability_Score': ('Cluster stability score (0-1)', 'Float', 'Quality', 'e.g., 0.75, 0.85, 0.95'),
    
    # Visualization
    'PCA1': ('First principal component', 'Float', 'Visualization', 'e.g., -1.5, 0.2, 2.3'),
    'PCA2': ('Second principal component', 'Float', 'Visualization', 'e.g., -0.8, 1.1, 3.2'),
}

for col in tableau_df.columns:
    if col in descriptions:
        data_dict['Column_Name'].append(col)
        data_dict['Description'].append(descriptions[col][0])
        data_dict['Data_Type'].append(descriptions[col][1])
        data_dict['Category'].append(descriptions[col][2])
        data_dict['Sample_Values'].append(descriptions[col][3])
    else:
        # Add basic info for columns not in descriptions
        data_dict['Column_Name'].append(col)
        data_dict['Description'].append(f'See documentation')
        data_dict['Data_Type'].append(str(tableau_df[col].dtype))
        data_dict['Category'].append('Other')
        data_dict['Sample_Values'].append('')

dict_df = pd.DataFrame(data_dict)
dict_df.to_csv('tableau_exports/00_data_dictionary.csv', index=False)
print(f"✓ Saved data dictionary: tableau_exports/00_data_dictionary.csv")

