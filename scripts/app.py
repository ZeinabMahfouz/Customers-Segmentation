
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# PAGE CONFIGURATION
# =============================================================================

st.set_page_config(
    page_title="Customer Segmentation Dashboard",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# CUSTOM CSS FOR DARK/LIGHT MODE COMPATIBILITY
# =============================================================================

st.markdown("""
    <style>
    /* Main container */
    .main {
        padding: 2rem;
    }
    
    /* Headers - works in both modes */
    h1, h2, h3 {
        color: var(--text-color);
    }
    
    /* Metric cards */
    .metric-card {
        background: var(--background-color);
        padding: 1.5rem;
        border-radius: 10px;
        border: 1px solid var(--secondary-background-color);
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    }
    
    /* Info boxes */
    .info-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    .success-box {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        color: white;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    .warning-box {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    /* Buttons */
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        font-size: 1rem;
        font-weight: 600;
        border-radius: 8px;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: var(--secondary-background-color);
    }
    
    /* Input fields - adaptive colors */
    .stTextInput>div>div>input,
    .stNumberInput>div>div>input,
    .stSelectbox>div>div>select {
        background: var(--background-color);
        color: var(--text-color);
        border: 1px solid var(--secondary-background-color);
        border-radius: 5px;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        padding: 10px 20px;
        background: var(--secondary-background-color);
        border-radius: 5px;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# =============================================================================
# LOAD MODELS AND DATA
# =============================================================================

@st.cache_resource
def load_models():
    """Load all necessary models and data"""
    try:
        # Load the best predictive model
        with open('D://My projects/customer_segmentation_project/outputs/models/random_forest.pkl', 'rb') as f:
            model = pickle.load(f)
        
        # Load scaler
        with open('D://My projects/customer_segmentation_project/outputs/models/scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        
        # Load model config
        with open('D://My projects/customer_segmentation_project/outputs/models/model_config.pkl', 'rb') as f:
            config = pickle.load(f)
        
        # Load PCA for visualization
        with open('D://My projects/customer_segmentation_project/outputs/models/pca_2d.pkl', 'rb') as f:
            pca_2d = pickle.load(f)
        
        return model, scaler, config, pca_2d
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None, None, None

@st.cache_data
def load_cluster_data():
    """Load clustered customer data"""
    try:
        df = pd.read_csv('D://My projects/customer_segmentation_project/data/processed/step6_evaluated_clusters.csv')
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

# Load models and data
model, scaler, config, pca_2d = load_models()
df = load_cluster_data()

# =============================================================================
# SIDEBAR NAVIGATION
# =============================================================================

st.sidebar.image("https://via.placeholder.com/200x80/667eea/ffffff?text=Customer+Insights",  width='stretch')
st.sidebar.title("🎯 Navigation")

page = st.sidebar.radio(
    "Select Page",
    ["🏠 Dashboard", "🔮 Predict Customer", "📊 Cluster Analysis", "📈 Model Performance", "ℹ️ About"],
    label_visibility="collapsed"
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📌 Quick Stats")
if df is not None:
    st.sidebar.metric("Total Customers", f"{len(df):,}")
    st.sidebar.metric("Number of Segments", f"{config['n_clusters']}")
    st.sidebar.metric("Model Accuracy", f"{config['accuracy']*100:.1f}%")

st.sidebar.markdown("---")
st.sidebar.markdown("*Powered by Machine Learning*")

# =============================================================================
# PAGE 1: DASHBOARD
# =============================================================================

if page == "🏠 Dashboard":
    st.title("🎯 Customer Segmentation Dashboard")
    st.markdown("### Welcome to the Customer Intelligence Platform")
    
    if df is None:
        st.error("Unable to load data. Please check if data files exist.")
        st.stop()
    
    # Key Metrics Row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric(
            label="📊 Total Customers",
            value=f"{len(df):,}",
            delta="Active Database"
        )
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        avg_clv = df['CLV_Estimate'].mean() if 'CLV_Estimate' in df.columns else 0
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric(
            label="💰 Avg Customer Value",
            value=f"${avg_clv:,.0f}",
            delta="Lifetime Value"
        )
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        avg_spending = df['Total_Spending'].mean() if 'Total_Spending' in df.columns else 0
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric(
            label="🛒 Avg Spending",
            value=f"${avg_spending:,.0f}",
            delta="Per Customer"
        )
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col4:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric(
            label="🎯 Segments",
            value=f"{config['n_clusters']}",
            delta="Clusters"
        )
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Cluster Distribution
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 📊 Customer Distribution by Segment")
        cluster_counts = df['Final_ML_Cluster'].value_counts().sort_index()
        
        fig = go.Figure(data=[
            go.Bar(
                x=[f'Segment {i}' for i in cluster_counts.index],
                y=cluster_counts.values,
                marker=dict(
                    color=cluster_counts.values,
                    colorscale='Viridis',
                    showscale=False
                ),
                text=cluster_counts.values,
                textposition='auto',
                hovertemplate='<b>%{x}</b><br>Customers: %{y}<extra></extra>'
            )
        ])
        
        fig.update_layout(
            xaxis_title="Customer Segment",
            yaxis_title="Number of Customers",
            template="plotly_white",
            height=400,
            hovermode='x'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### 📈 Segment Size")
        
        fig = go.Figure(data=[
            go.Pie(
                labels=[f'Segment {i}' for i in cluster_counts.index],
                values=cluster_counts.values,
                hole=0.4,
                marker=dict(colors=px.colors.qualitative.Set3),
                hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>'
            )
        ])
        
        fig.update_layout(
            template="plotly_white",
            height=400,
            showlegend=True,
            legend=dict(orientation="v", yanchor="middle", y=0.5)
        )
        
        st.plotly_chart(fig,  width='stretch')
    
    # Cluster Characteristics
    st.markdown("---")
    st.markdown("### 🎯 Segment Characteristics")
    
    characteristics = ['Total_Spending', 'Age', 'Income', 'Campaign_Acceptance_Rate']
    available_chars = [c for c in characteristics if c in df.columns]
    
    if available_chars:
        char_data = df.groupby('Final_ML_Cluster')[available_chars].mean().round(2)
        
        fig = go.Figure()
        
        for col in available_chars:
            fig.add_trace(go.Bar(
                name=col.replace('_', ' ').title(),
                x=[f'Segment {i}' for i in char_data.index],
                y=char_data[col],
                text=char_data[col],
                textposition='auto',
            ))
        
        fig.update_layout(
            barmode='group',
            xaxis_title="Customer Segment",
            yaxis_title="Average Value",
            template="plotly_white",
            height=500,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        st.plotly_chart(fig,  width='stretch')
    
    # PCA Visualization
    if 'PCA1' in df.columns and 'PCA2' in df.columns:
        st.markdown("---")
        st.markdown("### 🗺️ Customer Segmentation Map (PCA)")
        
        fig = px.scatter(
            df.sample(min(2000, len(df))),  # Sample for performance
            x='PCA1',
            y='PCA2',
            color='Final_ML_Cluster',
            color_continuous_scale='Viridis',
            opacity=0.6,
            template="plotly_white",
            labels={'PCA1': 'Principal Component 1', 'PCA2': 'Principal Component 2'},
            hover_data=['Total_Spending', 'Age'] if all(c in df.columns for c in ['Total_Spending', 'Age']) else None
        )
        
        fig.update_layout(height=600)
        fig.update_traces(marker=dict(size=8, line=dict(width=0.5, color='white')))
        
        st.plotly_chart(fig,width='stretch')

# =============================================================================
# PAGE 2: PREDICT CUSTOMER
# =============================================================================

elif page == "🔮 Predict Customer":
    st.title("🔮 Customer Segment Predictor")
    st.markdown("### Enter customer information to predict their segment")
    
    if model is None:
        st.error("Model not loaded. Please check if model files exist.")
        st.stop()
    
    st.markdown('<div class="info-box">💡 <strong>Tip:</strong> Fill in the customer details below and click "Predict Segment" to see which customer segment they belong to.</div>', unsafe_allow_html=True)
    
    # Input Form
    with st.form("prediction_form"):
        st.markdown("#### 📋 Customer Information")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**Demographics**")
            age = st.number_input("Age", min_value=18, max_value=100, value=40, step=1)
            income = st.number_input("Annual Income ($)", min_value=0, max_value=200000, value=50000, step=1000)
            education_level = st.selectbox("Education Level", [1, 2, 3, 4, 5], 
                                          format_func=lambda x: ["Basic", "High School", "Bachelor", "Master", "PhD"][x-1])
            family_size = st.number_input("Family Size", min_value=1, max_value=10, value=2, step=1)
            has_children = st.selectbox("Has Children", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
        
        with col2:
            st.markdown("**Purchase Behavior**")
            avg_order_value = st.number_input("Average Order Value ($)", min_value=0.0, max_value=1000.0, value=50.0, step=5.0)
            purchase_freq = st.slider("Purchase Frequency Rate", 0.0, 1.0, 0.5, 0.01)
            customer_tenure = st.number_input("Customer Tenure (days)", min_value=0, max_value=3650, value=365, step=30)
            wine_ratio = st.slider("Wine Spending Ratio", 0.0, 1.0, 0.3, 0.01)
            meat_ratio = st.slider("Meat Spending Ratio", 0.0, 1.0, 0.3, 0.01)
        
        with col3:
            st.markdown("**Engagement & Value**")
            campaign_acceptance = st.slider("Campaign Acceptance Rate", 0.0, 1.0, 0.2, 0.01)
            web_visits = st.number_input("Web Visits per Month", min_value=0, max_value=30, value=5, step=1)
            engagement_score = st.slider("Engagement Score", 0.0, 10.0, 5.0, 0.1)
            clv_estimate = st.number_input("CLV Estimate ($)", min_value=0, max_value=10000, value=2000, step=100)
            is_high_spender = st.selectbox("High Spender", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
        
        st.markdown("#### 🎯 RFM Scores")
        col1, col2, col3 = st.columns(3)
        with col1:
            r_score = st.slider("Recency Score (1-5)", 1, 5, 3, help="5 = Purchased recently")
        with col2:
            f_score = st.slider("Frequency Score (1-5)", 1, 5, 3, help="5 = Frequent purchaser")
        with col3:
            m_score = st.slider("Monetary Score (1-5)", 1, 5, 3, help="5 = High spender")
        
        # Additional features with default values
        product_diversity = 0.5
        premium_product_ratio = 0.3
        web_purchase_ratio = 0.6
        store_purchase_ratio = 0.4
        has_complained = 0
        customer_value_score = clv_estimate / 100
        is_active = 1
        is_campaign_responder = 1 if campaign_acceptance > 0.3 else 0
        is_web_shopper = 1 if web_purchase_ratio > 0.5 else 0
        is_deal_seeker = 0
        
        submit_button = st.form_submit_button("🎯 Predict Segment", use_container_width=True)
    
    if submit_button:
        # Prepare features in correct order
        feature_values = []
        feature_mapping = {
            'R_Score': r_score,
            'F_Score': f_score,
            'M_Score': m_score,
            'Age': age,
            'Income': income,
            'Education_Level': education_level,
            'Family_Size': family_size,
            'Has_Children': has_children,
            'Avg_Order_Value': avg_order_value,
            'Purchase_Frequency_Rate': purchase_freq,
            'Customer_Tenure_Days': customer_tenure,
            'Wine_Ratio': wine_ratio,
            'Meat_Ratio': meat_ratio,
            'Product_Diversity': product_diversity,
            'Premium_Product_Ratio': premium_product_ratio,
            'Web_Purchase_Ratio': web_purchase_ratio,
            'Store_Purchase_Ratio': store_purchase_ratio,
            'Campaign_Acceptance_Rate': campaign_acceptance,
            'NumWebVisitsMonth': web_visits,
            'Has_Complained': has_complained,
            'Engagement_Score': engagement_score,
            'CLV_Estimate': clv_estimate,
            'Customer_Value_Score': customer_value_score,
            'Is_High_Spender': is_high_spender,
            'Is_Active': is_active,
            'Is_Campaign_Responder': is_campaign_responder,
            'Is_Web_Shopper': is_web_shopper,
            'Is_Deal_Seeker': is_deal_seeker
        }
        
        # Build feature array in correct order
        for feature in config['features']:
            feature_values.append(feature_mapping.get(feature, 0))
        
        # Make prediction
        X_input = np.array(feature_values).reshape(1, -1)
        prediction = model.predict(X_input)[0]
        probabilities = model.predict_proba(X_input)[0]
        confidence = probabilities[prediction]
        
        # Display results
        st.markdown("---")
        st.markdown("### 🎯 Prediction Results")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown('<div class="success-box">', unsafe_allow_html=True)
            st.markdown(f"### Predicted Segment")
            st.markdown(f"# {prediction}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="success-box">', unsafe_allow_html=True)
            st.markdown(f"### Confidence Score")
            st.markdown(f"# {confidence*100:.1f}%")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col3:
            confidence_level = "High" if confidence >= 0.8 else "Medium" if confidence >= 0.6 else "Low"
            color_class = "success-box" if confidence >= 0.8 else "warning-box" if confidence >= 0.6 else "warning-box"
            st.markdown(f'<div class="{color_class}">', unsafe_allow_html=True)
            st.markdown(f"### Confidence Level")
            st.markdown(f"# {confidence_level}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Probability distribution
        st.markdown("#### 📊 Probability Distribution Across Segments")
        
        fig = go.Figure(data=[
            go.Bar(
                x=[f'Segment {i}' for i in range(len(probabilities))],
                y=probabilities,
                marker=dict(
                    color=probabilities,
                    colorscale='Viridis',
                    showscale=False
                ),
                text=[f'{p*100:.1f}%' for p in probabilities],
                textposition='auto',
            )
        ])
        
        fig.update_layout(
            xaxis_title="Segment",
            yaxis_title="Probability",
            template="plotly_white",
            height=400,
            yaxis=dict(range=[0, 1])
        )
        
        st.plotly_chart(fig,width='stretch')
        
        # Segment characteristics comparison
        if df is not None:
            st.markdown("#### 📈 How This Customer Compares to Segment Average")
            
            segment_data = df[df['Final_ML_Cluster'] == prediction]
            
            comparison_metrics = {
                'Age': age,
                'Income': income,
                'CLV_Estimate': clv_estimate,
                'Engagement_Score': engagement_score
            }
            
            available_comparisons = {k: v for k, v in comparison_metrics.items() if k in segment_data.columns}
            
            if available_comparisons:
                comparison_data = pd.DataFrame({
                    'Metric': list(available_comparisons.keys()),
                    'Customer': list(available_comparisons.values()),
                    'Segment Average': [segment_data[k].mean() for k in available_comparisons.keys()]
                })
                
                fig = go.Figure()
                
                fig.add_trace(go.Bar(
                    name='Customer',
                    x=comparison_data['Metric'],
                    y=comparison_data['Customer'],
                    marker_color='rgb(102, 126, 234)'
                ))
                
                fig.add_trace(go.Bar(
                    name='Segment Average',
                    x=comparison_data['Metric'],
                    y=comparison_data['Segment Average'],
                    marker_color='rgb(118, 75, 162)'
                ))
                
                fig.update_layout(
                    barmode='group',
                    template="plotly_white",
                    height=400,
                    xaxis_title="Metric",
                    yaxis_title="Value"
                )
                
                st.plotly_chart(fig,width='stretch')

# =============================================================================
# PAGE 3: CLUSTER ANALYSIS
# =============================================================================

elif page == "📊 Cluster Analysis":
    st.title("📊 Cluster Deep Dive Analysis")
    st.markdown("### Explore detailed characteristics of each customer segment")
    
    if df is None:
        st.error("Unable to load data.")
        st.stop()
    
    # Cluster selector
    selected_cluster = st.selectbox(
        "Select a Segment to Analyze",
        options=sorted(df['Final_ML_Cluster'].unique()),
        format_func=lambda x: f"Segment {x}"
    )

    cluster_data = df[df['Final_ML_Cluster'] == selected_cluster]
    other_data = df[df['Final_ML_Cluster'] != selected_cluster]
    
    # Cluster Overview
    st.markdown(f"### 📌 Segment {selected_cluster} Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Customer Count", f"{len(cluster_data):,}")
    with col2:
        percentage = len(cluster_data) / len(df) * 100
        st.metric("Percentage", f"{percentage:.1f}%")
    with col3:
        if 'Total_Spending' in df.columns:
            avg_spending = cluster_data['Total_Spending'].mean()
            st.metric("Avg Spending", f"${avg_spending:,.0f}")
    with col4:
        if 'CLV_Estimate' in df.columns:
            avg_clv = cluster_data['CLV_Estimate'].mean()
            st.metric("Avg CLV", f"${avg_clv:,.0f}")
    
    st.markdown("---")
    
    # Demographics
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 👥 Demographics")
        
        demo_features = ['Age', 'Income', 'Family_Size']
        available_demo = [f for f in demo_features if f in df.columns]
        
        if available_demo:
            demo_comparison = pd.DataFrame({
                'Metric': available_demo,
                'This Segment': [cluster_data[f].mean() for f in available_demo],
                'Other Segments': [other_data[f].mean() for f in available_demo]
            })
            
            fig = go.Figure()
            
            fig.add_trace(go.Bar(
                name='This Segment',
                x=demo_comparison['Metric'],
                y=demo_comparison['This Segment'],
                marker_color='rgb(102, 126, 234)'
            ))
            
            fig.add_trace(go.Bar(
                name='Other Segments',
                x=demo_comparison['Metric'],
                y=demo_comparison['Other Segments'],
                marker_color='rgba(118, 75, 162, 0.5)'
            ))
            
            fig.update_layout(
                barmode='group',
                template="plotly_white",
                height=400,
                showlegend=True
            )
            
            st.plotly_chart(fig, width='stretch')
    
    with col2:
        st.markdown("### 🛒 Spending Patterns")
        
        product_cols = ['MntWines', 'MntMeatProducts', 'MntFishProducts', 
                       'MntFruits', 'MntSweetProducts', 'MntGoldProds']
        available_products = [c for c in product_cols if c in df.columns]
        
        if available_products:
            product_spending = cluster_data[available_products].mean()
            
            fig = go.Figure(data=[
                go.Pie(
                    labels=[c.replace('Mnt', '').replace('Products', '') for c in available_products],
                    values=product_spending.values,
                    hole=0.4,
                    marker=dict(colors=px.colors.qualitative.Set3)
                )
            ])
            
            fig.update_layout(
                template="plotly_white",
                height=400,
                showlegend=True
            )
            
            st.plotly_chart(fig, width='stretch')
    
    # RFM Analysis
    st.markdown("---")
    st.markdown("### 🎯 RFM Profile")
    
    rfm_features = ['R_Score', 'F_Score', 'M_Score']
    if all(f in df.columns for f in rfm_features):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            r_avg = cluster_data['R_Score'].mean()
            st.metric("Recency Score", f"{r_avg:.2f}", 
                     delta=f"{r_avg - other_data['R_Score'].mean():.2f} vs others")
        
        with col2:
            f_avg = cluster_data['F_Score'].mean()
            st.metric("Frequency Score", f"{f_avg:.2f}",
                     delta=f"{f_avg - other_data['F_Score'].mean():.2f} vs others")
        
        with col3:
            m_avg = cluster_data['M_Score'].mean()
            st.metric("Monetary Score", f"{m_avg:.2f}",
                     delta=f"{m_avg - other_data['M_Score'].mean():.2f} vs others")
    
    # Engagement
    st.markdown("---")
    st.markdown("### 📈 Engagement Metrics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if 'Campaign_Acceptance_Rate' in df.columns:
            acceptance_rate = cluster_data['Campaign_Acceptance_Rate'].mean()
            other_rate = other_data['Campaign_Acceptance_Rate'].mean()
            
            fig = go.Figure()
            
            fig.add_trace(go.Indicator(
                mode="gauge+number+delta",
                value=acceptance_rate * 100,
                delta={'reference': other_rate * 100, 'suffix': "%"},
                title={'text': "Campaign Acceptance Rate"},
                gauge={
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "rgb(102, 126, 234)"},
                    'steps': [
                        {'range': [0, 33], 'color': "rgba(255, 0, 0, 0.3)"},
                        {'range': [33, 66], 'color': "rgba(255, 255, 0, 0.3)"},
                        {'range': [66, 100], 'color': "rgba(0, 255, 0, 0.3)"}
                    ],
                }
            ))
            
            fig.update_layout(height=300, template="plotly_white")
            st.plotly_chart(fig, width='stretch')
    
    with col2:
        if 'Engagement_Score' in df.columns:
            engagement_score = cluster_data['Engagement_Score'].mean()
            other_score = other_data['Engagement_Score'].mean()
            
            fig = go.Figure()
            
            fig.add_trace(go.Indicator(
                mode="gauge+number+delta",
                value=engagement_score,
                delta={'reference': other_score},
                title={'text': "Engagement Score"},
                gauge={
                    'axis': {'range': [0, 10]},
                    'bar': {'color': "rgb(118, 75, 162)"},
                    'steps': [
                        {'range': [0, 3.33], 'color': "rgba(255, 0, 0, 0.3)"},
                        {'range': [3.33, 6.66], 'color': "rgba(255, 255, 0, 0.3)"},
                        {'range': [6.66, 10], 'color': "rgba(0, 255, 0, 0.3)"}
                    ],
                }
            ))
            
            fig.update_layout(height=300, template="plotly_white")
            st.plotly_chart(fig, width='stretch')

# =============================================================================
# PAGE 4: MODEL PERFORMANCE
# =============================================================================

elif page == "📈 Model Performance":
    st.title("📈 Model Performance Metrics")
    st.markdown("### Detailed evaluation of predictive models")
    
    # Load model comparison
    try:
        comparison_df = pd.read_csv('D://My projects/customer_segmentation_project/outputs/reports/step8_model_comparison.csv')
        
        st.markdown("### 🏆 Model Comparison")
        
        # Display metrics table
        st.dataframe(
            comparison_df.style.highlight_max(axis=0, subset=['Accuracy', 'Precision', 'Recall', 'F1-Score']),
             width='stretch'
        )
        
        # Visualize comparison
        fig = go.Figure()
        
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        colors = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12']
        
        for i, metric in enumerate(metrics):
            fig.add_trace(go.Bar(
                name=metric,
                x=comparison_df['Model'],
                y=comparison_df[metric],
                marker_color=colors[i],
                text=comparison_df[metric].round(3),
                textposition='auto',
            ))
        
        fig.update_layout(
            barmode='group',
            template="plotly_white",
            height=500,
            xaxis_title="Model",
            yaxis_title="Score",
            yaxis=dict(range=[0, 1]),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        st.plotly_chart(fig,width='stretch')
        
    except Exception as e:
        st.warning(f"Could not load model comparison: {e}")
    
    # Feature Importance
    st.markdown("---")
    st.markdown("### 🎯 Feature Importance")
    
    try:
        importance_df = pd.read_csv('D://My projects/customer_segmentation_project/outputs/reports/step8_feature_importance.csv')
        
        top_n = st.slider("Number of features to display", 5, 30, 15)
        
        top_features = importance_df.head(top_n)
        
        fig = go.Figure(data=[
            go.Bar(
                x=top_features['Importance'],
                y=top_features['Feature'],
                orientation='h',
                marker=dict(
                    color=top_features['Importance'],
                    colorscale='Viridis',
                    showscale=True
                ),
                text=top_features['Importance'].round(4),
                textposition='auto',
            )
        ])
        
        fig.update_layout(
            template="plotly_white",
            height=max(400, top_n * 25),
            xaxis_title="Importance Score",
            yaxis_title="Feature",
            yaxis={'categoryorder': 'total ascending'}
        )
        
        st.plotly_chart(fig, width='stretch')
        
    except Exception as e:
        st.warning(f"Could not load feature importance: {e}")
    
    # Model info
    st.markdown("---")
    st.markdown("### ℹ️ Model Information")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown('<div class="info-box">', unsafe_allow_html=True)
        st.markdown("#### Best Model")
        st.markdown(f"**{config['best_model']}**")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="info-box">', unsafe_allow_html=True)
        st.markdown("#### Accuracy")
        st.markdown(f"**{config['accuracy']*100:.2f}%**")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="info-box">', unsafe_allow_html=True)
        st.markdown("#### Features Used")
        st.markdown(f"**{config['n_features']}**")
        st.markdown('</div>', unsafe_allow_html=True)

# =============================================================================
# PAGE 5: ABOUT
# =============================================================================

elif page == "ℹ️ About":
    st.title("ℹ️ About This Dashboard")
    
    st.markdown("""
    ### 🎯 Customer Segmentation Platform
    
    This interactive dashboard provides advanced customer segmentation and prediction capabilities 
    using machine learning techniques.
    
    #### 📊 Features:
    
    - **Dashboard**: Overview of customer distribution and key metrics
    - **Predict Customer**: Real-time prediction of customer segments
    - **Cluster Analysis**: Deep dive into each customer segment
    - **Model Performance**: Evaluation metrics and feature importance
    
    #### 🔬 Methodology:
    
    1. **Data Collection**: Customer transaction and demographic data
    2. **Feature Engineering**: RFM analysis, behavioral metrics, engagement scores
    3. **Clustering**: K-Means algorithm for customer segmentation
    4. **Dimensionality Reduction**: PCA for visualization
    5. **Predictive Modeling**: Random Forest, Gradient Boosting, Logistic Regression
    6. **Validation**: Cross-validation and performance metrics
    
    #### 🛠️ Technologies:
    
    - **Frontend**: Streamlit with Plotly visualizations
    - **ML Models**: Scikit-learn
    - **Data Processing**: Pandas, NumPy
    - **Visualization**: Plotly, Seaborn, Matplotlib
    
    #### 📈 Business Value:
    
    - **Targeted Marketing**: Create personalized campaigns for each segment
    - **Resource Optimization**: Focus efforts on high-value customers
    - **Churn Prevention**: Identify at-risk customers early
    - **Product Development**: Understand preferences by segment
    - **Revenue Growth**: Maximize lifetime customer value
    
    #### 🎨 Design Philosophy:
    
    This dashboard is designed to work seamlessly in both light and dark modes, with:
    - Adaptive color schemes
    - High contrast for readability
    - Intuitive navigation
    - Responsive layouts
    - Interactive visualizations
    
    ---
    
    ### 📞 Support
    
    For questions or feedback, please contact your data science team.
    
    **Version**: 1.0.0  
    **Last Updated**: December 2024
    """)
    
    # System Info
    st.markdown("---")
    st.markdown("### 🖥️ System Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if df is not None:
            st.info(f"""
            **Data Statistics**
            - Total Records: {len(df):,}
            - Features: {len(config['features'])}
            - Segments: {config['n_clusters']}
            - Model Accuracy: {config['accuracy']*100:.2f}%
            """)
    
    with col2:
        st.success("""
        **Model Status**
        - ✅ Models Loaded
        - ✅ Data Available
        - ✅ PCA Configured
        - ✅ Ready for Predictions
        """)

# =============================================================================
# FOOTER
# =============================================================================

st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray; padding: 1rem;'>
        <p>Customer Segmentation Dashboard | Powered by Machine Learning</p>
        <p style='font-size: 0.8rem;'>© 2024 - Built with Streamlit & Plotly</p>
    </div>
    """,
    unsafe_allow_html=True
)
