import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Audio Tech NA eCommerce Analytics",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- ENHANCED CUSTOM CSS ---
st.markdown("""
<style>
    /* Main background */
    .main {
        background-color: #0e1117;
    }

    /* Metric containers */
    div[data-testid="metric-container"] {
        background: linear-gradient(135deg, #1e2530 0%, #2d3748 100%);
        border: 1px solid #3d4758;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
    }

    /* Metric labels */
    div[data-testid="metric-container"] label {
        color: #a0aec0 !important;
        font-size: 14px !important;
        font-weight: 500 !important;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    /* Metric values */
    div[data-testid="metric-container"] [data-testid="stMetricValue"] {
        color: #ffffff !important;
        font-size: 32px !important;
        font-weight: 700 !important;
    }

    /* Metric delta - positive */
    div[data-testid="metric-container"] [data-testid="stMetricDelta"] svg[fill="#09ab3b"] {
        fill: #10b981 !important;
    }

    div[data-testid="metric-container"] [data-testid="stMetricDelta"] {
        color: #10b981 !important;
        font-size: 14px !important;
        font-weight: 600 !important;
    }

    /* Metric delta - negative */
    div[data-testid="metric-container"] [data-testid="stMetricDelta"] svg[fill="#ff2b2b"] {
        fill: #ef4444 !important;
    }

    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #1e2530;
        padding: 10px;
        border-radius: 10px;
    }

    .stTabs [data-baseweb="tab"] {
        background-color: #2d3748;
        color: #a0aec0;
        border-radius: 8px;
        padding: 12px 24px;
        font-weight: 600;
        border: 1px solid #3d4758;
    }

    .stTabs [data-baseweb="tab"]:hover {
        background-color: #374151;
        color: #ffffff;
    }

    .stTabs [aria-selected="true"] {
        background-color: #3b82f6 !important;
        color: #ffffff !important;
        border-color: #3b82f6 !important;
    }

    /* Subheaders */
    .stMarkdown h3 {
        color: #ffffff !important;
        font-weight: 600 !important;
        margin-bottom: 20px !important;
    }

    /* Expander */
    .streamlit-expanderHeader {
        background-color: #2d3748 !important;
        color: #ffffff !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
    }

    .streamlit-expanderContent {
        background-color: #1e2530 !important;
        border: 1px solid #3d4758 !important;
        border-radius: 8px !important;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #1e2530 !important;
    }

    section[data-testid="stSidebar"] .stMarkdown {
        color: #ffffff !important;
    }

    /* Info/Success/Warning boxes */
    .stAlert {
        background-color: #2d3748 !important;
        border: 1px solid #3d4758 !important;
        border-radius: 8px !important;
        color: #ffffff !important;
    }

    /* Dataframe */
    .stDataFrame {
        background-color: #2d3748 !important;
    }

    /* Title */
    h1 {
        color: #ffffff !important;
        font-weight: 700 !important;
    }
</style>
""", unsafe_allow_html=True)


# --- DATA LOADING ---
@st.cache_data
def load_data():
    """加载电商数据集"""
    np.random.seed(42)
    start_date = datetime(2024, 1, 1)
    end_date = datetime(2024, 12, 31)
    date_range = pd.date_range(start_date, end_date)

    # Customers
    num_customers = 1500
    df_customers = pd.DataFrame({
        'customer_id': [f'CUST-{i:04d}' for i in range(num_customers)],
        'region': np.random.choice(['US-East', 'US-West', 'Canada', 'US-Central'], num_customers),
        'signup_date': [start_date + timedelta(days=np.random.randint(0, 300)) for _ in range(num_customers)],
        'acquisition_channel': np.random.choice(['Organic', 'Paid Search', 'Social', 'Email'],
                                                num_customers, p=[0.3, 0.35, 0.25, 0.1])
    })

    # Orders
    orders = []
    skus = {
        'AudioPro X1': 179.95,
        'WaterSound S2': 149.95,
        'ComfortFit C3': 169.95,
        'ActiveMove A4': 79.95
    }

    for _, cust in df_customers.iterrows():
        num_purchases = np.random.choice([1, 2, 3], p=[0.7, 0.2, 0.1])

        for purchase_num in range(num_purchases):
            days_after_signup = np.random.randint(0, 180)
            p_date = cust['signup_date'] + timedelta(days=days_after_signup)

            if p_date > end_date:
                continue

            is_bfcm = (p_date.month == 11 and 20 <= p_date.day <= 30)
            order_multiplier = 2 if is_bfcm else 1

            for _ in range(order_multiplier):
                sku = np.random.choice(list(skus.keys()), p=[0.4, 0.2, 0.2, 0.2])
                platform = np.random.choice(['Amazon', 'Shopify'], p=[0.65, 0.35])
                base_price = skus[sku]

                if is_bfcm:
                    discount = np.random.uniform(0.15, 0.25)
                else:
                    discount = np.random.uniform(0, 0.05)

                final_price = base_price * (1 - discount)
                is_returned = np.random.random() < 0.05
                actual_revenue = 0 if is_returned else final_price

                orders.append([
                    cust['customer_id'],
                    p_date,
                    platform,
                    sku,
                    actual_revenue,
                    discount,
                    is_returned
                ])

    df_orders = pd.DataFrame(orders, columns=[
        'customer_id', 'date', 'platform', 'sku', 'revenue', 'discount_rate', 'returned'
    ])

    # Marketing
    mkt = []
    for d in date_range:
        if d.month == 11:
            spend = np.random.normal(3500, 300)
        else:
            spend = np.random.normal(800, 80)

        cpc = round(np.random.uniform(0.7, 1.2), 2)
        clicks = int(spend / cpc)
        conversions = int(clicks * np.random.uniform(0.02, 0.04))
        impressions = int(clicks / 0.02)

        mkt.append([d, spend, impressions, clicks, cpc, conversions])

    df_marketing = pd.DataFrame(mkt, columns=[
        'date', 'ad_spend', 'impressions', 'clicks', 'cpc', 'conversions'
    ])

    return df_orders, df_customers, df_marketing


df_orders, df_customers, df_marketing = load_data()

# --- SIDEBAR FILTERS ---
st.sidebar.markdown("""
<div style='text-align: center; padding: 20px;'>
    <h2 style='color: #3b82f6; margin: 0;'>🎧 SoundWave</h2>
    <p style='color: #a0aec0; font-size: 12px; margin-top: 5px;'>Audio Tech Analytics</p>
</div>
""", unsafe_allow_html=True)

st.sidebar.header("🔍 Filter Options")

selected_platform = st.sidebar.multiselect(
    "Platform",
    options=['Amazon', 'Shopify'],
    default=['Amazon', 'Shopify']
)

date_filter = st.sidebar.date_input(
    "Date Range",
    [df_orders['date'].min().date(), df_orders['date'].max().date()]
)

selected_region = st.sidebar.multiselect(
    "Region",
    options=df_customers['region'].unique().tolist(),
    default=df_customers['region'].unique().tolist()
)

# --- DATA FILTERING ---
if len(date_filter) == 2:
    mask_orders = (
            (df_orders['platform'].isin(selected_platform)) &
            (df_orders['date'].dt.date >= date_filter[0]) &
            (df_orders['date'].dt.date <= date_filter[1])
    )
    filtered_orders = df_orders[mask_orders].copy()

    filtered_customers = df_customers[df_customers['region'].isin(selected_region)]
    filtered_orders = filtered_orders[filtered_orders['customer_id'].isin(filtered_customers['customer_id'])]

    mask_marketing = (
            (df_marketing['date'].dt.date >= date_filter[0]) &
            (df_marketing['date'].dt.date <= date_filter[1])
    )
    filtered_marketing = df_marketing[mask_marketing].copy()
else:
    filtered_orders = df_orders[df_orders['platform'].isin(selected_platform)].copy()
    filtered_marketing = df_marketing.copy()

# --- CALCULATE METRICS ---
total_rev = filtered_orders['revenue'].sum()
total_orders = len(filtered_orders[filtered_orders['revenue'] > 0])
aov = total_rev / total_orders if total_orders > 0 else 0
total_spend = filtered_marketing['ad_spend'].sum()
roas = total_rev / total_spend if total_spend > 0 else 0
return_rate = (filtered_orders['returned'].sum() / len(filtered_orders) * 100) if len(filtered_orders) > 0 else 0

ly_revenue = total_rev / 1.12
ly_orders = int(total_orders / 1.08)
ly_aov = aov * 1.02
ly_roas = roas / 1.15
ly_return_rate = return_rate + 0.5

delta_rev = ((total_rev - ly_revenue) / ly_revenue * 100) if ly_revenue > 0 else 0
delta_orders = ((total_orders - ly_orders) / ly_orders * 100) if ly_orders > 0 else 0
delta_aov = ((aov - ly_aov) / ly_aov * 100) if ly_aov > 0 else 0
delta_roas = ((roas - ly_roas) / ly_roas * 100) if ly_roas > 0 else 0
delta_return = return_rate - ly_return_rate

# --- TOP METRICS ---
st.title("🎧 North America eCommerce Performance Dashboard")
st.markdown("---")

col1, col2, col3, col4, col5 = st.columns(5)

col1.metric(
    "Total Revenue",
    f"${total_rev:,.0f}",
    f"{delta_rev:+.1f}% vs LY"
)
col2.metric(
    "Total Orders",
    f"{total_orders:,}",
    f"{delta_orders:+.1f}% vs LY"
)
col3.metric(
    "AOV",
    f"${aov:.2f}",
    f"{delta_aov:+.1f}% vs LY"
)
col4.metric(
    "ROAS",
    f"{roas:.2f}x",
    f"{delta_roas:+.1f}% vs LY"
)
col5.metric(
    "Return Rate",
    f"{return_rate:.1f}%",
    f"{delta_return:+.1f}pp vs LY",
    delta_color="inverse"
)

# --- TABS ---
tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Executive Summary",
    "📊 Channel & Ad Performance",
    "📦 Product & Forecast",
    "👥 Customer RFM"
])

# --- TAB 1: EXECUTIVE SUMMARY ---
with tab1:
    col_left, col_right = st.columns([2, 1])

    with col_left:
        st.subheader("Daily Revenue Trend")
        daily_rev = filtered_orders.groupby('date')['revenue'].sum().reset_index()

        if len(daily_rev) > 0:
            fig_line = px.line(daily_rev, x='date', y='revenue',
                               title="Revenue over Time (NA Market)")
            fig_line.add_vrect(
                x0="2024-11-20", x1="2024-11-30",
                fillcolor="green", opacity=0.2,
                annotation_text="BFCM Period",
                annotation_position="top left"
            )
            fig_line.update_layout(
                xaxis_title="Date",
                yaxis_title="Revenue ($)",
                hovermode='x unified',
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#ffffff')
            )
            st.plotly_chart(fig_line, use_container_width=True)
        else:
            st.info("No data available for the selected filters")

    with col_right:
        st.subheader("Revenue by Region")
        region_rev = filtered_orders.merge(df_customers, on='customer_id') \
            .groupby('region')['revenue'].sum().reset_index()

        if len(region_rev) > 0:
            fig_region = px.bar(region_rev, x='region', y='revenue',
                                color='region', text='revenue')
            fig_region.update_traces(texttemplate='$%{text:,.0f}', textposition='outside')
            fig_region.update_layout(
                showlegend=False,
                xaxis_title="Region",
                yaxis_title="Revenue ($)",
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#ffffff')
            )
            st.plotly_chart(fig_region, use_container_width=True)
        else:
            st.info("No data available for the selected filters")

    with st.expander("💡 Business Insights - Executive Summary"):
        bfcm_revenue = filtered_orders[
            (filtered_orders['date'].dt.month == 11) &
            (filtered_orders['date'].dt.day >= 20) &
            (filtered_orders['date'].dt.day <= 30)
            ]['revenue'].sum()
        bfcm_pct = (bfcm_revenue / total_rev * 100) if total_rev > 0 else 0

        st.write(f"""
        - **Seasonality**: BFCM week accounts for ~{bfcm_pct:.1f}% of total revenue with 2x order volume
        - **Regional Performance**: Top region contributes {region_rev['revenue'].max() / total_rev * 100:.1f}% of revenue
        - **Action Plan**: Increase inventory buffer by 25% for premium products 4 weeks prior to Q4 peak
        """)

# --- TAB 2: CHANNEL & AD PERFORMANCE ---
with tab2:
    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Platform Revenue Distribution")
        plat_dist = filtered_orders.groupby('platform')['revenue'].sum().reset_index()

        if len(plat_dist) > 0:
            fig_pie = px.pie(plat_dist, values='revenue', names='platform',
                             hole=0.4, color_discrete_sequence=['#FF9900', '#95BF47'])
            fig_pie.update_traces(textposition='inside', textinfo='percent+label')
            fig_pie.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#ffffff')
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        else:
            st.info("No data available")

    with col_b:
        st.subheader("Advertising Metrics")
        total_clicks = filtered_marketing['clicks'].sum()
        total_conversions = filtered_marketing['conversions'].sum()
        total_impressions = filtered_marketing['impressions'].sum()

        cvr = (total_conversions / total_clicks * 100) if total_clicks > 0 else 0
        ctr = (total_clicks / total_impressions * 100) if total_impressions > 0 else 0
        cpa = total_spend / total_conversions if total_conversions > 0 else 0
        acos = (total_spend / total_rev) * 100 if total_rev > 0 else 0

        metric_col1, metric_col2 = st.columns(2)
        metric_col1.metric("ACOS", f"{acos:.2f}%")
        metric_col1.metric("CPA", f"${cpa:.2f}")
        metric_col2.metric("CVR", f"{cvr:.2f}%")
        metric_col2.metric("CTR", f"{ctr:.2f}%")

    st.markdown("---")
    st.subheader("Ad Spend vs Revenue Trend")

    daily_mkt = filtered_marketing.groupby('date')['ad_spend'].sum().reset_index()
    daily_rev_mkt = filtered_orders.groupby('date')['revenue'].sum().reset_index()
    combined = pd.merge(daily_mkt, daily_rev_mkt, on='date', how='left').fillna(0)

    if len(combined) > 0:
        fig_dual = go.Figure()
        fig_dual.add_trace(go.Bar(
            x=combined['date'],
            y=combined['ad_spend'],
            name='Ad Spend',
            yaxis='y',
            marker_color='#FF6B6B'
        ))
        fig_dual.add_trace(go.Scatter(
            x=combined['date'],
            y=combined['revenue'],
            name='Revenue',
            yaxis='y2',
            line=dict(color='#4ECDC4', width=3)
        ))

        fig_dual.update_layout(
            yaxis=dict(title='Ad Spend ($)', side='left'),
            yaxis2=dict(title='Revenue ($)', overlaying='y', side='right'),
            hovermode='x unified',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#ffffff')
        )
        st.plotly_chart(fig_dual, use_container_width=True)
    else:
        st.info("No data available")

    with st.expander("💡 Business Insights - Channel Strategy"):
        st.write("""
        - **Amazon Ads**: High volume but CPC volatility. Optimize 'Exact Match' keywords for better ACOS
        - **Shopify Growth**: D2C channel shows higher ROAS potential. Increase social ad budget
        - **BFCM Strategy**: Pre-campaign warm-up improves CVR significantly
        """)

# --- TAB 3: PRODUCT & FORECAST ---
with tab3:
    col_prod1, col_prod2 = st.columns([3, 2])

    with col_prod1:
        st.subheader("Revenue by SKU")
        sku_rev = filtered_orders[filtered_orders['revenue'] > 0].groupby('sku')['revenue'].agg(
            ['sum', 'count']).reset_index()
        sku_rev.columns = ['SKU', 'Total Revenue', 'Units Sold']
        sku_rev = sku_rev.sort_values('Total Revenue', ascending=False)

        if len(sku_rev) > 0:
            fig_bar = px.bar(sku_rev, x='SKU', y='Total Revenue',
                             color='SKU', text='Units Sold',
                             color_discrete_sequence=px.colors.qualitative.Set2)
            fig_bar.update_traces(texttemplate='%{text} units', textposition='outside')
            fig_bar.update_layout(
                showlegend=False,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#ffffff')
            )
            st.plotly_chart(fig_bar, use_container_width=True)
        else:
            st.info("No data available")

    with col_prod2:
        st.subheader("Product Performance")
        sku_metrics = filtered_orders.groupby('sku').agg({
            'revenue': 'sum',
            'returned': 'sum',
            'customer_id': 'count'
        }).reset_index()
        sku_metrics['return_rate'] = (sku_metrics['returned'] / sku_metrics['customer_id'] * 100)
        sku_metrics = sku_metrics.sort_values('revenue', ascending=False)

        st.dataframe(
            sku_metrics[['sku', 'revenue', 'return_rate']].style.format({
                'revenue': '${:,.0f}',
                'return_rate': '{:.1f}%'
            }),
            use_container_width=True,
            hide_index=True
        )

    st.markdown("---")
    st.subheader("🚀 30-Day Sales Forecast")

    daily_rev = filtered_orders.groupby('date')['revenue'].sum().reset_index()

    if len(daily_rev) >= 30:
        daily_rev['day_index'] = (daily_rev['date'] - daily_rev['date'].min()).dt.days

        X = daily_rev[['day_index']]
        y = daily_rev['revenue']
        model = LinearRegression().fit(X, y)
        r2 = r2_score(y, model.predict(X))

        future_days = np.array([[i] for i in range(
            daily_rev['day_index'].max() + 1,
            daily_rev['day_index'].max() + 31
        )])
        forecast = model.predict(future_days)

        future_dates = pd.date_range(
            daily_rev['date'].max() + timedelta(days=1),
            periods=30
        )
        forecast_df = pd.DataFrame({
            'date': future_dates,
            'revenue': forecast,
            'type': 'Forecast'
        })

        historical_df = daily_rev[['date', 'revenue']].copy()
        historical_df['type'] = 'Historical'

        combined_forecast = pd.concat([historical_df, forecast_df])

        fig_forecast = px.line(combined_forecast, x='date', y='revenue',
                               color='type',
                               title='Revenue Forecast with Linear Trend',
                               color_discrete_map={'Historical': '#1f77b4', 'Forecast': '#ff7f0e'})
        fig_forecast.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#ffffff')
        )
        st.plotly_chart(fig_forecast, use_container_width=True)

        col_f1, col_f2 = st.columns(2)
        col_f1.success(f"Estimated Revenue (Next 30 Days): **${forecast.sum():,.2f}**")
        col_f2.info(f"Model R² Score: **{r2:.3f}**")
    else:
        st.warning("Insufficient data for forecasting. Need at least 30 days of data.")

# --- TAB 4: CUSTOMER RFM ---
with tab4:
    st.subheader("Customer Segmentation (RFM Model)")

    if len(filtered_orders) > 0:
        rfm = filtered_orders.groupby('customer_id').agg({
            'date': lambda x: (datetime(2025, 1, 1) - x.max()).days,
            'customer_id': 'count',
            'revenue': 'sum'
        })
        rfm.columns = ['Recency', 'Frequency', 'Monetary']

        rfm['R_Score'] = pd.qcut(rfm['Recency'], 5, labels=[5, 4, 3, 2, 1], duplicates='drop')
        rfm['F_Score'] = pd.qcut(rfm['Frequency'].rank(method='first'), 5,
                                 labels=[1, 2, 3, 4, 5], duplicates='drop')
        rfm['M_Score'] = pd.qcut(rfm['Monetary'], 5, labels=[1, 2, 3, 4, 5], duplicates='drop')


        def segment_customer(row):
            r = int(row['R_Score'])
            f = int(row['F_Score'])
            if r >= 4 and f >= 4:
                return 'Champions'
            elif r >= 3 and f >= 3:
                return 'Loyal Customers'
            elif r <= 2:
                return 'At Risk'
            else:
                return 'Potential'


        rfm['Segment'] = rfm.apply(segment_customer, axis=1)

        col_rfm1, col_rfm2 = st.columns([2, 1])

        with col_rfm1:
            fig_scatter = px.scatter(
                rfm, x='Recency', y='Monetary',
                color='Segment', size='Frequency',
                title="Customer Value vs Recency Analysis",
                hover_data=['Frequency'],
                color_discrete_map={
                    'Champions': '#2ecc71',
                    'Loyal Customers': '#3498db',
                    'Potential': '#f39c12',
                    'At Risk': '#e74c3c'
                }
            )
            fig_scatter.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#ffffff')
            )
            st.plotly_chart(fig_scatter, use_container_width=True)

        with col_rfm2:
            st.markdown("#### Segment Distribution")
            segment_stats = rfm.groupby('Segment').agg({
                'Recency': 'count',
                'Monetary': 'sum'
            }).reset_index()
            segment_stats.columns = ['Segment', 'Count', 'Revenue']
            segment_stats['Avg Revenue'] = segment_stats['Revenue'] / segment_stats['Count']

            st.dataframe(
                segment_stats.style.format({
                    'Revenue': '${:,.0f}',
                    'Avg Revenue': '${:,.0f}'
                }),
                use_container_width=True,
                hide_index=True
            )

        with st.expander("💡 CRM Strategy Recommendations"):
            st.write("""
            - **Champions**: VIP early access to new products + loyalty rewards program
            - **Loyal Customers**: Trigger 'Bundle & Save' campaigns to increase basket size
            - **At Risk**: Win-back email series with 15% discount for inactive >90 days
            - **Potential**: Nurture with educational content and first-purchase incentives
            """)
    else:
        st.info("No customer data available for the selected filters")

# --- FOOTER ---
st.markdown("---")
st.caption("📊 Dashboard developed by MINOPEACHY | eCommerce Analytics Portfolio Project")
st.caption("⚠️ This dashboard uses simulated data for demonstration purposes only.")
