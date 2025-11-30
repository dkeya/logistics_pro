# pages/digital_intelligence/22_Digital_Overview.py
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import numpy as np
import sys
import os

# Add the core module to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

try:
    from logistics_core.analytics.digital_analytics import DigitalAnalyticsEngine
except ImportError:
    st.error("Digital Analytics Engine not found. Please run the initialization script first.")
    st.stop()


def safe_get_df(digital_data, key):
    """Safely get a DataFrame from the digital_data dict."""
    if isinstance(digital_data, dict) and key in digital_data:
        df = digital_data[key]
        if isinstance(df, pd.DataFrame):
            return df
    return pd.DataFrame()


def render():
    """🌐 DIGITAL INTELLIGENCE HUB - Unified Digital Command Center"""

    st.title("🌐 Digital Intelligence Hub")
    st.markdown(
        f"""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    padding: 20px; border-radius: 12px; color: white; margin-bottom: 12px;">
            <h3 style="margin: 0; color: white;">🎯 Unified Digital Command Center & Performance Intelligence</h3>
            <p style="margin: 5px 0 0 0; opacity: 0.9; color: white;">
                <strong>📍</strong> Digital Intelligence &gt; Overview |
                <strong>🏢</strong> {st.session_state.get('current_tenant', 'ELORA Holding')}
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 💚 Soft green marquee strip – aligned with 01_Dashboard style
    st.markdown(
        """
        <div style="background: #dcfce7; padding: 10px 16px; border-radius: 10px;
                    margin-bottom: 24px; border-left: 4px solid #16a34a;">
            <marquee behavior="scroll" direction="left" scrollamount="4"
                     style="font-size: 0.9rem; font-weight: 500; color: #166534;">
                🌐 <strong>Digital Pulse:</strong> Unified e-commerce, web and social intelligence • 
                🛒 <strong>E-commerce:</strong> Conversion uplift, cart value optimization, channel efficiency • 
                📱 <strong>Social:</strong> Engagement and sentiment tracking across key platforms • 
                🌍 <strong>Web:</strong> Traffic quality, user journeys, and bounce management • 
                ⚡ <strong>Ops:</strong> Inventory sync, fulfillment performance, and demand signals in one view
            </marquee>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Initialize digital analytics if not exists
    if 'digital_analytics' not in st.session_state:
        st.session_state.digital_analytics = DigitalAnalyticsEngine()
        # Generate 90 days of digital data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=90)
        st.session_state.digital_data = st.session_state.digital_analytics.generate_synthetic_digital_data(
            start_date.strftime('%Y-%m-%d'),
            end_date.strftime('%Y-%m-%d')
        )

    digital_data = st.session_state.get('digital_data')

    if not isinstance(digital_data, dict):
        st.error("❌ Digital data is not initialized correctly.")
        return

    # Main tabs for unified digital view
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🏅 Digital Health Dashboard",
        "🛒 Ecommerce Performance",
        "🌐 Web Analytics",
        "📱 Social Media",
        "⚡ Digital Operations"
    ])

    with tab1:
        render_digital_health_dashboard(digital_data)
    with tab2:
        render_ecommerce_performance(digital_data)
    with tab3:
        render_web_analytics(digital_data)
    with tab4:
        render_social_media(digital_data)
    with tab5:
        render_digital_operations(digital_data)


def render_digital_health_dashboard(digital_data):
    """Unified digital health dashboard"""

    st.header("🏅 Digital Health Dashboard")

    ecommerce_df = safe_get_df(digital_data, 'ecommerce')
    web_df = safe_get_df(digital_data, 'web_analytics')
    social_df = safe_get_df(digital_data, 'social_media')

    # Key Digital KPIs
    col1, col2, col3, col4 = st.columns(4)

    # --- Total Digital Revenue ---
    with col1:
        if not ecommerce_df.empty and 'revenue' in ecommerce_df.columns:
            total_revenue = ecommerce_df['revenue'].sum()
        else:
            total_revenue = 0.0
        st.metric("Total Digital Revenue", f"KES {total_revenue:,.0f}", "12.5%")

    # --- Total Website Sessions ---
    with col2:
        if not web_df.empty and 'sessions' in web_df.columns:
            total_sessions = web_df['sessions'].sum()
        else:
            total_sessions = 0
        st.metric("Total Website Sessions", f"{total_sessions:,.0f}", "8.3%")

    # --- Avg Social Sentiment ---
    with col3:
        if not social_df.empty and 'sentiment_score' in social_df.columns:
            avg_sentiment = social_df['sentiment_score'].mean() * 100
        else:
            avg_sentiment = 0.0
        st.metric("Avg. Social Sentiment", f"{avg_sentiment:.1f}%", "2.1%")

    # --- Avg Conversion Rate ---
    with col4:
        if not ecommerce_df.empty and 'conversion_rate' in ecommerce_df.columns:
            conversion_rate = ecommerce_df['conversion_rate'].mean()
        else:
            conversion_rate = 0.0
        st.metric("Avg. Conversion Rate", f"{conversion_rate:.2f}%", "0.4%")

    st.divider()

    # Cross-channel performance
    col1, col2 = st.columns(2)

    with col1:
        # Revenue by platform
        if not ecommerce_df.empty and {'platform', 'revenue'}.issubset(ecommerce_df.columns):
            platform_revenue = (
                ecommerce_df
                .groupby('platform')['revenue']
                .sum()
                .reset_index()
            )
            fig = px.pie(
                platform_revenue,
                values='revenue',
                names='platform',
                title="📊 Revenue Distribution by Platform"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("ℹ️ Ecommerce platform revenue data not available.")

    with col2:
        # Traffic sources
        if not web_df.empty and {'channel', 'sessions'}.issubset(web_df.columns):
            channel_sessions = (
                web_df
                .groupby('channel')['sessions']
                .sum()
                .reset_index()
            )
            fig = px.bar(
                channel_sessions,
                x='channel',
                y='sessions',
                title="🌐 Website Traffic by Channel",
                color='sessions'
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("ℹ️ Website traffic by channel data not available.")

    # Digital trends over time
    st.subheader("📈 Digital Performance Trends")

    if not ecommerce_df.empty and {'date', 'revenue'}.issubset(ecommerce_df.columns):
        revenue_trend = (
            ecommerce_df
            .groupby('date')['revenue']
            .sum()
            .reset_index()
        )
        fig = px.line(
            revenue_trend,
            x='date',
            y='revenue',
            title="💰 Daily Digital Revenue Trend"
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("ℹ️ Revenue trend data not available.")


def render_ecommerce_performance(digital_data):
    """Ecommerce performance analysis"""
    st.header("🛒 Ecommerce Performance")

    ecommerce_df = safe_get_df(digital_data, 'ecommerce')

    if ecommerce_df.empty:
        st.info("ℹ️ No ecommerce data available.")
        return

    # Platform comparison
    col1, col2 = st.columns(2)

    with col1:
        if {'platform', 'revenue', 'orders', 'visitors'}.issubset(ecommerce_df.columns):
            platform_metrics = (
                ecommerce_df
                .groupby('platform')
                .agg({
                    'revenue': 'sum',
                    'orders': 'sum',
                    'visitors': 'sum'
                })
                .reset_index()
            )

            platform_metrics['conversion_rate'] = (
                platform_metrics['orders'] / platform_metrics['visitors'] * 100
            ).replace([np.inf, -np.inf], np.nan).fillna(0)
            platform_metrics['aov'] = (
                platform_metrics['revenue'] / platform_metrics['orders']
            ).replace([np.inf, -np.inf], np.nan).fillna(0)

            fig = px.bar(
                platform_metrics,
                x='platform',
                y='revenue',
                title="💰 Revenue by Ecommerce Platform",
                color='revenue'
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("ℹ️ Insufficient columns to compute platform metrics.")

    with col2:
        if 'platform' in ecommerce_df.columns and 'revenue' in ecommerce_df.columns:
            # if platform_metrics exists from above branch, reuse; else recompute minimally
            if 'platform_metrics' not in locals():
                platform_metrics = (
                    ecommerce_df
                    .groupby('platform')
                    .agg({
                        'revenue': 'sum',
                        'orders': 'sum',
                        'visitors': 'sum'
                    })
                    .reset_index()
                )
                platform_metrics['conversion_rate'] = (
                    platform_metrics['orders'] / platform_metrics['visitors'] * 100
                ).replace([np.inf, -np.inf], np.nan).fillna(0)
                platform_metrics['aov'] = (
                    platform_metrics['revenue'] / platform_metrics['orders']
                ).replace([np.inf, -np.inf], np.nan).fillna(0)

            fig = px.scatter(
                platform_metrics,
                x='conversion_rate',
                y='aov',
                size='revenue',
                color='platform',
                title="🎯 Conversion Rate vs Average Order Value"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("ℹ️ Insufficient data to plot Conversion vs AOV.")

    # Product performance
    st.subheader("📦 Product Performance Analysis")
    if {'product', 'revenue', 'orders'}.issubset(ecommerce_df.columns):
        product_performance = (
            ecommerce_df
            .groupby('product')
            .agg({
                'revenue': 'sum',
                'orders': 'sum'
            })
            .reset_index()
        )

        fig = px.treemap(
            product_performance,
            path=['product'],
            values='revenue',
            title="🧩 Product Revenue Distribution"
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("ℹ️ Product-level ecommerce data not available.")


def render_web_analytics(digital_data):
    """Web analytics dashboard"""
    st.header("🌐 Web Analytics")

    web_df = safe_get_df(digital_data, 'web_analytics')

    if web_df.empty:
        st.info("ℹ️ No web analytics data available.")
        return

    col1, col2 = st.columns(2)

    with col1:
        # Traffic by channel
        if {'channel', 'sessions', 'users', 'pageviews'}.issubset(web_df.columns):
            channel_performance = (
                web_df
                .groupby('channel')
                .agg({
                    'sessions': 'sum',
                    'users': 'sum',
                    'pageviews': 'sum'
                })
                .reset_index()
            )

            fig = px.funnel(
                channel_performance,
                x='sessions',
                y='channel',
                title="🛠️ Traffic Acquisition Funnel"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("ℹ️ Channel-level web analytics not available.")

    with col2:
        # Device performance
        if {'device', 'sessions', 'bounce_rate', 'avg_session_duration'}.issubset(web_df.columns):
            device_performance = (
                web_df
                .groupby('device')
                .agg({
                    'sessions': 'sum',
                    'bounce_rate': 'mean',
                    'avg_session_duration': 'mean'
                })
                .reset_index()
            )

            fig = px.bar(
                device_performance,
                x='device',
                y='sessions',
                title="📱 Sessions by Device Type",
                color='sessions'
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("ℹ️ Device-level performance data not available.")

    # Behavioral metrics over time
    st.subheader("📊 User Behavior Trends")
    if {'date', 'sessions', 'bounce_rate', 'avg_session_duration'}.issubset(web_df.columns):
        behavior_trends = (
            web_df
            .groupby('date')
            .agg({
                'sessions': 'sum',
                'bounce_rate': 'mean',
                'avg_session_duration': 'mean'
            })
            .reset_index()
        )

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=behavior_trends['date'],
            y=behavior_trends['sessions'],
            mode='lines',
            name='Sessions',
            yaxis='y1'
        ))
        fig.add_trace(go.Scatter(
            x=behavior_trends['date'],
            y=behavior_trends['bounce_rate'],
            mode='lines',
            name='Bounce Rate %',
            yaxis='y2'
        ))

        fig.update_layout(
            title="📈 Sessions vs Bounce Rate Trend",
            yaxis=dict(title="Sessions"),
            yaxis2=dict(
                title="Bounce Rate %",
                overlaying='y',
                side='right'
            )
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("ℹ️ Behavioural trend data not available.")


def render_social_media(digital_data):
    """Social media intelligence"""
    st.header("📱 Social Media Intelligence")

    social_df = safe_get_df(digital_data, 'social_media')
    competitive_df = safe_get_df(digital_data, 'competitive')

    col1, col2 = st.columns(2)

    with col1:
        # Platform engagement
        if not social_df.empty and {'platform', 'followers', 'engagement_rate', 'sentiment_score'}.issubset(social_df.columns):
            platform_engagement = (
                social_df
                .groupby('platform')
                .agg({
                    'followers': 'last',
                    'engagement_rate': 'mean',
                    'sentiment_score': 'mean'
                })
                .reset_index()
            )

            fig = px.bar(
                platform_engagement,
                x='platform',
                y='engagement_rate',
                title="💬 Average Engagement Rate by Platform",
                color='engagement_rate'
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("ℹ️ Social engagement data not available.")

    with col2:
        # Sentiment analysis
        if not social_df.empty and {'date', 'sentiment_score'}.issubset(social_df.columns):
            sentiment_trend = (
                social_df
                .groupby('date')['sentiment_score']
                .mean()
                .reset_index()
            )
            fig = px.area(
                sentiment_trend,
                x='date',
                y='sentiment_score',
                title="😊 Social Media Sentiment Trend"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("ℹ️ Sentiment trend data not available.")

    # Competitive social landscape
    st.subheader("🏅 Competitive Social Landscape")
    if not competitive_df.empty and {'market_share', 'social_mentions', 'review_rating', 'competitor'}.issubset(competitive_df.columns):
        fig = px.scatter(
            competitive_df,
            x='market_share',
            y='social_mentions',
            size='review_rating',
            color='competitor',
            title="🎯 Market Share vs Social Mentions"
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("ℹ️ Competitive landscape data not available.")


def render_digital_operations(digital_data):
    """Digital operations and supply chain integration"""
    st.header("⚡ Digital Operations")

    ecommerce_df = safe_get_df(digital_data, 'ecommerce')
    web_df = safe_get_df(digital_data, 'web_analytics')

    # Inventory synchronization needs
    st.subheader("🔁 Ecommerce Inventory Health")

    # Simulate inventory sync status
    platforms = ['Amazon', 'Shopify', 'WooCommerce', 'eBay']
    sync_status = []

    for platform in platforms:
        if not ecommerce_df.empty and {'platform', 'orders'}.issubset(ecommerce_df.columns):
            platform_data = ecommerce_df[ecommerce_df['platform'] == platform]
            recent_orders = platform_data['orders'].sum()
        else:
            recent_orders = 0

        stockout_risk = (
            'Low' if recent_orders < 1000
            else 'Medium' if recent_orders < 2000
            else 'High'
        )

        sync_status.append({
            'platform': platform,
            'recent_orders': recent_orders,
            'sync_status': 'In Sync',
            'last_sync': '2 hours ago',
            'stockout_risk': stockout_risk
        })

    sync_df = pd.DataFrame(sync_status)
    st.dataframe(sync_df, use_container_width=True)

    # Fulfillment performance
    st.subheader("🚚 Digital Fulfillment Metrics")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Avg. Order Processing Time", "2.3 hours", "-0.5 hours")

    with col2:
        st.metric("On-time Delivery Rate", "96.7%", "1.2%")

    with col3:
        st.metric("Digital Return Rate", "3.2%", "-0.8%")

    # Demand forecasting integration
    st.subheader("📊 Digital Demand Signals")

    # Correlation between web traffic and sales
    if not web_df.empty and 'date' in web_df.columns and 'sessions' in web_df.columns:
        web_traffic = (
            web_df
            .groupby('date')['sessions']
            .sum()
            .reset_index()
        )
    else:
        web_traffic = pd.DataFrame(columns=['date', 'sessions'])

    if not ecommerce_df.empty and 'date' in ecommerce_df.columns and 'orders' in ecommerce_df.columns:
        sales_data = (
            ecommerce_df
            .groupby('date')['orders']
            .sum()
            .reset_index()
        )
    else:
        sales_data = pd.DataFrame(columns=['date', 'orders'])

    if not web_traffic.empty and not sales_data.empty:
        correlation_data = pd.merge(web_traffic, sales_data, on='date')
    else:
        correlation_data = pd.DataFrame(columns=['date', 'sessions', 'orders'])

    if len(correlation_data) >= 2:
        correlation = correlation_data['sessions'].corr(correlation_data['orders'])
        correlation_text = (
            "Strong positive correlation" if correlation > 0.7
            else "Moderate correlation" if correlation > 0.3
            else "Weak correlation"
        )
        st.metric("Traffic-Sales Correlation", f"{correlation:.3f}", correlation_text)
    else:
        st.metric("Traffic-Sales Correlation", "N/A", "Insufficient data to compute correlation")


if __name__ == "__main__":
    render()
