# logistics_pro/pages/05_Regional_Performance.py 
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import numpy as np


def render():
    """🌍 REGIONAL PERFORMANCE - Geographic Intelligence & Market Strategy"""

    st.title("🌍 Regional Performance")

    # 🌈 Gradient hero header – aligned with Executive Cockpit style
    st.markdown(
        f"""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                padding: 20px; border-radius: 12px; color: white; margin-bottom: 12px;">
        <h3 style="margin: 0; color: white;">🎯 Geographic Intelligence &amp; Market Strategy</h3>
        <p style="margin: 5px 0 0 0; opacity: 0.9; color: white;">
        <strong>📍</strong> Sales Intelligence &gt; Regional Performance |
        <strong>🏢</strong> {st.session_state.get('current_tenant', 'ELORA Holding')} |
        <strong>📅</strong> {datetime.now().strftime('%Y-%m-%d')}
        </p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # 💚 Soft green marquee strip – same pattern as Executive Cockpit
    st.markdown(
        """
    <div style="background: #dcfce7; padding: 10px 16px; border-radius: 10px;
                margin-bottom: 24px; border-left: 4px solid #16a34a;">
        <marquee behavior="scroll" direction="left" scrollamount="4"
                 style="font-size: 0.9rem; font-weight: 500; color: #166534;">
            🌍 Regional Lens: Nairobi driving core volume • Western emerging as high-growth corridor •
            Coastal under strategic watch • Central, Eldoret & Kisumu anchoring national coverage.
        </marquee>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Check if analytics is available
    if "analytics" not in st.session_state:
        st.error("❌ Please visit the main dashboard first to initialize data")
        return

    if "data_gen" not in st.session_state:
        st.error("❌ Data generator not found. Please initialize data from the main dashboard.")
        return

    analytics = st.session_state.analytics
    data_gen = st.session_state.data_gen

    # Get base sales data
    try:
        sales_data = analytics.sales_data.copy()
    except AttributeError:
        st.error("❌ Analytics object does not contain 'sales_data'.")
        return

    # Ensure required columns are present
    required_cols = {"customer_id", "quantity", "unit_price", "date"}
    missing = required_cols - set(sales_data.columns)
    if missing:
        st.error(
            "❌ Sales dataset is missing required columns: "
            + ", ".join(sorted(missing))
        )
        return

    # Normalize types
    sales_data["date"] = pd.to_datetime(sales_data["date"])

    # Ensure we have a SKU dimension (fallback if missing)
    if "sku_id" not in sales_data.columns:
        sales_data["sku_id"] = "SKU-" + sales_data["customer_id"].astype(str)

    # Get enriched sales data
    sales_enriched = get_enriched_sales_data(sales_data, data_gen)

    # Enhanced Tab Structure with Strategic Frameworks
    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        [
            "🏠 Geographic Intelligence",
            "📊 Performance Analytics",
            "📈 Market Share Strategy",
            "🎯 Growth Opportunities",
            "🚀 Strategic Planning",
        ]
    )

    with tab1:
        render_geographic_intelligence(sales_enriched, data_gen)

    with tab2:
        render_performance_analytics(sales_enriched)

    with tab3:
        render_market_share_strategy(sales_enriched)

    with tab4:
        render_growth_opportunities(sales_enriched)

    with tab5:
        render_strategic_planning(sales_enriched)


def get_enriched_sales_data(sales_data: pd.DataFrame, data_gen):
    """Enrich sales data with regional information"""
    sales_enriched = sales_data.copy()

    # Merge with customer data for region/type information if available
    if hasattr(data_gen, "customers") and isinstance(data_gen.customers, pd.DataFrame):
        customer_cols = ["customer_id", "customer_name", "type", "region"]
        available_customer_cols = [
            col for col in customer_cols if col in data_gen.customers.columns
        ]
        if available_customer_cols:
            sales_enriched = sales_enriched.merge(
                data_gen.customers[available_customer_cols],
                on="customer_id",
                how="left",
            )

    # Business metrics
    if {"quantity", "unit_price"}.issubset(sales_enriched.columns):
        sales_enriched["revenue"] = (
            sales_enriched["quantity"] * sales_enriched["unit_price"]
        )
    else:
        sales_enriched["revenue"] = 0.0

    # Handle missing regions
    if "region" not in sales_enriched.columns:
        # Create sample regions based on customer_id
        regions = [
            "Nairobi",
            "Mombasa",
            "Kisumu",
            "Nakuru",
            "Eldoret",
            "Western",
            "Coastal",
            "Central",
        ]
        sales_enriched["region"] = np.random.choice(regions, len(sales_enriched))

    # Fallbacks for category/type if missing
    if "category" not in sales_enriched.columns:
        sales_enriched["category"] = "General"
    if "type" not in sales_enriched.columns:
        sales_enriched["type"] = "Retail"

    return sales_enriched


def render_geographic_intelligence(sales_data: pd.DataFrame, data_gen):
    """Render comprehensive geographic intelligence dashboard"""
    st.header("🏠 Geographic Intelligence Dashboard")

    if sales_data.empty:
        st.info("No sales data available for geographic intelligence.")
        return

    # AI-Powered Regional Insights (static scaffold for now)
    with st.expander("🤖 AI Regional Intelligence Summary", expanded=True):
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("🏆 Top Performing", "Nairobi", "+12% growth")
            st.caption("Market Leader")

        with col2:
            st.metric("🚀 Fastest Growing", "Western", "+18% growth")
            st.caption("Emerging Market")

        with col3:
            st.metric("⚠️ Attention Needed", "Coastal", "-5% decline")
            st.caption("Requires Strategy")

        with col4:
            st.metric("💰 Untapped Potential", "KES 2.1M", "Market Gaps")
            st.caption("Expansion Opportunity")

        st.success(
            """
        **💡 Strategic Insight:**  
        - **Western Region**: High-growth emerging market – prioritize expansion investments  
        - **Nairobi**: Mature market – focus on operational efficiency and premium offerings  
        - **Coastal**: Declining performance – requires targeted marketing and distribution optimization  
        - **Rural Areas**: Significant untapped potential – develop specialized distribution strategies
        """
        )

    # Strategic Filters
    st.subheader("🎯 Strategic Analysis Parameters")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        date_range = st.selectbox(
            "Time Period",
            ["Last 30 Days", "Last 90 Days", "Year to Date", "All Time"],
            key="geo_time",
        )

    with col2:
        if "category" in sales_data.columns:
            categories = sorted(sales_data["category"].dropna().unique().tolist())
        else:
            categories = ["General"]
        product_category = st.selectbox(
            "Product Category", ["All"] + categories, key="geo_category"
        )

    with col3:
        if "type" in sales_data.columns:
            types = sorted(sales_data["type"].dropna().unique().tolist())
        else:
            types = ["Retail"]
        customer_type = st.selectbox(
            "Customer Type", ["All"] + types, key="geo_customer"
        )

    with col4:
        analysis_focus = st.selectbox(
            "Analysis Focus",
            ["Revenue", "Volume", "Customers", "Efficiency"],
            key="geo_focus",
        )

    # Apply filters
    filtered_data = apply_regional_filters(
        sales_data, date_range, product_category, customer_type
    )
    if filtered_data.empty:
        st.warning("No data after applying filters. Adjust filters and try again.")
        return

    # Regional Performance Scorecard
    st.subheader("📊 Regional Performance Scorecard")

    regional_metrics = (
        filtered_data.groupby("region")
        .agg(
            {
                "revenue": "sum",
                "quantity": "sum",
                "customer_id": "nunique",
                "sku_id": "nunique",
            }
        )
        .reset_index()
    )

    if regional_metrics.empty:
        st.info("No regional metrics available for current filters.")
        return

    total_revenue = regional_metrics["revenue"].sum()
    total_customers = regional_metrics["customer_id"].sum()
    total_products = regional_metrics["sku_id"].sum()

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric("Total Revenue", f"KES {total_revenue:,.0f}")
        st.caption("Market Size")

    with col2:
        st.metric("Active Regions", len(regional_metrics))
        st.caption("Geographic Coverage")

    with col3:
        st.metric("Total Customers", int(total_customers))
        st.caption("Customer Base")

    with col4:
        st.metric("Products Sold", int(total_products))
        st.caption("Product Diversity")

    with col5:
        market_density = (
            total_revenue / len(regional_metrics) if len(regional_metrics) > 0 else 0
        )
        st.metric("Avg Market Density", f"KES {market_density:,.0f}")
        st.caption("Revenue per Region")

    # Advanced Geographic Visualization
    st.subheader("🔥 Strategic Geographic Heat Map")

    # Enhanced Kenyan region coordinates with strategic context
    region_strategy_data = {
        "Nairobi": {
            "lat": -1.286389,
            "lon": 36.817223,
            "strategy": "Premium Market",
            "potential": "High",
            "competition": "Intense",
        },
        "Mombasa": {
            "lat": -4.0435,
            "lon": 39.6682,
            "strategy": "Port Logistics Hub",
            "potential": "Medium",
            "competition": "Moderate",
        },
        "Kisumu": {
            "lat": -0.1022,
            "lon": 34.7617,
            "strategy": "Lake Region Gateway",
            "potential": "High",
            "competition": "Low",
        },
        "Nakuru": {
            "lat": -0.3031,
            "lon": 36.0800,
            "strategy": "Agricultural Hub",
            "potential": "Medium",
            "competition": "Moderate",
        },
        "Eldoret": {
            "lat": 0.5143,
            "lon": 35.2698,
            "strategy": "Northern Corridor",
            "potential": "High",
            "competition": "Low",
        },
        "Western": {
            "lat": 0.5963,
            "lon": 34.5730,
            "strategy": "Emerging Market",
            "potential": "Very High",
            "competition": "Low",
        },
        "Coastal": {
            "lat": -3.1733,
            "lon": 40.1170,
            "strategy": "Tourism Recovery",
            "potential": "Medium",
            "competition": "High",
        },
        "Central": {
            "lat": -0.7167,
            "lon": 37.1500,
            "strategy": "Heartland Expansion",
            "potential": "Medium",
            "competition": "Moderate",
        },
    }

    # Prepare enhanced map data
    map_data = []
    for region in regional_metrics["region"].unique():
        region_revenue = regional_metrics.loc[
            regional_metrics["region"] == region, "revenue"
        ].sum()
        if region in region_strategy_data:
            strategy_info = region_strategy_data[region]
            if total_revenue > 0:
                size = (region_revenue / total_revenue) * 200
            else:
                size = 20
            map_data.append(
                {
                    "region": region,
                    "lat": strategy_info["lat"],
                    "lon": strategy_info["lon"],
                    "revenue": region_revenue,
                    "strategy": strategy_info["strategy"],
                    "potential": strategy_info["potential"],
                    "competition": strategy_info["competition"],
                    "size": max(20, min(80, size)),  # Dynamic sizing
                }
            )

    if map_data:
        map_df = pd.DataFrame(map_data)

        fig = px.scatter_mapbox(
            map_df,
            lat="lat",
            lon="lon",
            size="size",
            color="revenue",
            hover_name="region",
            hover_data={
                "revenue": ":.0f",
                "strategy": True,
                "potential": True,
                "competition": True,
                "lat": False,
                "lon": False,
            },
            color_continuous_scale="Viridis",
            size_max=60,
            zoom=5.5,
            center={"lat": -1.286389, "lon": 36.817223},
            title="Strategic Market Overview: Revenue & Growth Potential",
        )

        fig.update_layout(mapbox_style="open-street-map")
        fig.update_layout(margin={"r": 0, "t": 30, "l": 0, "b": 0}, height=500)
        st.plotly_chart(fig, use_container_width=True)

    # Regional Performance Matrix
    st.subheader("📈 Regional Performance Matrix")

    col1, col2 = st.columns(2)

    with col1:
        # Enhanced revenue analysis with strategic context
        top_regions = regional_metrics.nlargest(8, "revenue")
        fig = px.bar(
            top_regions,
            x="region",
            y="revenue",
            title="🏆 Top Performing Regions by Revenue",
            color="revenue",
            color_continuous_scale="Blues",
            text=[f"KES {x:,.0f}" for x in top_regions["revenue"]],
            hover_data=["customer_id"],
        )
        fig.update_layout(xaxis_tickangle=-45, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

        if total_revenue > 0:
            top_3_share = top_regions.head(3)["revenue"].sum() / total_revenue * 100
            st.info(
                f"**Market Concentration:** Top 3 regions generate {top_3_share:.1f}% of total revenue"
            )

    with col2:
        # Customer vs Revenue Efficiency Analysis
        regional_metrics["customers_share"] = (
            regional_metrics["customer_id"] / total_customers * 100
            if total_customers > 0
            else 0
        )
        regional_metrics["revenue_share"] = (
            regional_metrics["revenue"] / total_revenue * 100
            if total_revenue > 0
            else 0
        )

        regional_metrics["efficiency_ratio"] = regional_metrics["revenue_share"] / (
            regional_metrics["customers_share"].replace(0, np.nan)
        )
        regional_metrics["efficiency_ratio"] = regional_metrics[
            "efficiency_ratio"
        ].replace([np.inf, -np.inf], np.nan)

        fig = px.scatter(
            regional_metrics,
            x="customers_share",
            y="revenue_share",
            size="revenue",
            color="efficiency_ratio",
            title="📊 Market Efficiency: Customer vs Revenue Share",
            hover_data=["region", "efficiency_ratio"],
            size_max=60,
            color_continuous_scale="RdYlGn",
        )

        fig.add_trace(
            go.Scatter(
                x=[0, 100],
                y=[0, 100],
                mode="lines",
                line=dict(dash="dash", color="gray"),
                name="Ideal Efficiency Line",
            )
        )

        high_efficiency = regional_metrics[
            regional_metrics["efficiency_ratio"] > 1.2
        ].dropna(subset=["efficiency_ratio"])
        if not high_efficiency.empty:
            for _, row in high_efficiency.iterrows():
                fig.add_annotation(
                    x=row["customers_share"],
                    y=row["revenue_share"],
                    text=row["region"],
                    showarrow=True,
                    arrowhead=2,
                )

        st.plotly_chart(fig, use_container_width=True)


def apply_regional_filters(data: pd.DataFrame, date_range, product_category, customer_type):
    """Apply filters to regional data"""
    filtered_data = data.copy()

    # Ensure date is datetime
    filtered_data["date"] = pd.to_datetime(filtered_data["date"])

    # Date filter
    today = datetime.now().date()
    if date_range == "Last 30 Days":
        cutoff_date = today - timedelta(days=30)
        filtered_data = filtered_data[filtered_data["date"].dt.date >= cutoff_date]
    elif date_range == "Last 90 Days":
        cutoff_date = today - timedelta(days=90)
        filtered_data = filtered_data[filtered_data["date"].dt.date >= cutoff_date]
    elif date_range == "Year to Date":
        cutoff_date = today.replace(month=1, day=1)
        filtered_data = filtered_data[filtered_data["date"].dt.date >= cutoff_date]

    # Product category filter
    if product_category != "All" and "category" in filtered_data.columns:
        filtered_data = filtered_data[filtered_data["category"] == product_category]

    # Customer type filter
    if customer_type != "All" and "type" in filtered_data.columns:
        filtered_data = filtered_data[filtered_data["type"] == customer_type]

    return filtered_data


def render_performance_analytics(sales_data: pd.DataFrame):
    """Render advanced regional performance analytics"""
    st.header("📊 Advanced Performance Analytics")

    if sales_data.empty:
        st.info("No sales data available for performance analytics.")
        return

    # Time-based Regional Intelligence
    st.subheader("📈 Regional Trends & Seasonality Analysis")

    df = sales_data.copy()
    df["month"] = pd.to_datetime(df["date"]).dt.to_period("M")
    monthly_regional = (
        df.groupby(["month", "region"])
        .agg(
            {
                "revenue": "sum",
                "quantity": "sum",
                "customer_id": "nunique",
            }
        )
        .reset_index()
    )
    if monthly_regional.empty:
        st.info("Not enough data to compute monthly regional trends.")
        return

    monthly_regional["month"] = monthly_regional["month"].dt.to_timestamp()

    # Strategic Region Selector
    regions = sorted(df["region"].dropna().unique())
    selected_regions = st.multiselect(
        "🎯 Select Regions for Strategic Analysis",
        regions,
        default=regions[:3] if len(regions) >= 3 else regions,
        key="performance_regions",
    )

    if selected_regions:
        filtered_monthly = monthly_regional[
            monthly_regional["region"].isin(selected_regions)
        ]

        col1, col2 = st.columns(2)

        with col1:
            # Enhanced revenue trends with growth annotations
            fig = px.line(
                filtered_monthly,
                x="month",
                y="revenue",
                color="region",
                title="📈 Monthly Revenue Trends by Region",
                markers=True,
                line_shape="spline",
            )

            # Add trend lines
            for region in selected_regions:
                region_data = filtered_monthly[
                    filtered_monthly["region"] == region
                ].sort_values("month")
                if len(region_data) > 1:
                    z = np.polyfit(
                        range(len(region_data)), region_data["revenue"], 1
                    )
                    trend_line = np.poly1d(z)(range(len(region_data)))

                    fig.add_trace(
                        go.Scatter(
                            x=region_data["month"],
                            y=trend_line,
                            mode="lines",
                            line=dict(dash="dash", width=1),
                            name=f"{region} Trend",
                            showlegend=False,
                        )
                    )

            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Customer growth analysis
            fig = px.line(
                filtered_monthly,
                x="month",
                y="customer_id",
                color="region",
                title="👥 Active Customer Growth by Region",
                markers=True,
                line_shape="spline",
            )
            st.plotly_chart(fig, use_container_width=True)

    # Regional Performance Benchmarking
    st.subheader("🏆 Regional Performance Benchmarks")

    regional_benchmarks = (
        df.groupby("region")
        .agg(
            {
                "revenue": ["sum", "mean", "std"],
                "quantity": "sum",
                "customer_id": "nunique",
                "sku_id": "nunique",
            }
        )
        .reset_index()
    )

    regional_benchmarks.columns = [
        "region",
        "total_revenue",
        "avg_transaction",
        "revenue_std",
        "total_quantity",
        "unique_customers",
        "unique_products",
    ]

    if regional_benchmarks.empty:
        st.info("No benchmark data available.")
        return

    # Avoid division by zero
    regional_benchmarks["unique_customers"] = regional_benchmarks[
        "unique_customers"
    ].replace(0, np.nan)

    regional_benchmarks["revenue_per_customer"] = (
        regional_benchmarks["total_revenue"]
        / regional_benchmarks["unique_customers"]
    )
    regional_benchmarks["quantity_per_customer"] = (
        regional_benchmarks["total_quantity"]
        / regional_benchmarks["unique_customers"]
    )
    regional_benchmarks["product_diversity"] = (
        regional_benchmarks["unique_products"]
        / regional_benchmarks["unique_customers"]
    )

    # Comprehensive performance scoring
    metrics_to_score = [
        "total_revenue",
        "revenue_per_customer",
        "unique_customers",
        "product_diversity",
    ]
    for metric in metrics_to_score:
        max_val = regional_benchmarks[metric].max()
        if pd.isna(max_val) or max_val == 0:
            regional_benchmarks[f"{metric}_score"] = 0
        else:
            regional_benchmarks[f"{metric}_score"] = (
                regional_benchmarks[metric] / max_val * 100
            )

    regional_benchmarks["strategic_performance_score"] = (
        regional_benchmarks["total_revenue_score"] * 0.35
        + regional_benchmarks["revenue_per_customer_score"] * 0.25
        + regional_benchmarks["unique_customers_score"] * 0.25
        + regional_benchmarks["product_diversity_score"] * 0.15
    )

    col1, col2 = st.columns(2)

    with col1:
        top_perf = regional_benchmarks.nlargest(
            8, "strategic_performance_score"
        )
        fig = px.bar(
            top_perf,
            x="region",
            y="strategic_performance_score",
            title="🎯 Regional Strategic Performance Score",
            color="strategic_performance_score",
            color_continuous_scale="RdYlGn",
            text=top_perf["strategic_performance_score"].round(1),
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.scatter(
            regional_benchmarks,
            x="unique_customers",
            y="revenue_per_customer",
            size="total_revenue",
            color="strategic_performance_score",
            title="📊 Strategic Performance Matrix",
            hover_data=["region", "product_diversity"],
            color_continuous_scale="RdYlGn",
            size_max=60,
        )

        median_customers = regional_benchmarks["unique_customers"].median()
        median_revenue_per_customer = regional_benchmarks[
            "revenue_per_customer"
        ].median()

        fig.add_hline(
            y=median_revenue_per_customer, line_dash="dash", line_color="red"
        )
        fig.add_vline(
            x=median_customers, line_dash="dash", line_color="red"
        )

        fig.add_annotation(
            x=median_customers * 1.8,
            y=median_revenue_per_customer * 1.8,
            text="Stars",
            showarrow=False,
            font=dict(size=12, color="green"),
        )
        fig.add_annotation(
            x=median_customers * 0.5,
            y=median_revenue_per_customer * 1.8,
            text="Question Marks",
            showarrow=False,
            font=dict(size=12, color="orange"),
        )
        fig.add_annotation(
            x=median_customers * 1.8,
            y=median_revenue_per_customer * 0.5,
            text="Cash Cows",
            showarrow=False,
            font=dict(size=12, color="blue"),
        )
        fig.add_annotation(
            x=median_customers * 0.5,
            y=median_revenue_per_customer * 0.5,
            text="Dogs",
            showarrow=False,
            font=dict(size=12, color="red"),
        )

        st.plotly_chart(fig, use_container_width=True)

    # Regional Efficiency Intelligence
    st.subheader("⚡ Regional Efficiency & Productivity")

    efficiency_metrics = regional_benchmarks[
        [
            "region",
            "total_revenue",
            "unique_customers",
            "revenue_per_customer",
            "quantity_per_customer",
            "strategic_performance_score",
            "product_diversity",
        ]
    ].copy()

    efficiency_metrics["quantity_per_customer"] = efficiency_metrics[
        "quantity_per_customer"
    ].replace(0, np.nan)
    efficiency_metrics["efficiency_ratio"] = (
        efficiency_metrics["revenue_per_customer"]
        / efficiency_metrics["quantity_per_customer"]
    )

    max_customers = efficiency_metrics["unique_customers"].max()
    if pd.isna(max_customers) or max_customers == 0:
        efficiency_metrics["market_saturation"] = 0
    else:
        efficiency_metrics["market_saturation"] = (
            efficiency_metrics["unique_customers"] / max_customers * 100
        )

    col1, col2, col3, col4 = st.columns(4)

    if not efficiency_metrics.empty:
        with col1:
            best_row = efficiency_metrics.loc[
                efficiency_metrics["strategic_performance_score"].idxmax()
            ]
            st.metric("🏆 Best Performing", best_row["region"])
            st.caption("Highest strategic score")

        with col2:
            eff_valid = efficiency_metrics.dropna(subset=["efficiency_ratio"])
            if not eff_valid.empty:
                most_row = eff_valid.loc[
                    eff_valid["efficiency_ratio"].idxmax()
                ]
                st.metric("⚡ Most Efficient", most_row["region"])
                st.caption("Best revenue per unit")
            else:
                st.metric("⚡ Most Efficient", "N/A")

        with col3:
            gp_candidates = efficiency_metrics[
                (efficiency_metrics["market_saturation"] < 60)
                & (efficiency_metrics["strategic_performance_score"] > 50)
            ]
            if not gp_candidates.empty:
                growth_row = gp_candidates.loc[
                    gp_candidates["strategic_performance_score"].idxmax()
                ]
                st.metric("🚀 Growth Potential", growth_row["region"])
                st.caption("High score, low saturation")
            else:
                st.metric("🚀 Growth Potential", "N/A")

        with col4:
            worst_row = efficiency_metrics.loc[
                efficiency_metrics["strategic_performance_score"].idxmin()
            ]
            st.metric("⚠️ Needs Attention", worst_row["region"])
            st.caption("Lowest strategic score")

    # Detailed table
    st.subheader("📋 Comprehensive Regional Performance Details")

    display_metrics = efficiency_metrics.rename(
        columns={
            "region": "Region",
            "total_revenue": "Total Revenue",
            "unique_customers": "Customers",
            "revenue_per_customer": "Revenue/Customer",
            "quantity_per_customer": "Quantity/Customer",
            "strategic_performance_score": "Performance Score",
            "efficiency_ratio": "Efficiency Ratio",
            "market_saturation": "Market Saturation %",
            "product_diversity": "Product Diversity",
        }
    ).round(2)

    st.dataframe(display_metrics, use_container_width=True)


def render_market_share_strategy(sales_data: pd.DataFrame):
    """Render market share and competitive strategy analysis"""
    st.header("📈 Market Share Strategy & Competitive Intelligence")

    st.info(
        """
    **💡 Strategic Context:** Analyzing regional market penetration, competitive positioning, 
    and identifying strategic opportunities for market share growth.
    """
    )

    if sales_data.empty:
        st.info("No sales data available for market share analysis.")
        return

    # Market Share Intelligence
    st.subheader("🎯 Market Share Analysis")

    regional_share = (
        sales_data.groupby("region")
        .agg({"revenue": "sum", "customer_id": "nunique"})
        .reset_index()
    )

    if regional_share.empty:
        st.info("No regional data available for market share.")
        return

    total_market_revenue = regional_share["revenue"].sum()
    if total_market_revenue > 0:
        regional_share["market_share"] = (
            regional_share["revenue"] / total_market_revenue * 100
        )
    else:
        regional_share["market_share"] = 0

    # Static competitive intelligence scaffold
    competitors_strategy = {
        "Nairobi": {
            "total_market": total_market_revenue * 2.5,
            "our_share": 35,
            "main_competitors": ["Competitor A (40%)", "Competitor B (25%)"],
            "competitive_intensity": "High",
        },
        "Mombasa": {
            "total_market": total_market_revenue * 1.8,
            "our_share": 28,
            "main_competitors": ["Competitor C (45%)", "Competitor D (27%)"],
            "competitive_intensity": "Medium-High",
        },
        "Kisumu": {
            "total_market": total_market_revenue * 1.2,
            "our_share": 42,
            "main_competitors": ["Competitor E (35%)", "Competitor F (23%)"],
            "competitive_intensity": "Medium",
        },
        "Nakuru": {
            "total_market": total_market_revenue * 1.5,
            "our_share": 38,
            "main_competitors": ["Competitor G (40%)", "Competitor H (22%)"],
            "competitive_intensity": "Medium",
        },
        "Eldoret": {
            "total_market": total_market_revenue * 1.1,
            "our_share": 45,
            "main_competitors": ["Competitor I (30%)", "Competitor J (25%)"],
            "competitive_intensity": "Low-Medium",
        },
        "Western": {
            "total_market": total_market_revenue * 0.9,
            "our_share": 52,
            "main_competitors": ["Competitor K (25%)", "Competitor L (23%)"],
            "competitive_intensity": "Low",
        },
        "Coastal": {
            "total_market": total_market_revenue * 1.3,
            "our_share": 22,
            "main_competitors": ["Competitor M (50%)", "Competitor N (28%)"],
            "competitive_intensity": "High",
        },
        "Central": {
            "total_market": total_market_revenue * 1.4,
            "our_share": 31,
            "main_competitors": ["Competitor O (45%)", "Competitor P (24%)"],
            "competitive_intensity": "Medium-High",
        },
    }

    col1, col2 = st.columns(2)

    with col1:
        fig = px.pie(
            regional_share,
            values="market_share",
            names="region",
            title="📊 Our Strategic Market Share by Region",
            hole=0.4,
            color_discrete_sequence=px.colors.qualitative.Set3,
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        competitor_data = []
        for region, data in competitors_strategy.items():
            competitor_data.append(
                {
                    "region": region,
                    "our_share": data["our_share"],
                    "competitor_share": 100 - data["our_share"],
                    "total_market": data["total_market"],
                    "competitive_intensity": data["competitive_intensity"],
                }
            )

        competitor_df = pd.DataFrame(competitor_data)

        fig = px.bar(
            competitor_df,
            x="region",
            y=["our_share", "competitor_share"],
            title="🕹 Competitive Market Share Analysis",
            barmode="stack",
            color_discrete_map={
                "our_share": "#1f77b4",
                "competitor_share": "#ff7f0e",
            },
            hover_data=["competitive_intensity"],
        )
        fig.update_layout(xaxis_tickangle=-45, yaxis_title="Market Share %")
        st.plotly_chart(fig, use_container_width=True)

    # Strategic Market Penetration Analysis
    st.subheader("🎯 Market Penetration Strategy Framework")

    penetration_analysis = []
    for region, data in competitors_strategy.items():
        our_revenue = regional_share.loc[
            regional_share["region"] == region, "revenue"
        ].sum()
        total_market = data["total_market"]
        penetration_potential = max(total_market - our_revenue, 0)

        if our_revenue > 0:
            growth_opportunity = penetration_potential / our_revenue * 100
        else:
            growth_opportunity = 100.0

        penetration_analysis.append(
            {
                "region": region,
                "our_revenue": our_revenue,
                "total_market": total_market,
                "current_share": data["our_share"],
                "penetration_potential": penetration_potential,
                "growth_opportunity": growth_opportunity,
                "competitive_intensity": data["competitive_intensity"],
                "main_competitors": ", ".join(data["main_competitors"]),
            }
        )

    penetration_df = pd.DataFrame(penetration_analysis)

    col1, col2 = st.columns(2)

    with col1:
        top_p = penetration_df.nlargest(6, "penetration_potential")
        fig = px.bar(
            top_p,
            x="region",
            y="penetration_potential",
            title="🚀 Top Market Penetration Opportunities",
            color="growth_opportunity",
            color_continuous_scale="Viridis",
            hover_data=["competitive_intensity"],
            text=[
                f"KES {x:,.0f}" for x in top_p["penetration_potential"]
            ],
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.scatter(
            penetration_df,
            x="current_share",
            y="growth_opportunity",
            size="total_market",
            color="competitive_intensity",
            title="📈 Strategic Growth Matrix",
            hover_data=["region", "penetration_potential", "main_competitors"],
            size_max=60,
        )

        median_share = penetration_df["current_share"].median()
        median_growth = penetration_df["growth_opportunity"].median()

        fig.add_hline(y=median_growth, line_dash="dash", line_color="red")
        fig.add_vline(x=median_share, line_dash="dash", line_color="red")

        st.plotly_chart(fig, use_container_width=True)

    # Enhanced Regional Strategy Recommendations
    st.subheader("💡 Regional Strategy Playbook")

    strategy_framework = {
        "High Penetration, High Growth": {
            "strategy": "🚀 **Aggressive Expansion & Market Leadership**",
            "focus": "Dominate market through increased investment and competitive positioning",
            "key_actions": [
                "Increase marketing budget by 40–60%",
                "Expand distribution network aggressively",
                "Launch premium product lines",
                "Implement competitive pricing strategies",
            ],
            "investment_level": "High",
            "expected_roi": "25–40%",
        },
        "High Penetration, Low Growth": {
            "strategy": "🛡️ **Defensive Optimization & Profit Maximization**",
            "focus": "Protect market share while improving operational efficiency",
            "key_actions": [
                "Focus on customer retention programs",
                "Optimize operational costs",
                "Enhance product quality and service",
                "Develop loyalty and referral programs",
            ],
            "investment_level": "Medium",
            "expected_roi": "15–25%",
        },
        "Low Penetration, High Growth": {
            "strategy": "🎯 **Targeted Growth & Strategic Investment**",
            "focus": "Selective investments in high-potential market segments",
            "key_actions": [
                "Identify and target underserved segments",
                "Develop specialized product offerings",
                "Build strategic partnerships",
                "Implement phased market entry",
            ],
            "investment_level": "Medium-High",
            "expected_roi": "20–35%",
        },
        "Low Penetration, Low Growth": {
            "strategy": "📊 **Strategic Re-evaluation & Niche Focus**",
            "focus": "Either find profitable niches or consider market exit",
            "key_actions": [
                "Conduct deep market analysis",
                "Identify profitable niche opportunities",
                "Test limited product offerings",
                "Consider partnership or acquisition strategies",
            ],
            "investment_level": "Low",
            "expected_roi": "10–20% (variable)",
        },
    }

    for region_data in penetration_analysis:
        with st.expander(
            f"📋 {region_data['region']} - {region_data['current_share']}% Market Share | Competition: {region_data['competitive_intensity']}"
        ):
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Our Revenue", f"KES {region_data['our_revenue']:,.0f}")

            with col2:
                st.metric(
                    "Total Market",
                    f"KES {region_data['total_market']:,.0f}",
                )

            with col3:
                st.metric(
                    "Growth Potential",
                    f"+{region_data['growth_opportunity']:.1f}%",
                )

            with col4:
                st.metric(
                    "Competitive Intensity",
                    region_data["competitive_intensity"],
                )

            # Strategy classification
            if region_data["current_share"] > 40 and region_data[
                "growth_opportunity"
            ] > 50:
                strategy_key = "High Penetration, High Growth"
            elif region_data["current_share"] > 40:
                strategy_key = "High Penetration, Low Growth"
            elif region_data["growth_opportunity"] > 50:
                strategy_key = "Low Penetration, High Growth"
            else:
                strategy_key = "Low Penetration, Low Growth"

            strategy = strategy_framework[strategy_key]

            st.success(f"**Recommended Strategy:** {strategy['strategy']}")
            st.write(f"**Strategic Focus:** {strategy['focus']}")
            st.write(
                f"**Investment Level:** {strategy['investment_level']} | **Expected ROI:** {strategy['expected_roi']}"
            )

            st.write("**Key Strategic Actions:**")
            for action in strategy["key_actions"]:
                st.write(f"✅ {action}")

            st.write("**Competitive Landscape:**")
            st.write(f"Main Competitors: {region_data['main_competitors']}")

            if st.button(
                f"🚀 Execute {region_data['region']} Strategy",
                key=f"strategy_{region_data['region']}",
            ):
                st.success(
                    f"Strategic plan for {region_data['region']} initiated!"
                )


def render_growth_opportunities(sales_data: pd.DataFrame):
    """Render growth opportunity analysis with strategic frameworks"""
    st.header("🎯 Strategic Growth Opportunities")

    if sales_data.empty:
        st.info("No sales data available for growth analysis.")
        return

    # Enhanced Growth Analysis
    st.subheader("📈 Advanced Growth Potential Analysis")

    df = sales_data.copy()
    df["month"] = pd.to_datetime(df["date"]).dt.to_period("M")
    monthly_growth = (
        df.groupby(["month", "region"])
        .agg({"revenue": "sum", "customer_id": "nunique"})
        .reset_index()
    )

    if monthly_growth.empty:
        st.info("Not enough data to compute growth metrics.")
        return

    monthly_growth["month"] = monthly_growth["month"].dt.to_timestamp()
    monthly_growth = monthly_growth.sort_values(["region", "month"])

    monthly_growth["revenue_growth"] = (
        monthly_growth.groupby("region")["revenue"].pct_change() * 100
    )
    monthly_growth["customer_growth"] = (
        monthly_growth.groupby("region")["customer_id"].pct_change() * 100
    )

    # Focus on recent performance (last 3 months)
    recent_months = monthly_growth["month"].nlargest(3).unique()
    recent_growth = monthly_growth[monthly_growth["month"].isin(recent_months)]

    if recent_growth.empty:
        st.info("No recent growth data available.")
        return

    avg_growth = (
        recent_growth.groupby("region")
        .agg(
            {
                "revenue_growth": "mean",
                "customer_growth": "mean",
                "revenue": "mean",
                "customer_id": "mean",
            }
        )
        .reset_index()
    )

    st.subheader("🚀 Growth Opportunity Matrix (BCG Framework)")

    avg_growth["growth_potential"] = (
        avg_growth["revenue_growth"].fillna(0) * 0.6
        + avg_growth["customer_growth"].fillna(0) * 0.4
    )

    # ✅ Ensure bubble sizes are non-negative
    avg_growth["revenue_growth_size"] = (
        avg_growth["revenue_growth"]
        .fillna(0)
        .clip(lower=0)
        + 1
    )

    fig = px.scatter(
        avg_growth,
        x="revenue",
        y="growth_potential",
        size="revenue_growth_size",
        color="region",
        title="🎯 BCG Growth-Share Matrix: Strategic Opportunity Analysis",
        hover_data=["customer_growth", "customer_id", "revenue_growth"],
        size_max=80,
        color_discrete_sequence=px.colors.qualitative.Set3,
    )

    median_revenue = avg_growth["revenue"].median()
    median_growth = avg_growth["growth_potential"].median()

    fig.add_hline(y=median_growth, line_dash="dash", line_color="red", line_width=2)
    fig.add_vline(x=median_revenue, line_dash="dash", line_color="red", line_width=2)

    fig.add_annotation(
        x=median_revenue * 1.8,
        y=median_growth * 1.8,
        text="🌟 STARS\n(High Growth, High Share)",
        showarrow=False,
        font=dict(size=12, color="green"),
        bgcolor="rgba(0,255,0,0.1)",
    )

    fig.add_annotation(
        x=median_revenue * 0.5,
        y=median_growth * 1.8,
        text="❓ QUESTION MARKS\n(High Growth, Low Share)",
        showarrow=False,
        font=dict(size=12, color="orange"),
        bgcolor="rgba(255,165,0,0.1)",
    )

    fig.add_annotation(
        x=median_revenue * 1.8,
        y=median_growth * 0.5,
        text="🐄 CASH COWS\n(Low Growth, High Share)",
        showarrow=False,
        font=dict(size=12, color="blue"),
        bgcolor="rgba(0,0,255,0.1)",
    )

    fig.add_annotation(
        x=median_revenue * 0.5,
        y=median_growth * 0.5,
        text="🐕 DOGS\n(Low Growth, Low Share)",
        showarrow=False,
        font=dict(size=12, color="red"),
        bgcolor="rgba(255,0,0,0.1)",
    )

    st.plotly_chart(fig, use_container_width=True)

    # Strategic Investment Priority Analysis
    st.subheader("💰 Strategic Investment Priority Framework")

    investment_metrics = avg_growth.copy()

    # Scores with safe denominators
    for col, score_col in [
        ("revenue", "market_size_score"),
        ("growth_potential", "growth_potential_score"),
        ("customer_growth", "customer_momentum_score"),
    ]:
        max_val = investment_metrics[col].max()
        if pd.isna(max_val) or max_val == 0:
            investment_metrics[score_col] = 0
        else:
            investment_metrics[score_col] = (
                investment_metrics[col].fillna(0) / max_val * 100
            )

    investment_metrics["strategic_investment_score"] = (
        investment_metrics["market_size_score"] * 0.25
        + investment_metrics["growth_potential_score"] * 0.35
        + investment_metrics["customer_momentum_score"] * 0.25
        + (100 - investment_metrics["market_size_score"]) * 0.15
    )

    investment_metrics["investment_priority"] = pd.cut(
        investment_metrics["strategic_investment_score"],
        bins=[0, 30, 60, 80, 100],
        labels=["Low Priority", "Medium Priority", "High Priority", "Critical Priority"],
        include_lowest=True,
    )

    col1, col2 = st.columns(2)

    with col1:
        im_sorted = investment_metrics.sort_values(
            "strategic_investment_score", ascending=False
        )
        fig = px.bar(
            im_sorted,
            x="region",
            y="strategic_investment_score",
            color="investment_priority",
            title="🎯 Strategic Investment Priority Score",
            color_discrete_map={
                "Critical Priority": "#FF0000",
                "High Priority": "#FFA500",
                "Medium Priority": "#1E90FF",
                "Low Priority": "#32CD32",
            },
            text=im_sorted["strategic_investment_score"].round(1),
        )
        fig.update_layout(xaxis_tickangle=-45, showlegend=True)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        priority_summary = investment_metrics["investment_priority"].value_counts()
        if not priority_summary.empty:
            fig = px.pie(
                values=priority_summary.values,
                names=priority_summary.index,
                title="📊 Investment Priority Distribution",
                color=priority_summary.index,
                color_discrete_map={
                    "Critical Priority": "#FF0000",
                    "High Priority": "#FFA500",
                    "Medium Priority": "#1E90FF",
                    "Low Priority": "#32CD32",
                },
            )
            st.plotly_chart(fig, use_container_width=True)

    # Actionable Strategic Growth Initiatives (static scaffolding)
    st.subheader("💡 Strategic Growth Initiatives & Execution Plan")

    strategic_initiatives = [
        {
            "initiative": "Western Region Market Dominance Program",
            "region": "Western",
            "strategic_focus": "Market Leadership",
            "impact": "Very High",
            "investment": "KES 1.5M",
            "timeline": "6 months",
            "expected_growth": "+25–35%",
            "key_metrics": [
                "Market Share",
                "Revenue Growth",
                "Customer Acquisition",
            ],
            "success_probability": "85%",
        },
        {
            "initiative": "Coastal Market Penetration & Recovery",
            "region": "Coastal",
            "strategic_focus": "Market Recovery",
            "impact": "High",
            "investment": "KES 800K",
            "timeline": "4 months",
            "expected_growth": "+18–25%",
            "key_metrics": [
                "Customer Retention",
                "Revenue Recovery",
                "Competitive Positioning",
            ],
            "success_probability": "70%",
        },
        {
            "initiative": "Nairobi Premium Product & Service Expansion",
            "region": "Nairobi",
            "strategic_focus": "Premium Market Development",
            "impact": "High",
            "investment": "KES 2.0M",
            "timeline": "8 months",
            "expected_growth": "+15–20%",
            "key_metrics": [
                "Average Order Value",
                "Premium Customer Acquisition",
                "Service Revenue",
            ],
            "success_probability": "80%",
        },
        {
            "initiative": "Central Region Distribution & Efficiency Optimization",
            "region": "Central",
            "strategic_focus": "Operational Excellence",
            "impact": "Medium-High",
            "investment": "KES 500K",
            "timeline": "3 months",
            "expected_growth": "+12–18%",
            "key_metrics": [
                "Cost Efficiency",
                "Delivery Speed",
                "Customer Satisfaction",
            ],
            "success_probability": "90%",
        },
        {
            "initiative": "Emerging Markets (Eldoret & Kisumu) Expansion",
            "region": "Eldoret, Kisumu",
            "strategic_focus": "Strategic Expansion",
            "impact": "Medium",
            "investment": "KES 1.2M",
            "timeline": "5 months",
            "expected_growth": "+20–30%",
            "key_metrics": [
                "Market Penetration",
                "New Customer Acquisition",
                "Brand Awareness",
            ],
            "success_probability": "75%",
        },
    ]

    for initiative in strategic_initiatives:
        with st.expander(
            f"🎯 {initiative['initiative']} | Impact: {initiative['impact']} | Success: {initiative['success_probability']}"
        ):
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Strategic Focus", initiative["strategic_focus"])

            with col2:
                st.metric("Investment", initiative["investment"])

            with col3:
                st.metric("Timeline", initiative["timeline"])

            with col4:
                st.metric("Expected Growth", initiative["expected_growth"])

            st.write("**Key Success Metrics:**")
            for metric in initiative["key_metrics"]:
                st.write(f"📊 {metric}")

            col1, col2 = st.columns(2)
            with col1:
                if st.button(
                    "📋 View Detailed Plan",
                    key=f"plan_{initiative['initiative']}",
                ):
                    st.success(
                        f"Detailed strategic plan loaded for {initiative['initiative']}"
                    )

            with col2:
                if st.button(
                    "🚀 Launch Initiative",
                    key=f"launch_{initiative['initiative']}",
                ):
                    st.success(
                        f"Strategic initiative '{initiative['initiative']}' launched successfully!"
                    )


def render_strategic_planning(sales_data: pd.DataFrame):
    """Render strategic planning and execution framework"""
    st.header("🚀 Strategic Planning & Execution Framework")

    st.markdown(
        """
    <div style="background: #f0f8ff; padding: 20px; border-radius: 8px; border-left: 4px solid #007bff; margin-bottom: 20px;">
        <strong>🎯 Comprehensive Strategic Planning</strong><br>
        Develop and execute data-driven regional strategies with clear objectives, measurable KPIs, and actionable execution plans.
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Strategic Planning Framework
    st.subheader("📋 Strategic Planning Canvas")

    col1, col2 = st.columns(2)

    with col1:
        st.info(
            """
        **🎯 Strategic Objectives:**
        - Achieve 25% market share in Western region within 12 months  
        - Increase overall regional revenue by 18% in next fiscal year  
        - Expand geographic coverage to 2 new regions  
        - Improve regional operational efficiency by 15%
        """
        )

    with col2:
        st.info(
            """
        **📊 Key Performance Indicators:**
        - Regional Revenue Growth Rate  
        - Market Share by Region  
        - Customer Acquisition Cost  
        - Operational Efficiency Ratio  
        - Customer Satisfaction Scores
        """
        )

    # Strategic Initiative Portfolio
    st.subheader("💼 Strategic Initiative Portfolio")

    initiative_portfolio = [
        {
            "type": "Market Expansion",
            "initiatives": 4,
            "total_budget": "KES 3.2M",
            "expected_roi": "28%",
            "status": "Planning",
        },
        {
            "type": "Operational Excellence",
            "initiatives": 3,
            "total_budget": "KES 1.5M",
            "expected_roi": "22%",
            "status": "Execution",
        },
        {
            "type": "Customer Experience",
            "initiatives": 2,
            "total_budget": "KES 800K",
            "expected_roi": "35%",
            "status": "Planning",
        },
        {
            "type": "Technology Enablement",
            "initiatives": 2,
            "total_budget": "KES 1.2M",
            "expected_roi": "30%",
            "status": "Evaluation",
        },
    ]

    portfolio_df = pd.DataFrame(initiative_portfolio)

    col1, col2 = st.columns(2)

    with col1:
        fig = px.bar(
            portfolio_df,
            x="type",
            y="initiatives",
            color="expected_roi",
            title="📈 Strategic Initiative Portfolio",
            text="initiatives",
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.pie(
            portfolio_df,
            values="total_budget",
            names="type",
            title="💰 Budget Allocation by Initiative Type",
            hole=0.4,
        )
        st.plotly_chart(fig, use_container_width=True)

    # Strategic Execution Dashboard
    st.subheader("📊 Strategic Execution Dashboard")

    execution_metrics = {
        "Initiatives Planned": 11,
        "Initiatives in Progress": 4,
        "Initiatives Completed": 2,
        "Budget Utilized": "42%",
        "Overall Progress": "58%",
        "ROI Tracking": "On Target",
    }

    col1, col2, col3, col4, col5, col6 = st.columns(6)
    metrics_cols = [col1, col2, col3, col4, col5, col6]

    for (metric, value), col in zip(execution_metrics.items(), metrics_cols):
        with col:
            st.metric(metric, value)

    # Quick Action Center
    st.subheader("⚡ Quick Action Center")

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("📋 Generate Strategic Report"):
            st.success("Comprehensive strategic report generated!")

    with col2:
        if st.button("🎯 Update Performance Metrics"):
            st.success("Performance metrics updated successfully!")

    with col3:
        if st.button("🚀 Launch New Initiative"):
            st.success("New initiative launch process started!")


if __name__ == "__main__":
    render()
