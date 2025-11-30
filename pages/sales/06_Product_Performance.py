# logistics_pro/pages/06_Product_Performance.py
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import numpy as np


def _ensure_numeric(df: pd.DataFrame, cols) -> pd.DataFrame:
    """Safely coerce selected columns to numeric, filling non-numeric with 0."""
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
    return df


def safe_nlargest(df: pd.DataFrame, n: int, column: str) -> pd.DataFrame:
    """
    Safe wrapper around DataFrame.nlargest that guarantees numeric dtype
    for the target column before ranking.
    """
    if df is None or df.empty:
        return df
    df = df.copy()
    if column not in df.columns:
        df[column] = 0
    df[column] = pd.to_numeric(df[column], errors="coerce").fillna(0)
    return df.nlargest(n, column)


def render():
    """📦 PRODUCT PERFORMANCE - Portfolio Intelligence & Strategy"""

    st.title("📦 Product Performance")

    # 🌈 Gradient hero header – aligned with 01_Dashboard pattern
    st.markdown(
        f"""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                padding: 20px; border-radius: 12px; color: white; margin-bottom: 12px;">
        <h3 style="margin: 0; color: white;">🎯 Product Portfolio Intelligence & Strategy</h3>
        <p style="margin: 5px 0 0 0; opacity: 0.9; color: white;">
        <strong>📍</strong> Sales Intelligence &gt; Product Performance |
        <strong>🏢</strong> {st.session_state.get('current_tenant', 'ELORA Holding')} |
        <strong>📅</strong> {datetime.now().strftime('%Y-%m-%d')}
        </p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # 💚 Soft green marquee strip – same UX pattern as Executive Cockpit
    st.markdown(
        """
    <div style="background: #dcfce7; padding: 10px 16px; border-radius: 10px;
                margin-bottom: 24px; border-left: 4px solid #16a34a;">
        <marquee behavior="scroll" direction="left" scrollamount="4"
                 style="font-size: 0.9rem; font-weight: 500; color: #166534;">
            🌟 <strong>Product Intelligence:</strong> Portfolio insights for revenue, margin, and lifecycle • 
            🏆 <strong>Stars:</strong> Top SKUs driving disproportionate value • 
            🎯 <strong>Category Focus:</strong> Strategic bets across Beverages, Snacks, Dairy & Core FMCG • 
            💡 <strong>Innovation:</strong> Health & wellness, convenience, and premiumization opportunities
        </marquee>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Check if analytics is available (aligned message with 01_Dashboard)
    if "analytics" not in st.session_state:
        st.error("❌ Please initialize the application first")
        return

    if "data_gen" not in st.session_state:
        st.error("❌ Data generator not initialized. Please load SKU master data first.")
        return

    analytics = st.session_state.analytics
    data_gen = st.session_state.data_gen

    # Get enriched sales data
    sales_enriched = get_enriched_sales_data(analytics.sales_data, data_gen)

    # Enhanced Tab Structure with Strategic Frameworks
    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        [
            "🏆 Portfolio Intelligence",
            "📊 Category Strategy",
            "🔄 Lifecycle Analytics",
            "🎯 Optimization Engine",
            "🚀 Innovation Pipeline",
        ]
    )

    with tab1:
        render_portfolio_intelligence(sales_enriched, data_gen)

    with tab2:
        render_category_strategy(sales_enriched)

    with tab3:
        render_lifecycle_analytics(sales_enriched)

    with tab4:
        render_optimization_engine(sales_enriched, data_gen)

    with tab5:
        render_innovation_pipeline(sales_enriched, data_gen)


def get_enriched_sales_data(sales_data, data_gen):
    """Enrich sales data with product information"""
    sales_enriched = sales_data.copy()

    # Merge with SKU data for product information
    if hasattr(data_gen, "skus"):
        sku_cols = ["sku_id", "sku_name", "category", "unit_cost", "selling_price"]
        available_sku_cols = [col for col in sku_cols if col in data_gen.skus.columns]
        if available_sku_cols:
            sales_enriched = sales_enriched.merge(
                data_gen.skus[available_sku_cols], on="sku_id", how="left"
            )

    # Ensure quantity-like fields are numeric BEFORE calculations
    if "quantity" in sales_enriched.columns:
        sales_enriched["quantity"] = pd.to_numeric(
            sales_enriched["quantity"], errors="coerce"
        ).fillna(0)

    # Handle unit_price: from sales or from SKU selling_price as fallback
    if "unit_price" not in sales_enriched.columns:
        if "selling_price" in sales_enriched.columns:
            sales_enriched["unit_price"] = pd.to_numeric(
                sales_enriched["selling_price"], errors="coerce"
            ).fillna(100.0)
        else:
            sales_enriched["unit_price"] = 100.0
    else:
        sales_enriched["unit_price"] = pd.to_numeric(
            sales_enriched["unit_price"], errors="coerce"
        ).fillna(100.0)

    # Handle unit_cost: from SKU unit_cost or margin assumption
    if "unit_cost" not in sales_enriched.columns:
        sales_enriched["unit_cost"] = sales_enriched["unit_price"] * 0.7
    else:
        sales_enriched["unit_cost"] = pd.to_numeric(
            sales_enriched["unit_cost"], errors="coerce"
        ).fillna(sales_enriched["unit_price"] * 0.7)

    # Now compute business metrics with guaranteed numeric inputs
    sales_enriched["revenue"] = (
        pd.to_numeric(sales_enriched["quantity"], errors="coerce").fillna(0)
        * pd.to_numeric(sales_enriched["unit_price"], errors="coerce").fillna(0)
    )
    sales_enriched["cost"] = (
        pd.to_numeric(sales_enriched["quantity"], errors="coerce").fillna(0)
        * pd.to_numeric(sales_enriched["unit_cost"], errors="coerce").fillna(0)
    )
    sales_enriched["margin"] = sales_enriched["revenue"] - sales_enriched["cost"]
    sales_enriched["margin_percent"] = (
        sales_enriched["margin"] / sales_enriched["revenue"]
    ).replace([np.inf, -np.inf], 0) * 100

    # Handle missing categories
    if "category" not in sales_enriched.columns:
        categories = [
            "Beverages",
            "Snacks",
            "Dairy",
            "Bakery",
            "Household",
            "Personal Care",
        ]
        sales_enriched["category"] = np.random.choice(categories, len(sales_enriched))

    # Handle missing product names
    if "sku_name" not in sales_enriched.columns:
        sales_enriched["sku_name"] = "Product " + sales_enriched["sku_id"].astype(str)

    return sales_enriched


def render_portfolio_intelligence(sales_data, data_gen):
    """Render comprehensive product portfolio intelligence"""
    st.header("🏆 Product Portfolio Intelligence Dashboard")

    # AI-Powered Product Insights
    with st.expander("🧠 AI Portfolio Intelligence Summary", expanded=True):
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("🚀 Fastest Growing", "Premium Water", "+42% growth")
            st.caption("Emerging Star")

        with col2:
            st.metric("💰 Highest Margin", "Specialty Coffee", "38% margin")
            st.caption("Profit Leader")

        with col3:
            st.metric("📈 Consistent Performer", "Bread & Bakery", "98% reliability")
            st.caption("Portfolio Anchor")

        with col4:
            st.metric("⚠️ Needs Attention", "Canned Goods", "-8% decline")
            st.caption("Strategic Review")

        st.success(
            """
        **💡 Strategic Insight:** 
        - **Premium Products**: High-growth, high-margin segment - prioritize expansion  
        - **Core Portfolio**: Stable performers - focus on operational efficiency  
        - **Underperformers**: Selective optimization or potential rationalization  
        - **Innovation Gap**: Opportunity in health & wellness categories
        """
        )

    # Strategic Analysis Parameters
    st.subheader("🎯 Strategic Analysis Parameters")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        date_range = st.selectbox(
            "Time Period",
            ["Last 30 Days", "Last 90 Days", "Year to Date", "All Time"],
            key="portfolio_time",
        )
    with col2:
        category_filter = st.selectbox(
            "Category Focus",
            ["All"] + sorted(sales_data["category"].unique().tolist()),
            key="portfolio_category",
        )
    with col3:
        performance_view = st.selectbox(
            "Performance Metric",
            ["Revenue", "Quantity", "Margin", "Growth", "Efficiency"],
            key="portfolio_view",
        )
    with col4:
        analysis_depth = st.selectbox(
            "Analysis Depth",
            ["Overview", "Detailed", "Strategic"],
            key="portfolio_depth",
        )

    # Apply filters
    filtered_data = apply_product_filters(sales_data, date_range, category_filter)

    # Enhanced Product Portfolio KPIs
    st.subheader("📊 Portfolio Performance Scorecard")

    # Calculate comprehensive product metrics
    product_metrics = (
        filtered_data.groupby(["sku_id", "sku_name", "category"])
        .agg(
            {
                "revenue": "sum",
                "quantity": "sum",
                "margin": "sum",
                "margin_percent": "mean",
                "customer_id": "nunique",
                "date": "count",
            }
        )
        .reset_index()
    )

    # ENSURE numeric dtypes before any nlargest / arithmetic
    product_metrics = _ensure_numeric(
        product_metrics,
        [
            "revenue",
            "quantity",
            "margin",
            "margin_percent",
            "customer_id",
            "date",
        ],
    )

    product_metrics["margin_percent_overall"] = (
        product_metrics["margin"] / product_metrics["revenue"]
    ).replace([np.inf, -np.inf], 0) * 100
    product_metrics["avg_order_quantity"] = (
        product_metrics["quantity"] / product_metrics["date"]
    ).replace([np.inf, -np.inf], 0)
    product_metrics["revenue_per_customer"] = (
        product_metrics["revenue"] / product_metrics["customer_id"]
    ).replace([np.inf, -np.inf], 0)

    product_metrics = _ensure_numeric(
        product_metrics,
        [
            "margin_percent_overall",
            "avg_order_quantity",
            "revenue_per_customer",
        ],
    )

    total_revenue = float(product_metrics["revenue"].sum())
    total_products = int(len(product_metrics))
    avg_margin = float(product_metrics["margin_percent_overall"].mean())
    portfolio_efficiency = (
        total_revenue / total_products if total_products > 0 else 0.0
    )

    # Enhanced KPI Dashboard
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric("Total Products", total_products)
        st.caption("Portfolio Size")

    with col2:
        st.metric("Total Revenue", f"KES {total_revenue:,.0f}")
        st.caption("Portfolio Value")

    with col3:
        st.metric("Avg Margin %", f"{avg_margin:.1f}%")
        st.caption("Profitability")

    with col4:
        profitable_products = len(
            product_metrics[product_metrics["margin_percent_overall"] > 15]
        )
        st.metric("Profitable Products", f"{profitable_products}/{total_products}")
        st.caption("Portfolio Health")

    with col5:
        st.metric("Portfolio Efficiency", f"KES {portfolio_efficiency:,.0f}")
        st.caption("Revenue per Product")

    # Strategic Product Portfolio Analysis
    st.subheader("📈 Strategic Portfolio Analysis")

    # Determine sorting based on view
    if performance_view == "Revenue":
        y_column = "revenue"
        title_suffix = "Revenue"
        color_scale = "Viridis"
    elif performance_view == "Quantity":
        y_column = "quantity"
        title_suffix = "Volume"
        color_scale = "Blues"
    elif performance_view == "Margin":
        y_column = "margin"
        title_suffix = "Margin"
        color_scale = "Greens"
    elif performance_view == "Growth":
        # Using revenue as base; could be extended with explicit growth metric
        y_column = "revenue"
        title_suffix = "Revenue"
        color_scale = "Oranges"
    else:  # Efficiency
        y_column = "revenue_per_customer"
        title_suffix = "Revenue per Customer"
        color_scale = "Purples"

    # Final safety: ensure the chosen y_column is numeric
    product_metrics = _ensure_numeric(product_metrics, [y_column])

    top_products = safe_nlargest(product_metrics, 10, y_column)

    col1, col2 = st.columns(2)

    with col1:
        # Enhanced top products visualization
        text_vals = []
        for x in top_products[y_column]:
            if y_column in ["revenue", "margin", "revenue_per_customer"]:
                text_vals.append(f"KES {x:,.0f}")
            else:
                text_vals.append(f"{x:,.0f}")

        fig = px.bar(
            top_products,
            x="sku_name",
            y=y_column,
            title=f"🏆 Top 10 Products by {title_suffix}",
            color=y_column,
            color_continuous_scale=color_scale,
            text=text_vals,
            hover_data=["category", "margin_percent_overall"],
        )
        fig.update_layout(
            xaxis_tickangle=-45,
            showlegend=False,
            height=500,
        )
        st.plotly_chart(fig, use_container_width=True)

        # Strategic insight
        if product_metrics[y_column].sum() > 0:
            top_5_share = (
                top_products.head(5)[y_column].sum()
                / product_metrics[y_column].sum()
                * 100
            )
            st.info(
                f"**Portfolio Concentration:** Top 5 products generate {top_5_share:.1f}% of total {title_suffix.lower()}"
            )

    with col2:
        # Enhanced product performance matrix
        matrix_products = safe_nlargest(product_metrics, 25, "revenue")
        fig = px.scatter(
            matrix_products,
            x="quantity",
            y="margin_percent_overall",
            size="revenue",
            color="revenue_per_customer",
            title="📊 Product Performance Matrix: Volume vs Margin %",
            hover_data=["sku_name", "category", "customer_id"],
            color_continuous_scale="RdYlGn",
            size_max=40,
        )

        # Add strategic quadrants
        median_quantity = float(product_metrics["quantity"].median())
        median_margin = float(product_metrics["margin_percent_overall"].median())

        fig.add_hline(y=median_margin, line_dash="dash", line_color="red", line_width=2)
        fig.add_vline(x=median_quantity, line_dash="dash", line_color="red", line_width=2)

        # Strategic quadrant annotations
        fig.add_annotation(
            x=median_quantity * 1.8 if median_quantity > 0 else 1,
            y=median_margin * 1.8 if median_margin > 0 else 1,
            text="🌟 STARS\n(High Volume, High Margin)",
            showarrow=False,
            font=dict(size=10, color="green"),
        )

        fig.add_annotation(
            x=max(median_quantity * 0.5, 0),
            y=median_margin * 1.8 if median_margin > 0 else 1,
            text="💎 HIGH POTENTIAL\n(Low Volume, High Margin)",
            showarrow=False,
            font=dict(size=10, color="orange"),
        )

        fig.add_annotation(
            x=median_quantity * 1.8 if median_quantity > 0 else 1,
            y=max(median_margin * 0.5, 0),
            text="📦 CORE BUSINESS\n(High Volume, Low Margin)",
            showarrow=False,
            font=dict(size=10, color="blue"),
        )

        fig.add_annotation(
            x=max(median_quantity * 0.5, 0),
            y=max(median_margin * 0.5, 0),
            text="🔧 REVIEW NEEDED\n(Low Volume, Low Margin)",
            showarrow=False,
            font=dict(size=10, color="red"),
        )

        fig.update_layout(
            xaxis_title="Total Quantity Sold",
            yaxis_title="Margin Percentage (%)",
            height=500,
        )
        st.plotly_chart(fig, use_container_width=True)

    # Advanced Portfolio Segmentation
    st.subheader("🎯 Strategic Portfolio Segmentation")

    rev_q80 = float(product_metrics["revenue"].quantile(0.8))
    rev_q60 = float(product_metrics["revenue"].quantile(0.6))
    margin_q80 = float(product_metrics["margin_percent_overall"].quantile(0.8))
    margin_q60 = float(product_metrics["margin_percent_overall"].quantile(0.6))

    product_metrics["portfolio_segment"] = np.select(
        [
            (product_metrics["revenue"] > rev_q80)
            & (product_metrics["margin_percent_overall"] > margin_q80),
            (product_metrics["revenue"] > rev_q80)
            & (product_metrics["margin_percent_overall"] <= margin_q80),
            (product_metrics["revenue"] <= rev_q80)
            & (product_metrics["margin_percent_overall"] > margin_q80),
            (product_metrics["revenue"] <= rev_q60)
            & (product_metrics["margin_percent_overall"] <= margin_q60),
        ],
        [
            "Strategic Stars",
            "Cash Generators",
            "Growth Opportunities",
            "Review Candidates",
        ],
        default="Core Portfolio",
    )

    segment_summary = product_metrics["portfolio_segment"].value_counts()

    col1, col2 = st.columns(2)

    with col1:
        # Enhanced portfolio segmentation
        fig = px.pie(
            segment_summary,
            values=segment_summary.values,
            names=segment_summary.index,
            title="📊 Product Portfolio Segmentation",
            color=segment_summary.index,
            color_discrete_map={
                "Strategic Stars": "#FFD700",
                "Cash Generators": "#00CC96",
                "Growth Opportunities": "#1F77B4",
                "Core Portfolio": "#636EFA",
                "Review Candidates": "#EF553B",
            },
            hole=0.4,
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Segment performance analysis
        segment_performance = (
            product_metrics.groupby("portfolio_segment")
            .agg(
                {
                    "revenue": "sum",
                    "margin_percent_overall": "mean",
                    "sku_id": "count",
                    "revenue_per_customer": "mean",
                }
            )
            .reset_index()
        )

        segment_performance = _ensure_numeric(
            segment_performance,
            [
                "revenue",
                "margin_percent_overall",
                "sku_id",
                "revenue_per_customer",
            ],
        )

        fig = px.bar(
            segment_performance,
            x="portfolio_segment",
            y="revenue",
            color="margin_percent_overall",
            title="💰 Revenue Contribution by Segment",
            text=[f"KES {x:,.0f}" for x in segment_performance["revenue"]],
            color_continuous_scale="Viridis",
            hover_data=["margin_percent_overall"],
        )
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)

    # Enhanced Product Performance Intelligence
    st.subheader("📋 Product Performance Intelligence")

    # Add comprehensive performance rating
    def get_strategic_rating(row):
        if (
            row["margin_percent_overall"] > 25
            and row["revenue"] > product_metrics["revenue"].quantile(0.7)
        ):
            return "⭐ Strategic Star"
        elif (
            row["margin_percent_overall"] > 20
            and row["revenue_per_customer"]
            > product_metrics["revenue_per_customer"].quantile(0.7)
        ):
            return "💎 Premium Performer"
        elif row["margin_percent_overall"] > 15:
            return "✅ Solid Contributor"
        elif row["revenue"] > product_metrics["revenue"].quantile(0.6):
            return "📦 Volume Driver"
        else:
            return "🔴 Strategic Review"

    product_metrics["strategic_rating"] = product_metrics.apply(
        get_strategic_rating, axis=1
    )

    # Display enhanced product intelligence
    display_columns = [
        "sku_name",
        "category",
        "revenue",
        "quantity",
        "margin",
        "margin_percent_overall",
        "revenue_per_customer",
        "strategic_rating",
        "portfolio_segment",
    ]

    df_show = (
        safe_nlargest(product_metrics, 15, "revenue")[display_columns]
        .rename(
            columns={
                "sku_name": "Product",
                "category": "Category",
                "revenue": "Revenue",
                "quantity": "Quantity",
                "margin": "Margin",
                "margin_percent_overall": "Margin %",
                "revenue_per_customer": "Revenue/Customer",
                "strategic_rating": "Strategic Rating",
                "portfolio_segment": "Portfolio Segment",
            }
        )
        .round(2)
    )

    st.dataframe(
        df_show.style.format(
            {
                "Revenue": "KES {:,.0f}",
                "Margin": "KES {:,.0f}",
                "Revenue/Customer": "KES {:,.0f}",
            }
        ),
        use_container_width=True,
        height=400,
    )


def apply_product_filters(data, date_range, category_filter):
    """Apply filters to product data"""
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

    # Category filter
    if category_filter != "All":
        filtered_data = filtered_data[filtered_data["category"] == category_filter]

    return filtered_data


def render_category_strategy(sales_data):
    """Render comprehensive category strategy analysis"""
    st.header("📊 Category Strategy & Performance")

    # Enhanced Category Performance Intelligence
    st.subheader("🏅 Category Performance Intelligence")

    category_metrics = (
        sales_data.groupby("category")
        .agg(
            {
                "revenue": "sum",
                "quantity": "sum",
                "margin": "sum",
                "sku_id": "nunique",
                "customer_id": "nunique",
                "date": "count",
            }
        )
        .reset_index()
    )

    category_metrics = _ensure_numeric(
        category_metrics,
        ["revenue", "quantity", "margin", "sku_id", "customer_id", "date"],
    )

    category_metrics["margin_percent"] = (
        category_metrics["margin"] / category_metrics["revenue"]
    ).replace([np.inf, -np.inf], 0) * 100
    category_metrics["revenue_share"] = (
        category_metrics["revenue"] / category_metrics["revenue"].sum()
    ) * 100
    category_metrics["avg_revenue_per_sku"] = (
        category_metrics["revenue"] / category_metrics["sku_id"]
    ).replace([np.inf, -np.inf], 0)
    category_metrics["transaction_frequency"] = (
        category_metrics["date"] / category_metrics["customer_id"]
    ).replace([np.inf, -np.inf], 0)

    category_metrics = _ensure_numeric(
        category_metrics,
        [
            "margin_percent",
            "revenue_share",
            "avg_revenue_per_sku",
            "transaction_frequency",
        ],
    )

    col1, col2 = st.columns(2)

    with col1:
        # Enhanced category treemap
        fig = px.treemap(
            category_metrics,
            path=["category"],
            values="revenue",
            color="margin_percent",
            title="📈 Revenue Distribution by Category",
            hover_data=["quantity", "sku_id", "transaction_frequency"],
            color_continuous_scale="RdYlGn",
            color_continuous_midpoint=float(category_metrics["margin_percent"].median()),
        )
        st.plotly_chart(fig, use_container_width=True)

        # Category concentration insight
        top_3_categories = safe_nlargest(category_metrics, 3, "revenue")
        top_3_share = (
            top_3_categories["revenue"].sum() / category_metrics["revenue"].sum() * 100
        )
        st.info(
            f"**Market Concentration:** Top 3 categories represent {top_3_share:.1f}% of total revenue"
        )

    with col2:
        # Enhanced category performance matrix
        fig = px.scatter(
            category_metrics,
            x="revenue_share",
            y="margin_percent",
            size="quantity",
            color="transaction_frequency",
            title="🎯 Category Performance: Market Share vs Margin %",
            hover_data=["category", "sku_id", "customer_id"],
            size_max=60,
            color_continuous_scale="Viridis",
        )

        # Add strategic performance bands
        fig.add_hline(
            y=20,
            line_dash="dot",
            line_color="green",
            annotation_text="High Margin",
        )
        fig.add_hline(
            y=10,
            line_dash="dot",
            line_color="orange",
            annotation_text="Medium Margin",
        )
        fig.add_vline(
            x=20,
            line_dash="dot",
            line_color="blue",
            annotation_text="Significant Share",
        )

        st.plotly_chart(fig, use_container_width=True)

    # Advanced Category Growth Intelligence
    st.subheader("📈 Category Growth & Trend Analysis")

    # Enhanced monthly category trends
    sales_data = sales_data.copy()
    sales_data["month"] = pd.to_datetime(sales_data["date"]).dt.to_period("M")
    monthly_category = (
        sales_data.groupby(["month", "category"])
        .agg({"revenue": "sum", "quantity": "sum", "customer_id": "nunique"})
        .reset_index()
    )

    monthly_category = _ensure_numeric(
        monthly_category, ["revenue", "quantity", "customer_id"]
    )

    monthly_category["month"] = monthly_category["month"].dt.to_timestamp()

    # Strategic Category Selector
    categories = sorted(sales_data["category"].unique())
    default_cats = categories[:3] if len(categories) >= 3 else categories
    selected_categories = st.multiselect(
        "🎯 Select Categories for Strategic Analysis",
        categories,
        default=default_cats,
        key="category_analysis",
    )

    if selected_categories:
        filtered_trends = monthly_category[
            monthly_category["category"].isin(selected_categories)
        ]

        col1, col2 = st.columns(2)

        with col1:
            # Enhanced revenue trends with strategic context
            fig = px.line(
                filtered_trends,
                x="month",
                y="revenue",
                color="category",
                title="📊 Monthly Revenue Trends by Category",
                markers=True,
                line_shape="spline",
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Customer engagement trends
            fig = px.line(
                filtered_trends,
                x="month",
                y="customer_id",
                color="category",
                title="🧑‍🤝‍🧑 Customer Engagement Trends by Category",
                markers=True,
                line_shape="spline",
            )
            st.plotly_chart(fig, use_container_width=True)

    # Enhanced Category Portfolio Strategy Framework
    st.subheader("🎯 Category Portfolio Strategy Framework")

    # Calculate comprehensive category strategic scores
    category_metrics["growth_score"] = np.random.uniform(
        0, 100, len(category_metrics)
    )  # Simulated growth
    category_metrics["profitability_score"] = category_metrics["margin_percent"] * 2
    category_metrics["market_position_score"] = category_metrics["revenue_share"] * 2
    category_metrics["customer_engagement_score"] = (
        category_metrics["transaction_frequency"] * 20
    )

    category_metrics = _ensure_numeric(
        category_metrics,
        [
            "growth_score",
            "profitability_score",
            "market_position_score",
            "customer_engagement_score",
        ],
    )

    # Enhanced strategic scoring
    category_metrics["strategic_importance_score"] = (
        category_metrics["market_position_score"] * 0.3
        + category_metrics["profitability_score"] * 0.3
        + category_metrics["growth_score"] * 0.2
        + category_metrics["customer_engagement_score"] * 0.2
    )

    category_metrics = _ensure_numeric(
        category_metrics, ["strategic_importance_score"]
    )

    # Enhanced strategy recommendations
    def get_category_strategy(row):
        if row["strategic_importance_score"] > 80:
            return "🚀 Strategic Growth & Investment"
        elif row["strategic_importance_score"] > 65:
            return "💎 Optimize & Defend"
        elif row["strategic_importance_score"] > 50:
            return "📊 Selective Development"
        elif row["strategic_importance_score"] > 35:
            return "🔧 Transform & Reposition"
        else:
            return "⚡ Review & Rationalize"

    category_metrics["strategy_recommendation"] = category_metrics.apply(
        get_category_strategy, axis=1
    )

    # Enhanced Strategy Visualization
    fig = px.scatter(
        category_metrics,
        x="revenue_share",
        y="profitability_score",
        size="strategic_importance_score",
        color="strategy_recommendation",
        title="📈 Category Strategy Matrix: Market Share vs Profitability",
        hover_data=["category", "growth_score", "customer_engagement_score"],
        size_max=50,
        color_discrete_map={
            "🚀 Strategic Growth & Investment": "#00CC96",
            "💎 Optimize & Defend": "#1F77B4",
            "📊 Selective Development": "#FFA500",
            "🔧 Transform & Reposition": "#EF553B",
            "⚡ Review & Rationalize": "#FF0000",
        },
    )

    median_share = float(category_metrics["revenue_share"].median())
    median_profitability = float(category_metrics["profitability_score"].median())

    fig.add_hline(
        y=median_profitability, line_dash="dash", line_color="red", line_width=2
    )
    fig.add_vline(x=median_share, line_dash="dash", line_color="red", line_width=2)

    st.plotly_chart(fig, use_container_width=True)

    # Enhanced Category Strategy Playbook
    st.subheader("📋 Category Strategy Playbook")

    strategy_framework = {
        "🚀 Strategic Growth & Investment": {
            "focus": "Market leadership and aggressive expansion",
            "key_initiatives": [
                "Increase marketing investment by 40-60%",
                "Expand product portfolio and variants",
                "Enhance distribution network coverage",
                "Develop premium and innovative offerings",
            ],
            "investment_level": "High",
            "expected_roi": "25-40%",
            "risk_level": "Medium-High",
        },
        "💎 Optimize & Defend": {
            "focus": "Profitability optimization and market defense",
            "key_initiatives": [
                "Focus on operational efficiency improvements",
                "Enhance customer loyalty programs",
                "Optimize pricing and promotion strategies",
                "Improve product quality and service delivery",
            ],
            "investment_level": "Medium",
            "expected_roi": "18-28%",
            "risk_level": "Low-Medium",
        },
        "📊 Selective Development": {
            "focus": "Targeted growth in promising segments",
            "key_initiatives": [
                "Identify and target high-potential customer segments",
                "Develop specialized product offerings",
                "Build strategic partnerships and alliances",
                "Implement test-and-learn approach for new initiatives",
            ],
            "investment_level": "Medium",
            "expected_roi": "20-30%",
            "risk_level": "Medium",
        },
        "🔧 Transform & Reposition": {
            "focus": "Fundamental business model transformation",
            "key_initiatives": [
                "Conduct comprehensive market repositioning",
                "Rebrand and relaunch product offerings",
                "Explore new distribution channels",
                "Implement radical cost restructuring",
            ],
            "investment_level": "High",
            "expected_roi": "15-25%",
            "risk_level": "High",
        },
        "⚡ Review & Rationalize": {
            "focus": "Portfolio optimization and potential exit",
            "key_initiatives": [
                "Conduct deep performance analysis",
                "Identify cost reduction opportunities",
                "Evaluate strategic alternatives including divestment",
                "Develop exit strategy if necessary",
            ],
            "investment_level": "Low",
            "expected_roi": "Variable",
            "risk_level": "Medium-High",
        },
    }

    for _, category in category_metrics.iterrows():
        with st.expander(
            f"📋 {category['category']} - {category['strategy_recommendation']}"
        ):
            strategy = strategy_framework[category["strategy_recommendation"]]

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Revenue Share", f"{category['revenue_share']:.1f}%")

            with col2:
                st.metric("Margin %", f"{category['margin_percent']:.1f}%")

            with col3:
                st.metric(
                    "Strategic Score",
                    f"{category['strategic_importance_score']:.0f}",
                )

            with col4:
                st.metric("Risk Level", strategy["risk_level"])

            st.write(f"**Strategic Focus:** {strategy['focus']}")
            st.write(
                f"**Investment Level:** {strategy['investment_level']} | **Expected ROI:** {strategy['expected_roi']}"
            )

            st.write("**Key Strategic Initiatives:**")
            for initiative in strategy["key_initiatives"]:
                st.write(f"✅ {initiative}")

            col1, col2 = st.columns(2)
            with col1:
                if st.button(
                    f"📋 View Detailed Plan",
                    key=f"cat_plan_{category['category']}",
                ):
                    st.success(
                        f"Detailed strategic plan loaded for {category['category']}"
                    )

            with col2:
                if st.button(
                    f"🚀 Execute Strategy",
                    key=f"cat_execute_{category['category']}",
                ):
                    st.success(
                        f"Strategy execution initiated for {category['category']}"
                    )


def render_lifecycle_analytics(sales_data):
    """Render advanced product lifecycle and trend intelligence"""
    st.header("🔄 Product Lifecycle Analytics & Trend Intelligence")

    st.info(
        """
    **💡 Strategic Context:** Track product evolution through lifecycle stages, identify emerging trends, 
    and optimize portfolio timing for maximum market impact and profitability.
    """
    )

    sales_data = sales_data.copy()
    sales_data["date"] = pd.to_datetime(sales_data["date"])

    # Enhanced Seasonality Intelligence
    st.subheader("📅 Advanced Seasonality & Pattern Analysis")

    # Add comprehensive time-based features
    sales_data["day_of_week"] = sales_data["date"].dt.day_name()
    sales_data["month"] = sales_data["date"].dt.month_name()
    sales_data["quarter"] = sales_data["date"].dt.quarter
    sales_data["week_of_year"] = sales_data["date"].dt.isocalendar().week

    # Weekly Pattern Intelligence
    weekly_patterns = (
        sales_data.groupby(["day_of_week", "category"])
        .agg(
            {
                "revenue": "mean",
                "quantity": "mean",
                "margin_percent": "mean",
                "customer_id": "nunique",
            }
        )
        .reset_index()
    )

    weekly_patterns = _ensure_numeric(
        weekly_patterns,
        ["revenue", "quantity", "margin_percent", "customer_id"],
    )

    # Order days properly for meaningful analysis
    days_order = [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
    ]
    weekly_patterns["day_of_week"] = pd.Categorical(
        weekly_patterns["day_of_week"], categories=days_order, ordered=True
    )
    weekly_patterns = weekly_patterns.sort_values("day_of_week")

    col1, col2 = st.columns(2)

    with col1:
        # Enhanced weekly revenue patterns
        fig = px.line(
            weekly_patterns,
            x="day_of_week",
            y="revenue",
            color="category",
            title="📊 Strategic Weekly Revenue Patterns by Category",
            markers=True,
            line_shape="spline",
        )
        fig.update_layout(
            xaxis_title="Day of Week",
            yaxis_title="Average Revenue (KES)",
            height=400,
        )
        st.plotly_chart(fig, use_container_width=True)

        # Weekly pattern insights
        if not weekly_patterns.empty:
            peak_row = weekly_patterns.loc[weekly_patterns["revenue"].idxmax()]
            peak_day = peak_row["day_of_week"]
            st.success(
                f"**Peak Performance:** Highest revenue typically occurs on **{peak_day}**"
            )

    with col2:
        # Customer engagement patterns
        fig = px.line(
            weekly_patterns,
            x="day_of_week",
            y="customer_id",
            color="category",
            title="🧑‍🤝‍🧑 Weekly Customer Engagement Patterns",
            markers=True,
            line_shape="spline",
        )
        fig.update_layout(
            xaxis_title="Day of Week",
            yaxis_title="Unique Customers",
            height=400,
        )
        st.plotly_chart(fig, use_container_width=True)

    # Monthly Seasonality Intelligence
    st.subheader("🌦️ Monthly Seasonality & Cyclical Patterns")

    monthly_seasonality = (
        sales_data.groupby(["month", "category"])
        .agg(
            {
                "revenue": "sum",
                "quantity": "sum",
                "margin_percent": "mean",
                "customer_id": "nunique",
            }
        )
        .reset_index()
    )

    monthly_seasonality = _ensure_numeric(
        monthly_seasonality,
        ["revenue", "quantity", "margin_percent", "customer_id"],
    )

    # Proper month ordering
    months_order = [
        "January",
        "February",
        "March",
        "April",
        "May",
        "June",
        "July",
        "August",
        "September",
        "October",
        "November",
        "December",
    ]
    monthly_seasonality["month"] = pd.Categorical(
        monthly_seasonality["month"],
        categories=months_order,
        ordered=True,
    )
    monthly_seasonality = monthly_seasonality.sort_values("month")

    col1, col2 = st.columns(2)

    with col1:
        # Revenue seasonality
        fig = px.line(
            monthly_seasonality,
            x="month",
            y="revenue",
            color="category",
            title="📈 Monthly Revenue Seasonality Analysis",
            markers=True,
            line_shape="spline",
        )
        fig.update_layout(
            xaxis_tickangle=-45,
            height=400,
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Margin seasonality
        fig = px.line(
            monthly_seasonality,
            x="month",
            y="margin_percent",
            color="category",
            title="💰 Monthly Margin Pattern Analysis",
            markers=True,
            line_shape="spline",
        )
        fig.update_layout(
            xaxis_tickangle=-45,
            yaxis_title="Margin Percentage (%)",
            height=400,
        )
        st.plotly_chart(fig, use_container_width=True)

    # Advanced Product Lifecycle Intelligence
    st.subheader("📈 Strategic Product Lifecycle Analysis")

    product_lifecycle = (
        sales_data.groupby(["sku_id", "sku_name", "category"])
        .agg(
            {
                "date": ["min", "max", "count"],
                "revenue": "sum",
                "quantity": "sum",
                "margin": "sum",
                "customer_id": "nunique",
            }
        )
        .reset_index()
    )

    product_lifecycle.columns = [
        "sku_id",
        "sku_name",
        "category",
        "first_sale",
        "last_sale",
        "transaction_count",
        "total_revenue",
        "total_quantity",
        "total_margin",
        "unique_customers",
    ]

    product_lifecycle = _ensure_numeric(
        product_lifecycle,
        [
            "transaction_count",
            "total_revenue",
            "total_quantity",
            "total_margin",
            "unique_customers",
        ],
    )

    product_lifecycle["product_age_days"] = (
        product_lifecycle["last_sale"] - product_lifecycle["first_sale"]
    ).dt.days

    # Avoid division by zero
    product_lifecycle["product_age_days"] = product_lifecycle["product_age_days"].replace(
        0, 1
    )

    product_lifecycle["avg_daily_revenue"] = (
        product_lifecycle["total_revenue"] / product_lifecycle["product_age_days"]
    ).replace([np.inf, -np.inf], 0)
    product_lifecycle["avg_daily_quantity"] = (
        product_lifecycle["total_quantity"] / product_lifecycle["product_age_days"]
    ).replace([np.inf, -np.inf], 0)
    product_lifecycle["customer_penetration"] = (
        product_lifecycle["unique_customers"] / product_lifecycle["transaction_count"]
    ).replace([np.inf, -np.inf], 0)
    product_lifecycle["margin_percent"] = (
        product_lifecycle["total_margin"] / product_lifecycle["total_revenue"]
    ).replace([np.inf, -np.inf], 0) * 100

    product_lifecycle = _ensure_numeric(
        product_lifecycle,
        [
            "product_age_days",
            "avg_daily_revenue",
            "avg_daily_quantity",
            "customer_penetration",
            "margin_percent",
        ],
    )

    def get_lifecycle_stage(row):
        age = row["product_age_days"]
        avg_rev = row["avg_daily_revenue"]
        median_rev = product_lifecycle["avg_daily_revenue"].median()
        q70_rev = product_lifecycle["avg_daily_revenue"].quantile(0.7)
        q30_rev = product_lifecycle["avg_daily_revenue"].quantile(0.3)

        if age < 90:
            return "🚀 Introduction"
        elif age < 365:
            if avg_rev > median_rev:
                return "📈 Growth"
            else:
                return "🔧 Early Maturity"
        elif avg_rev > q70_rev:
            return "💎 Peak Maturity"
        elif avg_rev > q30_rev:
            return "📊 Late Maturity"
        else:
            return "📉 Decline"

    product_lifecycle["lifecycle_stage"] = product_lifecycle.apply(
        get_lifecycle_stage, axis=1
    )

    col1, col2 = st.columns(2)

    with col1:
        stage_summary = product_lifecycle["lifecycle_stage"].value_counts()
        fig = px.pie(
            stage_summary,
            values=stage_summary.values,
            names=stage_summary.index,
            title="🔄 Product Lifecycle Stage Distribution",
            color=stage_summary.index,
            color_discrete_map={
                "🚀 Introduction": "#1F77B4",
                "📈 Growth": "#00CC96",
                "🔧 Early Maturity": "#FFA500",
                "💎 Peak Maturity": "#FFD700",
                "📊 Late Maturity": "#636EFA",
                "📉 Decline": "#EF553B",
            },
            hole=0.4,
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.scatter(
            product_lifecycle,
            x="product_age_days",
            y="avg_daily_revenue",
            color="lifecycle_stage",
            size="total_revenue",
            title="🎯 Product Lifecycle Matrix: Age vs Daily Revenue",
            hover_data=["sku_name", "category", "margin_percent"],
            size_max=40,
            color_discrete_map={
                "🚀 Introduction": "#1F77B4",
                "📈 Growth": "#00CC96",
                "🔧 Early Maturity": "#FFA500",
                "💎 Peak Maturity": "#FFD700",
                "📊 Late Maturity": "#636EFA",
                "📉 Decline": "#EF553B",
            },
        )
        fig.update_layout(
            xaxis_title="Product Age (Days)",
            yaxis_title="Average Daily Revenue (KES)",
            height=500,
        )
        st.plotly_chart(fig, use_container_width=True)

    # Lifecycle Performance Intelligence
    st.subheader("📊 Lifecycle Stage Performance Analysis")

    lifecycle_performance = (
        product_lifecycle.groupby("lifecycle_stage")
        .agg(
            {
                "sku_id": "count",
                "total_revenue": "sum",
                "avg_daily_revenue": "mean",
                "margin_percent": "mean",
                "customer_penetration": "mean",
                "product_age_days": "mean",
            }
        )
        .reset_index()
    )

    lifecycle_performance = _ensure_numeric(
        lifecycle_performance,
        [
            "sku_id",
            "total_revenue",
            "avg_daily_revenue",
            "margin_percent",
            "customer_penetration",
            "product_age_days",
        ],
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        intro_products = len(
            product_lifecycle[product_lifecycle["lifecycle_stage"] == "🚀 Introduction"]
        )
        st.metric("Introduction Stage", intro_products)

    with col2:
        growth_products = len(
            product_lifecycle[product_lifecycle["lifecycle_stage"] == "📈 Growth"]
        )
        st.metric("Growth Stage", growth_products)

    with col3:
        maturity_products = len(
            product_lifecycle[
                product_lifecycle["lifecycle_stage"].str.contains("Maturity")
            ]
        )
        st.metric("Maturity Stage", maturity_products)

    with col4:
        decline_products = len(
            product_lifecycle[product_lifecycle["lifecycle_stage"] == "📉 Decline"]
        )
        st.metric("Decline Stage", decline_products)

    # Lifecycle Strategy Recommendations
    st.subheader("💡 Lifecycle Stage Strategy Playbook")

    lifecycle_strategies = {
        "🚀 Introduction": {
            "focus": "Market Entry & Awareness Building",
            "key_actions": [
                "Aggressive sampling and trial campaigns",
                "Strategic pricing to gain market share",
                "Limited distribution in key outlets",
                "Heavy investment in consumer education",
            ],
            "investment_level": "High",
            "profit_expectation": "Negative to Break-even",
            "success_metrics": ["Awareness", "Trial Rate", "Distribution Coverage"],
        },
        "📈 Growth": {
            "focus": "Rapid Expansion & Market Penetration",
            "key_actions": [
                "Expand distribution network aggressively",
                "Increase production capacity",
                "Build brand loyalty through consistent marketing",
                "Introduce product variants and extensions",
            ],
            "investment_level": "Very High",
            "profit_expectation": "Growing Profitability",
            "success_metrics": [
                "Market Share",
                "Revenue Growth",
                "Customer Acquisition",
            ],
        },
        "🔧 Early Maturity": {
            "focus": "Market Consolidation & Efficiency",
            "key_actions": [
                "Optimize operational efficiency",
                "Focus on customer retention",
                "Defend market share against competitors",
                "Selective product improvement",
            ],
            "investment_level": "Medium",
            "profit_expectation": "Peak Profitability",
            "success_metrics": [
                "Market Share",
                "Profit Margin",
                "Customer Loyalty",
            ],
        },
        "💎 Peak Maturity": {
            "focus": "Cash Generation & Market Defense",
            "key_actions": [
                "Maximize cash flow generation",
                "Maintain competitive positioning",
                "Cost optimization and efficiency",
                "Prepare for product evolution",
            ],
            "investment_level": "Low-Medium",
            "profit_expectation": "High but Declining",
            "success_metrics": ["Cash Flow", "Market Position", "Operational Efficiency"],
        },
        "📊 Late Maturity": {
            "focus": "Selective Investment & Harvesting",
            "key_actions": [
                "Selective marketing investment",
                "Cost reduction initiatives",
                "Explore niche market opportunities",
                "Prepare replacement strategy",
            ],
            "investment_level": "Low",
            "profit_expectation": "Declining",
            "success_metrics": ["Profit Margin", "Cash Generation", "Cost Control"],
        },
        "📉 Decline": {
            "focus": "Managed Exit & Resource Reallocation",
            "key_actions": [
                "Gradual market withdrawal",
                "Minimize maintenance costs",
                "Liquidate inventory strategically",
                "Reallocate resources to growth products",
            ],
            "investment_level": "Minimal",
            "profit_expectation": "Minimal or Negative",
            "success_metrics": [
                "Cost Minimization",
                "Resource Recovery",
                "Smooth Exit",
            ],
        },
    }

    for stage, strategy in lifecycle_strategies.items():
        with st.expander(f"{stage} - {strategy['focus']}"):
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Investment Level", strategy["investment_level"])

            with col2:
                st.metric("Profit Expectation", strategy["profit_expectation"])

            with col3:
                stage_count = len(
                    product_lifecycle[product_lifecycle["lifecycle_stage"] == stage]
                )
                st.metric("Products in Stage", stage_count)

            st.write("**Key Strategic Actions:**")
            for action in strategy["key_actions"]:
                st.write(f"✅ {action}")

            st.write("**Success Metrics:**")
            st.write(f"📊 {', '.join(strategy['success_metrics'])}")

            stage_products = product_lifecycle[
                product_lifecycle["lifecycle_stage"] == stage
            ]
            if not stage_products.empty:
                st.write("**Products in this Stage:**")
                for _, product in stage_products.head(3).iterrows():
                    st.write(
                        f"- {product['sku_name']} (KES {product['avg_daily_revenue']:,.0f}/day)"
                    )

    # Trend Detection & Forecasting
    st.subheader("🔍 Emerging Trend Detection")

    # Use month as a proper period/timestamp for ordering
    trend_data = sales_data.copy()
    trend_data["month"] = trend_data["date"].dt.to_period("M")

    monthly_trends = (
        trend_data.groupby(["month", "sku_id", "sku_name"])
        .agg(
            {
                "revenue": "sum",
                "quantity": "sum",
                "customer_id": "nunique",
            }
        )
        .reset_index()
    )

    monthly_trends = _ensure_numeric(
        monthly_trends, ["revenue", "quantity", "customer_id"]
    )

    monthly_trends["month"] = monthly_trends["month"].dt.to_timestamp()

    monthly_trends = monthly_trends.sort_values(["sku_id", "month"])
    monthly_trends["revenue_growth"] = (
        monthly_trends.groupby("sku_id")["revenue"].pct_change() * 100
    )
    monthly_trends["customer_growth"] = (
        monthly_trends.groupby("sku_id")["customer_id"].pct_change() * 100
    )

    if not monthly_trends.empty:
        recent_months = (
            monthly_trends["month"].sort_values(ascending=False).unique()[:3]
        )
        recent_trends = monthly_trends[monthly_trends["month"].isin(recent_months)]

        trending_products = (
            recent_trends.groupby(["sku_id", "sku_name"])
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

        trending_products = _ensure_numeric(
            trending_products,
            ["revenue_growth", "customer_growth", "revenue", "customer_id"],
        )

        trending_products["trend_score"] = (
            trending_products["revenue_growth"].fillna(0) * 0.6
            + trending_products["customer_growth"].fillna(0) * 0.4
        )

        trending_products = _ensure_numeric(trending_products, ["trend_score"])

        st.write("**🚀 High-Potential Trending Products (Last 3 Months):**")

        high_potential_trends = safe_nlargest(trending_products, 5, "trend_score")
        for _, product in high_potential_trends.iterrows():
            col1, col2, col3 = st.columns([3, 2, 1])
            with col1:
                st.write(f"**{product['sku_name']}**")
            with col2:
                growth_color = "green" if product["revenue_growth"] > 0 else "red"
                st.write(
                    f"<span style='color: {growth_color};'>"
                    f"{product['revenue_growth']:+.1f}% growth</span>",
                    unsafe_allow_html=True,
                )
            with col3:
                if st.button("📈 Analyze", key=f"trend_{product['sku_id']}"):
                    st.success(
                        f"Deep trend analysis started for {product['sku_name']}"
                    )


def render_optimization_engine(sales_data, data_gen):
    """Render advanced product optimization and recommendation engine"""
    st.header("🎯 Product Optimization Engine")

    st.info(
        """
    **💡 Strategic Context:** AI-powered optimization engine identifying margin improvement opportunities, 
    pricing optimization, bundling strategies, and portfolio rationalization recommendations.
    """
    )

    # Advanced Product Optimization Analysis
    st.subheader("💰 Advanced Margin & Pricing Optimization")

    product_analysis = (
        sales_data.groupby(["sku_id", "sku_name", "category"])
        .agg(
            {
                "revenue": "sum",
                "quantity": "sum",
                "margin": "sum",
                "margin_percent": "mean",
                "unit_price": "mean",
                "unit_cost": "mean",
                "customer_id": "nunique",
                "date": "count",
            }
        )
        .reset_index()
    )

    product_analysis = _ensure_numeric(
        product_analysis,
        [
            "revenue",
            "quantity",
            "margin",
            "margin_percent",
            "unit_price",
            "unit_cost",
            "customer_id",
            "date",
        ],
    )

    product_analysis["margin_percent_overall"] = (
        product_analysis["margin"] / product_analysis["revenue"]
    ).replace([np.inf, -np.inf], 0) * 100
    product_analysis["revenue_per_transaction"] = (
        product_analysis["revenue"] / product_analysis["date"]
    ).replace([np.inf, -np.inf], 0)
    product_analysis["customers_per_product"] = (
        product_analysis["customer_id"] / product_analysis["date"]
    ).replace([np.inf, -np.inf], 0)

    product_analysis = _ensure_numeric(
        product_analysis,
        [
            "margin_percent_overall",
            "revenue_per_transaction",
            "customers_per_product",
        ],
    )

    # Enhanced Optimization Scoring
    max_margin_pct = float(
        product_analysis["margin_percent_overall"].replace(0, np.nan).max() or 1
    )
    max_revenue = float(product_analysis["revenue"].replace(0, np.nan).max() or 1)
    max_cust = float(
        product_analysis["customers_per_product"].replace(0, np.nan).max() or 1
    )

    product_analysis["margin_optimization_score"] = (
        (1 - (product_analysis["margin_percent_overall"] / max_margin_pct)) * 0.4
        + (product_analysis["revenue"] / max_revenue) * 0.3
        + (product_analysis["customers_per_product"] / max_cust) * 0.3
    ) * 100

    product_analysis = _ensure_numeric(
        product_analysis, ["margin_optimization_score"]
    )

    def get_optimization_recommendation(row):
        high_rev_70 = product_analysis["revenue"].quantile(0.7)
        high_rev_50 = product_analysis["revenue"].quantile(0.5)
        low_rev_20 = product_analysis["revenue"].quantile(0.2)

        if row["margin_percent_overall"] < 10 and row["revenue"] > high_rev_70:
            return (
                "🚨 CRITICAL: High volume, critically low margin - urgent pricing review required"
            )
        elif row["margin_percent_overall"] < 15 and row["revenue"] > high_rev_50:
            return "💰 HIGH PRIORITY: Significant margin optimization opportunity"
        elif row["margin_percent_overall"] > 25 and row["revenue"] < low_rev_20:
            return "📈 GROWTH OPPORTUNITY: High margin, low volume - increase promotion & distribution"
        elif row["margin_percent_overall"] < 8:
            return "⚡ STRATEGIC REVIEW: Very low margin - consider cost reduction or discontinuation"
        elif row["margin_percent_overall"] > 30:
            return "💎 PREMIUM POSITIONING: Excellent margin - consider premium branding"
        else:
            return "✅ HEALTHY: Maintain current strategy with continuous monitoring"

    product_analysis["optimization_recommendation"] = product_analysis.apply(
        get_optimization_recommendation, axis=1
    )

    # Optimization Priority Dashboard
    st.subheader("📊 Optimization Priority Dashboard")

    product_analysis["optimization_priority"] = pd.cut(
        product_analysis["margin_optimization_score"],
        bins=[0, 30, 60, 80, 100],
        labels=[
            "Low Priority",
            "Medium Priority",
            "High Priority",
            "Critical Priority",
        ],
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        critical_count = len(
            product_analysis[product_analysis["optimization_priority"] == "Critical Priority"]
        )
        st.metric("🔴 Critical Priority", critical_count)

    with col2:
        high_count = len(
            product_analysis[product_analysis["optimization_priority"] == "High Priority"]
        )
        st.metric("🟠 High Priority", high_count)

    with col3:
        medium_count = len(
            product_analysis[product_analysis["optimization_priority"] == "Medium Priority"]
        )
        st.metric("🟡 Medium Priority", medium_count)

    with col4:
        low_count = len(
            product_analysis[product_analysis["optimization_priority"] == "Low Priority"]
        )
        st.metric("🟢 Low Priority", low_count)

    col1, col2 = st.columns(2)

    with col1:
        priority_summary = product_analysis["optimization_priority"].value_counts()
        fig = px.pie(
            priority_summary,
            values=priority_summary.values,
            names=priority_summary.index,
            title="🎯 Optimization Priority Distribution",
            color=priority_summary.index,
            color_discrete_map={
                "Critical Priority": "#FF0000",
                "High Priority": "#FFA500",
                "Medium Priority": "#FFD700",
                "Low Priority": "#00CC96",
            },
            hole=0.4,
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        top_opt = safe_nlargest(product_analysis, 20, "margin_optimization_score")
        fig = px.scatter(
            top_opt,
            x="revenue",
            y="margin_percent_overall",
            size="margin_optimization_score",
            color="optimization_priority",
            title="📈 Optimization Opportunity Matrix",
            hover_data=["sku_name", "optimization_recommendation"],
            size_max=40,
            color_discrete_map={
                "Critical Priority": "#FF0000",
                "High Priority": "#FFA500",
                "Medium Priority": "#FFD700",
                "Low Priority": "#00CC96",
            },
        )
        fig.update_layout(
            xaxis_title="Total Revenue (KES)",
            yaxis_title="Margin Percentage (%)",
        )
        st.plotly_chart(fig, use_container_width=True)

    # Detailed Optimization Recommendations
    st.subheader("💡 Detailed Optimization Recommendations")

    optimization_opportunities = safe_nlargest(
        product_analysis, 10, "margin_optimization_score"
    )

    for _, product in optimization_opportunities.iterrows():
        title_prefix = product["optimization_recommendation"].split(":")[0]
        with st.expander(f"🔍 {product['sku_name']} - {title_prefix}"):
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Current Margin", f"{product['margin_percent_overall']:.1f}%")

            with col2:
                st.metric("Total Revenue", f"KES {product['revenue']:,.0f}")

            with col3:
                st.metric(
                    "Optimization Score",
                    f"{product['margin_optimization_score']:.0f}",
                )

            with col4:
                st.metric("Priority", str(product["optimization_priority"]))

            st.write("**Recommended Actions:**")

            rec = product["optimization_recommendation"]
            if "CRITICAL" in rec:
                st.error(
                    """
                **Immediate Actions Required:**
                - Conduct urgent pricing strategy review  
                - Analyze cost structure for reduction opportunities  
                - Consider product reformulation or sourcing alternatives  
                - Prepare discontinuation plan if margin cannot be improved
                """
                )
            elif "HIGH PRIORITY" in rec:
                st.warning(
                    """
                **Priority Optimization Actions:**
                - Review and adjust pricing strategy  
                - Negotiate with suppliers for better terms  
                - Optimize promotional spending  
                - Explore packaging or size optimization
                """
                )
            elif "GROWTH OPPORTUNITY" in rec:
                st.success(
                    """
                **Growth Acceleration Actions:**
                - Increase marketing and promotion investment  
                - Expand distribution channels  
                - Develop complementary product bundles  
                - Enhance product visibility and positioning
                """
                )
            elif "PREMIUM POSITIONING" in rec:
                st.info(
                    """
                **Premium Strategy Actions:**
                - Develop premium branding and packaging  
                - Explore price premium opportunities  
                - Target high-value customer segments  
                - Enhance product features and benefits
                """
                )

            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button(
                    f"📊 Price Analysis", key=f"price_{product['sku_id']}"
                ):
                    st.success(
                        f"Price optimization analysis started for {product['sku_name']}"
                    )

            with col2:
                if st.button(
                    f"💰 Cost Review", key=f"cost_{product['sku_id']}"
                ):
                    st.success(
                        f"Cost structure review initiated for {product['sku_name']}"
                    )

            with col3:
                if st.button(
                    f"🎯 Action Plan", key=f"plan_{product['sku_id']}"
                ):
                    st.success(
                        f"Comprehensive optimization plan created for {product['sku_name']}"
                    )

    # Advanced Product Bundling Intelligence
    st.subheader("📦 Strategic Product Bundling Opportunities")

    st.write("**💡 AI-Powered Bundle Recommendations:**")

    strategic_bundles = [
        {
            "bundle_name": "🏠 Household Essentials Bundle",
            "products": [
                "Detergent 2L",
                "Fabric Softener",
                "Dish Soap",
                "Surface Cleaner",
            ],
            "strategic_rationale": (
                "Complementary usage patterns, high cross-purchase probability"
            ),
            "expected_impact": "+18% revenue, +22% customer value",
            "implementation_complexity": "Low",
            "estimated_roi": "35%",
        },
        {
            "bundle_name": "🥤 Beverage Refreshment Combo",
            "products": ["Soda 500ml", "Juice 1L", "Bottled Water", "Energy Drink"],
            "strategic_rationale": (
                "Seasonal demand alignment, impulse purchase synergy"
            ),
            "expected_impact": "+15% volume, +12% margin",
            "implementation_complexity": "Low",
            "estimated_roi": "28%",
        },
        {
            "bundle_name": "🍳 Premium Breakfast Package",
            "products": ["Artisan Bread", "Premium Butter", "Gourmet Jam", "Specialty Coffee"],
            "strategic_rationale": "Premium positioning, high-margin product combination",
            "expected_impact": (
                "+25% margin, +30% premium customer acquisition"
            ),
            "implementation_complexity": "Medium",
            "estimated_roi": "42%",
        },
        {
            "bundle_name": "🎯 Health & Wellness Pack",
            "products": ["Organic Snacks", "Herbal Tea", "Vitamin Water", "Protein Bars"],
            "strategic_rationale": (
                "Growing health-conscious segment, emerging trend alignment"
            ),
            "expected_impact": "+20% new customer acquisition, +35% category growth",
            "implementation_complexity": "Medium",
            "estimated_roi": "38%",
        },
    ]

    for bundle in strategic_bundles:
        with st.expander(f"{bundle['bundle_name']} - ROI: {bundle['estimated_roi']}"):
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Expected Impact", bundle["expected_impact"])

            with col2:
                st.metric("Implementation", bundle["implementation_complexity"])

            with col3:
                st.metric("Estimated ROI", bundle["estimated_roi"])

            st.write(f"**Products:** {', '.join(bundle['products'])}")
            st.write(f"**Strategic Rationale:** {bundle['strategic_rationale']}")

            col1, col2 = st.columns(2)
            with col1:
                if st.button(
                    f"📋 Create Bundle", key=f"create_{bundle['bundle_name']}"
                ):
                    st.success(
                        f"Bundle creation process started: {bundle['bundle_name']}"
                    )

            with col2:
                if st.button(
                    f"📊 Test Market", key=f"test_{bundle['bundle_name']}"
                ):
                    st.success(
                        f"Market testing initiated for: {bundle['bundle_name']}"
                    )

    # Portfolio Rationalization Analysis
    st.subheader("⚡ Portfolio Rationalization & Optimization")

    rationalization_candidates_base = product_analysis[
        (product_analysis["margin_percent_overall"] < 8)
        | (product_analysis["revenue"] < product_analysis["revenue"].quantile(0.2))
    ]
    rationalization_candidates = safe_nlargest(
        rationalization_candidates_base, 5, "margin_optimization_score"
    )

    if not rationalization_candidates.empty:
        st.write("**🔍 Products Recommended for Strategic Review:**")

        for _, product in rationalization_candidates.iterrows():
            col1, col2, col3 = st.columns([3, 2, 1])
            with col1:
                st.write(f"**{product['sku_name']}**")
                st.write(
                    f"Margin: {product['margin_percent_overall']:.1f}% | "
                    f"Revenue: KES {product['revenue']:,.0f}"
                )
            with col2:
                st.write(f"**{product['optimization_recommendation']}**")
            with col3:
                if st.button(
                    "🔧 Review", key=f"review_{product['sku_id']}"
                ):
                    st.success(
                        f"Strategic review initiated for {product['sku_name']}"
                    )


def render_innovation_pipeline(sales_data, data_gen):
    """Render innovation pipeline and new product opportunity analysis"""
    st.header("🚀 Product Innovation Pipeline")

    st.info(
        """
    **💡 Strategic Context:** Identify emerging market trends, evaluate new product opportunities, 
    and manage innovation pipeline from ideation to commercialization for sustained growth.
    """
    )

    # Market Opportunity Analysis
    st.subheader("📈 Emerging Market Opportunity Analysis")

    innovation_opportunities = [
        {
            "category": "🌿 Health & Wellness",
            "sub_segment": "Organic & Natural Products",
            "market_size": "KES 45M",
            "growth_rate": "+18% YoY",
            "competitive_intensity": "Medium",
            "strategic_fit": "92%",
            "implementation_timeline": "6-9 months",
            "estimated_roi": "35-45%",
            "key_drivers": ["Health consciousness", "Premiumization", "Sustainability"],
            "risk_level": "Medium",
        },
        {
            "category": "🥤 Premium Beverages",
            "sub_segment": "Craft & Specialty Drinks",
            "market_size": "KES 28M",
            "growth_rate": "+22% YoY",
            "competitive_intensity": "High",
            "strategic_fit": "85%",
            "implementation_timeline": "4-6 months",
            "estimated_roi": "30-40%",
            "key_drivers": [
                "Experiential consumption",
                "Brand storytelling",
                "Product differentiation",
            ],
            "risk_level": "Medium-High",
        },
        {
            "category": "🍽️ Ethnic & Traditional",
            "sub_segment": "Authentic Kenyan Cuisine",
            "market_size": "KES 32M",
            "growth_rate": "+15% YoY",
            "competitive_intensity": "Low-Medium",
            "strategic_fit": "88%",
            "implementation_timeline": "8-12 months",
            "estimated_roi": "25-35%",
            "key_drivers": [
                "Cultural pride",
                "Authenticity demand",
                "Tourism growth",
            ],
            "risk_level": "Low-Medium",
        },
        {
            "category": "🌱 Sustainable & Eco-Friendly",
            "sub_segment": "Green Packaging & Products",
            "market_size": "KES 22M",
            "growth_rate": "+25% YoY",
            "competitive_intensity": "High",
            "strategic_fit": "78%",
            "implementation_timeline": "9-15 months",
            "estimated_roi": "20-30%",
            "key_drivers": [
                "Environmental awareness",
                "Regulatory push",
                "Consumer preference",
            ],
            "risk_level": "Medium",
        },
        {
            "category": "⚡ Convenience & On-the-Go",
            "sub_segment": "Single-Serve & Portable",
            "market_size": "KES 38M",
            "growth_rate": "+12% YoY",
            "competitive_intensity": "Very High",
            "strategic_fit": "82%",
            "implementation_timeline": "3-5 months",
            "estimated_roi": "28-38%",
            "key_drivers": [
                "Urbanization",
                "Busy lifestyles",
                "Impulse purchasing",
            ],
            "risk_level": "Medium",
        },
        {
            "category": "🎯 Functional & Fortified",
            "sub_segment": "Health-Enhanced Products",
            "market_size": "KES 26M",
            "growth_rate": "+20% YoY",
            "competitive_intensity": "Medium",
            "strategic_fit": "90%",
            "implementation_timeline": "6-8 months",
            "estimated_roi": "32-42%",
            "key_drivers": [
                "Preventive health",
                "Nutrition awareness",
                "Aging population",
            ],
            "risk_level": "Low-Medium",
        },
    ]

    # Innovation Opportunity Dashboard
    st.subheader("💡 Innovation Opportunity Dashboard")

    # Compute innovation scores safely as numeric
    intensity_map = {
        "Very High": 20,
        "High": 40,
        "Medium": 60,
        "Low-Medium": 80,
        "Low": 100,
    }
    risk_map = {
        "Very High": 20,
        "High": 40,
        "Medium-High": 50,   # added to handle that risk level
        "Medium": 60,
        "Low-Medium": 80,
        "Low": 100,
    }

    for opportunity in innovation_opportunities:
        growth_score = int(
            opportunity["growth_rate"].replace("+", "").replace("% YoY", "")
        )
        fit_score = int(opportunity["strategic_fit"].replace("%", ""))
        roi_low = opportunity["estimated_roi"].split("-")[0].replace("%", "")
        roi_score = int(roi_low)

        competitive_score = intensity_map[opportunity["competitive_intensity"]]
        risk_score = risk_map[opportunity["risk_level"]]

        opportunity["innovation_score"] = (
            growth_score * 0.25
            + fit_score * 0.25
            + roi_score * 0.20
            + competitive_score * 0.15
            + risk_score * 0.15
        )

    innovation_opportunities.sort(key=lambda x: x["innovation_score"], reverse=True)

    for opportunity in innovation_opportunities[:4]:
        with st.expander(
            f"{opportunity['category']} - {opportunity['sub_segment']} | Score: {opportunity['innovation_score']:.0f}"
        ):
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Market Size", opportunity["market_size"])
                st.metric("Growth Rate", opportunity["growth_rate"])

            with col2:
                st.metric("Strategic Fit", opportunity["strategic_fit"])
                st.metric("Competitive Intensity", opportunity["competitive_intensity"])

            with col3:
                st.metric("Estimated ROI", opportunity["estimated_roi"])
                st.metric("Risk Level", opportunity["risk_level"])

            with col4:
                st.metric("Timeline", opportunity["implementation_timeline"])
                st.metric(
                    "Innovation Score", f"{opportunity['innovation_score']:.0f}"
                )

            st.write("**Key Market Drivers:**")
            for driver in opportunity["key_drivers"]:
                st.write(f"✅ {driver}")

            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button(
                    f"📋 Research", key=f"research_{opportunity['category']}"
                ):
                    st.success(
                        f"Market research initiated for {opportunity['category']}"
                    )
            with col2:
                if st.button(
                    f"🎯 Develop", key=f"develop_{opportunity['category']}"
                ):
                    st.success(
                        f"Product development started for {opportunity['category']}"
                    )
            with col3:
                if st.button(
                    f"🚀 Launch", key=f"launch_{opportunity['category']}"
                ):
                    st.success(
                        f"Commercial launch planning for {opportunity['category']}"
                    )

    # New Product Concept Development
    st.subheader("💎 New Product Concept Portfolio")

    product_concepts = [
        {
            "concept_name": "🌿 Organic Kenyan Tea Collection",
            "description": (
                "Premium organic tea blends featuring traditional Kenyan herbs "
                "and modern wellness benefits"
            ),
            "target_segment": "Health-conscious urban professionals, 25-45",
            "unique_value": "Authentic Kenyan heritage meets modern wellness trends",
            "development_stage": "Concept Validation",
            "estimated_launch": "Q3 2024",
            "required_investment": "KES 3.5M",
            "expected_revenue_y1": "KES 12M",
            "success_probability": "75%",
        },
        {
            "concept_name": "🥤 Moringa-Infused Energy Drink",
            "description": (
                "Natural energy beverage featuring superfood moringa and local fruit flavors"
            ),
            "target_segment": "Young adults, athletes, health enthusiasts",
            "unique_value": "First locally sourced natural energy drink in market",
            "development_stage": "Product Development",
            "estimated_launch": "Q4 2024",
            "required_investment": "KES 2.8M",
            "expected_revenue_y1": "KES 8.5M",
            "success_probability": "70%",
        },
        {
            "concept_name": "🍽️ Ready-to-Eat Traditional Meals",
            "description": (
                "Authentic Kenyan meals in convenient, premium packaging for urban consumers"
            ),
            "target_segment": "Busy professionals, students, expatriates",
            "unique_value": "Restaurant-quality traditional meals with convenience",
            "development_stage": "Market Testing",
            "estimated_launch": "Q2 2024",
            "required_investment": "KES 4.2M",
            "expected_revenue_y1": "KES 15M",
            "success_probability": "80%",
        },
        {
            "concept_name": "🌱 Eco-Friendly Household Range",
            "description": (
                "Sustainable cleaning and household products with biodegradable packaging"
            ),
            "target_segment": "Eco-conscious families, millennials",
            "unique_value": "First comprehensive eco-range in mainstream retail",
            "development_stage": "Concept Validation",
            "estimated_launch": "Q1 2025",
            "required_investment": "KES 5.1M",
            "expected_revenue_y1": "KES 18M",
            "success_probability": "65%",
        },
    ]

    # Innovation Pipeline Management
    st.subheader("📊 Innovation Pipeline Management")

    stage_progress = {
        "Concept Validation": 25,
        "Product Development": 50,
        "Market Testing": 75,
        "Commercial Ready": 90,
    }

    for concept in product_concepts:
        with st.expander(
            f"{concept['concept_name']} - Stage: {concept['development_stage']}"
        ):
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Required Investment", concept["required_investment"])
                st.metric("Expected Revenue Y1", concept["expected_revenue_y1"])

            with col2:
                st.metric("Success Probability", concept["success_probability"])
                st.metric("Estimated Launch", concept["estimated_launch"])

            with col3:
                progress = stage_progress.get(concept["development_stage"], 10)
                st.metric("Development Progress", f"{progress}%")
                st.progress(progress / 100)

            st.write(f"**Concept Description:** {concept['description']}")
            st.write(f"**Target Segment:** {concept['target_segment']}")
            st.write(
                f"**Unique Value Proposition:** {concept['unique_value']}"
            )

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                if st.button(
                    f"📊 Business Case", key=f"case_{concept['concept_name']}"
                ):
                    st.success(
                        f"Business case development for {concept['concept_name']}"
                    )
            with col2:
                if st.button(
                    f"🔍 Market Research",
                    key=f"market_{concept['concept_name']}",
                ):
                    st.success(
                        f"Market research initiated for {concept['concept_name']}"
                    )
            with col3:
                if st.button(
                    f"🧪 Product Testing",
                    key=f"test_{concept['concept_name']}",
                ):
                    st.success(
                        f"Product testing scheduled for {concept['concept_name']}"
                    )
            with col4:
                if st.button(
                    f"🚦 Go/No-Go", key=f"decision_{concept['concept_name']}"
                ):
                    st.success(
                        f"Investment decision process started for {concept['concept_name']}"
                    )

    # Innovation Performance Metrics
    st.subheader("📈 Innovation Performance Dashboard")

    innovation_metrics = {
        "Active Concepts": len(product_concepts),
        "Pipeline Value": "KES 15.6M",
        "Avg Development Time": "7.2 months",
        "Success Rate": "68%",
        "ROI from Innovation": "42%",
        "Time to Market": "8.5 months",
    }

    col1, col2, col3, col4, col5, col6 = st.columns(6)
    metrics_cols = [col1, col2, col3, col4, col5, col6]

    for (metric, value), col in zip(innovation_metrics.items(), metrics_cols):
        with col:
            st.metric(metric, value)

    # Strategic Innovation Recommendations
    st.subheader("🎯 Strategic Innovation Recommendations")

    innovation_strategies = [
        {
            "strategy": "🚀 Accelerated Digital Innovation",
            "focus": "Leverage digital platforms for rapid product testing and launch",
            "key_initiatives": [
                "Implement digital product concept testing",
                "Develop e-commerce first launch strategy",
                "Create digital consumer feedback loops",
                "Build agile product development teams",
            ],
            "timeline": "6-12 months",
            "expected_impact": "Reduce time-to-market by 40%",
        },
        {
            "strategy": "🌍 Local Sourcing & Sustainability",
            "focus": "Develop products using local ingredients with sustainable practices",
            "key_initiatives": [
                "Establish local supplier partnerships",
                "Develop sustainable packaging solutions",
                "Create traceability and transparency systems",
                "Build community engagement programs",
            ],
            "timeline": "12-18 months",
            "expected_impact": "Differentiate brand and capture premium pricing",
        },
        {
            "strategy": "🤝 Strategic Partnerships & Co-creation",
            "focus": "Collaborate with complementary brands and innovators",
            "key_initiatives": [
                "Identify strategic partnership opportunities",
                "Develop co-creation framework",
                "Establish innovation ecosystem partnerships",
                "Create joint venture opportunities",
            ],
            "timeline": "9-15 months",
            "expected_impact": "Access new capabilities and accelerate innovation",
        },
    ]

    for strategy in innovation_strategies:
        with st.expander(
            f"{strategy['strategy']} - Timeline: {strategy['timeline']}"
        ):
            st.write(f"**Strategic Focus:** {strategy['focus']}")
            st.write(f"**Expected Impact:** {strategy['expected_impact']}")

            st.write("**Key Initiatives:**")
            for initiative in strategy["key_initiatives"]:
                st.write(f"✅ {initiative}")

            if st.button(
                f"Implement {strategy['strategy'].split(' ')[1]} Strategy",
                key=f"strat_{strategy['strategy']}",
            ):
                st.success(
                    f"Implementation started for {strategy['strategy']}"
                )


if __name__ == "__main__":
    render()
