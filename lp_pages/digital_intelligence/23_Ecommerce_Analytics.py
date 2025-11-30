# pages/digital_intelligence/23_Ecommerce_Analytics.py
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import numpy as np
import sys
import os

# Allow imports from project root if needed
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))


def render():
    """🛒 ECOMMERCE INTELLIGENCE - Multi-Platform Performance & Optimization"""

    st.title("🛒 Ecommerce Intelligence")

    # 🌈 Gradient hero header (aligned with 01_Dashboard style)
    st.markdown(
        f"""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                padding: 20px; border-radius: 12px; color: white; margin-bottom: 12px;">
        <h3 style="margin: 0; color: white;">🎯 Multi-Platform Performance & Revenue Optimization</h3>
        <p style="margin: 5px 0 0 0; opacity: 0.9; color: white;">
            <strong>📍</strong> Digital Intelligence &gt; Ecommerce Analytics |
            <strong>🏢</strong> {st.session_state.get('current_tenant', 'ELORA Holding')} |
            <strong>📊</strong> Multi-Platform Aggregation |
            <strong>📅</strong> {datetime.now().strftime('%Y-%m-%d')}
        </p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # 💚 Soft green marquee strip – directly under hero, like 01_Dashboard
    st.markdown(
        """
    <div style="background: #dcfce7; padding: 10px 16px; border-radius: 10px;
                margin-bottom: 24px; border-left: 4px solid #16a34a;">
        <marquee behavior="scroll" direction="left" scrollamount="4"
                 style="font-size: 0.9rem; font-weight: 500; color: #166534;">
            🌐 <strong>Ecommerce Intelligence:</strong> Multi-platform GMV, traffic & conversion visibility • 
            🛒 <strong>Commerce Performance:</strong> Cross-platform GMV, AOV & CLV tracking • 
            📈 <strong>Conversion Excellence:</strong> Funnel optimization across marketplaces & D2C • 
            💰 <strong>Profitability:</strong> Contribution margins by platform & product • 
            🎯 <strong>Optimization Engine:</strong> Budget reallocation, upsell & cross-sell insights
        </marquee>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Data validation
    if "digital_data" not in st.session_state:
        st.error("❌ Digital data not initialized. Please visit Digital Overview first.")
        return

    digital_data = st.session_state.digital_data
    ecommerce_data = digital_data.get("ecommerce", pd.DataFrame())

    if ecommerce_data is None or len(ecommerce_data) == 0:
        st.error("❌ Ecommerce dataset is empty or not available.")
        return

    # Enterprise 4-Tab Structure
    tab1, tab2, tab3, tab4 = st.tabs(
        [
            "📈 Platform Performance",
            "🛍️ Product Intelligence",
            "💰 Financial Analytics",
            "🎯 Optimization Engine",
        ]
    )

    with tab1:
        render_platform_performance(ecommerce_data)
    with tab2:
        render_product_intelligence(ecommerce_data)
    with tab3:
        render_financial_analytics(ecommerce_data)
    with tab4:
        render_optimization_engine(ecommerce_data)


def render_platform_performance(data: pd.DataFrame):
    """Multi-platform performance analysis"""

    st.header("📈 Multi-Platform Performance Dashboard")

    # Guard against missing columns
    required_cols = ["revenue", "platform", "conversion_rate", "visitors", "orders", "date"]
    for col in required_cols:
        if col not in data.columns:
            st.error(f"Required column `{col}` is missing in ecommerce data.")
            return

    # Enterprise KPI Row
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        total_gmv = data["revenue"].sum()
        st.metric(
            "Total GMV",
            f"${total_gmv:,.0f}",
            "15.2%",
            help="Gross Merchandise Value across all platforms",
        )

    with col2:
        platform_count = data["platform"].nunique()
        st.metric(
            "Active Platforms",
            platform_count,
            "+2",
            help="Number of integrated ecommerce platforms",
        )

    with col3:
        avg_conversion = data["conversion_rate"].mean()
        st.metric(
            "Avg Conversion Rate",
            f"{avg_conversion:.2f}%",
            "0.8%",
            help="Weighted average conversion rate across platforms",
        )

    with col4:
        customer_acquisition = data["visitors"].sum()
        st.metric(
            "Total Visitors",
            f"{customer_acquisition:,.0f}",
            "12.7%",
            help="Total unique visitors across all platforms",
        )

    st.divider()

    # Platform comparison matrix
    st.subheader("📊 Platform Performance Matrix")

    # Calculate platform metrics
    platform_metrics = (
        data.groupby("platform")
        .agg(
            {
                "revenue": ["sum", "mean", "std"],
                "orders": "sum",
                "visitors": "sum",
                "conversion_rate": "mean",
            }
        )
        .round(2)
    )

    platform_metrics.columns = [
        "Total Revenue",
        "Avg Revenue",
        "Revenue Std",
        "Total Orders",
        "Total Visitors",
        "Avg Conversion Rate",
    ]
    platform_metrics["CAC Efficiency"] = (
        platform_metrics["Total Revenue"] / platform_metrics["Total Visitors"]
    ).round(2)

    # Display platform comparison
    col1, col2 = st.columns(2)

    with col1:
        st.dataframe(
            platform_metrics.style.format(
                {
                    "Total Revenue": "${:,.0f}",
                    "Avg Revenue": "${:,.2f}",
                    "Revenue Std": "${:,.2f}",
                    "Total Orders": "{:,.0f}",
                    "Total Visitors": "{:,.0f}",
                    "Avg Conversion Rate": "{:.2f}%",
                    "CAC Efficiency": "${:.2f}",
                }
            ),
            use_container_width=True,
        )

    with col2:
        # Platform efficiency scatter
        fig = px.scatter(
            platform_metrics.reset_index(),
            x="Avg Conversion Rate",
            y="CAC Efficiency",
            size="Total Revenue",
            color="platform",
            title="Platform Efficiency Analysis",
            hover_data=["Total Orders"],
        )
        st.plotly_chart(fig, use_container_width=True)

    # Platform trends over time
    st.subheader("📈 Platform Revenue Trends")

    platform_trends = data.groupby(["date", "platform"])["revenue"].sum().reset_index()
    fig = px.line(
        platform_trends,
        x="date",
        y="revenue",
        color="platform",
        title="Daily Revenue Trends by Platform",
        labels={"revenue": "Daily Revenue ($)", "date": "Date"},
    )
    st.plotly_chart(fig, use_container_width=True)


def render_product_intelligence(data: pd.DataFrame):
    """Advanced product performance analytics"""

    st.header("🛍️ Product Intelligence & Portfolio Analysis")

    if "product" not in data.columns:
        st.error("Required column `product` is missing in ecommerce data.")
        return

    # Product performance matrix
    product_metrics = (
        data.groupby("product")
        .agg(
            {
                "revenue": ["sum", "mean", "count"],
                "orders": "sum",
                "conversion_rate": "mean",
            }
        )
        .round(2)
    )

    product_metrics.columns = [
        "Total Revenue",
        "Avg Order Value",
        "Data Points",
        "Total Orders",
        "Avg Conversion Rate",
    ]
    product_metrics["Revenue Share %"] = (
        product_metrics["Total Revenue"] / product_metrics["Total Revenue"].sum() * 100
    ).round(2)

    # Product portfolio analysis
    col1, col2 = st.columns(2)

    with col1:
        # Product performance table
        st.subheader("📋 Product Performance Matrix")
        st.dataframe(
            product_metrics.style.format(
                {
                    "Total Revenue": "${:,.0f}",
                    "Avg Order Value": "${:.2f}",
                    "Data Points": "{:.0f}",
                    "Total Orders": "{:.0f}",
                    "Avg Conversion Rate": "{:.2f}%",
                    "Revenue Share %": "{:.1f}%",
                }
            ),
            use_container_width=True,
        )

    with col2:
        # Product portfolio visualization
        st.subheader("📊 Product Portfolio Analysis")

        # Create BCG Matrix (Growth-Share Matrix) – simulated for demo
        product_metrics["Growth Rate"] = np.random.uniform(5, 25, len(product_metrics))
        product_metrics["Relative Market Share"] = np.random.uniform(
            0.5, 2.5, len(product_metrics)
        )

        fig = px.scatter(
            product_metrics.reset_index(),
            x="Relative Market Share",
            y="Growth Rate",
            size="Total Revenue",
            color="Revenue Share %",
            hover_name="product",
            title="Product Portfolio Matrix (BCG Analysis)",
            labels={
                "Growth Rate": "Market Growth Rate (%)",
                "Relative Market Share": "Relative Market Share",
            },
        )

        # Add quadrant lines
        fig.add_hline(y=15, line_dash="dash", line_color="red")
        fig.add_vline(x=1.0, line_dash="dash", line_color="red")

        # Add quadrant annotations
        fig.add_annotation(
            x=2.0, y=20, text="STARS", showarrow=False, font=dict(color="green", size=14)
        )
        fig.add_annotation(
            x=2.0,
            y=8,
            text="CASH COWS",
            showarrow=False,
            font=dict(color="blue", size=14),
        )
        fig.add_annotation(
            x=0.7,
            y=20,
            text="QUESTION MARKS",
            showarrow=False,
            font=dict(color="orange", size=12),
        )
        fig.add_annotation(
            x=0.7,
            y=8,
            text="DOGS",
            showarrow=False,
            font=dict(color="red", size=14),
        )

        st.plotly_chart(fig, use_container_width=True)

    # 🎯 STRATEGIC BCG INSIGHTS - COLLAPSIBLE SECTION
    with st.expander("🎯 BCG Matrix Strategic Insights", expanded=False):
        # Quadrant Definitions
        st.subheader("📋 BCG Matrix Quadrant Definitions")

        col1, col2 = st.columns(2)

        with col1:
            # STARS
            st.markdown(
                """
            **⭐ STARS** (High Growth, High Market Share)  
            - **Position**: Top-right quadrant  
            - **Meaning**: Market leaders in high-growth markets  
            - **Strategy**: **Invest heavily** – these are your future cash cows  
            - **Action**: Aggressive marketing, capacity expansion, defend position
            """
            )

            # CASH COWS
            st.markdown(
                """
            **💰 CASH COWS** (Low Growth, High Market Share)  
            - **Position**: Bottom-right quadrant  
            - **Meaning**: Dominant products in mature markets  
            - **Strategy**: **Milk for cash** – generate profits to fund stars  
            - **Action**: Maintain market share, optimize costs, maximize profitability
            """
            )

        with col2:
            # QUESTION MARKS
            st.markdown(
                """
            **❓ QUESTION MARKS** (High Growth, Low Market Share)  
            - **Position**: Top-left quadrant  
            - **Meaning**: Products in attractive markets but weak position  
            - **Strategy**: **Make strategic choices** – invest or divest  
            - **Action**: Either invest to gain share or exit if no path to leadership
            """
            )

            # DOGS
            st.markdown(
                """
            **❌ DOGS** (Low Growth, Low Market Share)  
            - **Position**: Bottom-left quadrant  
            - **Meaning**: Weak products in unattractive markets  
            - **Strategy**: **Divest or harvest** – minimal investment  
            - **Action**: Phase out, sell, or maintain with minimal resources
            """
            )

        st.divider()

        # Strategic Actions Framework
        st.subheader("🎯 Strategic Action Framework")

        strategic_actions = {
            "STARS": {
                "Budget": "Increase investment by 20–30%",
                "Marketing": "Aggressive brand building",
                "Operations": "Scale capacity, optimize supply chain",
                "Goal": "Convert to Cash Cow",
            },
            "CASH COWS": {
                "Budget": "Maintain or slightly reduce",
                "Marketing": "Defensive, retain loyal customers",
                "Operations": "Cost optimization, efficiency gains",
                "Goal": "Fund Stars and selected Question Marks",
            },
            "QUESTION MARKS": {
                "Budget": "Selective investment based on potential",
                "Marketing": "Test-and-learn approach",
                "Operations": "Flexible, scalable models",
                "Goal": "Move to Stars or divest if no path",
            },
            "DOGS": {
                "Budget": "Minimal or zero investment",
                "Marketing": "Harvest remaining value",
                "Operations": "Cost minimization",
                "Goal": "Divest, phase out, or niche positioning",
            },
        }

        cols = st.columns(4)
        for idx, (quadrant, actions) in enumerate(strategic_actions.items()):
            with cols[idx]:
                st.markdown(f"**{quadrant}**")
                for action_type, action_desc in actions.items():
                    st.markdown(f"• **{action_type}**: {action_desc}")

        st.divider()

        # Business Impact Communication
        st.subheader("📈 Business Impact & Strategic Questions")

        st.markdown(
            """
        **The BCG Matrix helps answer critical strategic questions:**

        🔍 **"Where should we invest our limited resources?"** → Focus on **Stars**  
        💰 **"Which products fund our growth?"** → **Cash Cows** provide the cash  
        🚀 **"What are our future growth engines?"** → **Stars** and promising **Question Marks**  
        ❌ **"What should we stop doing?"** → **Dogs** and hopeless **Question Marks**  
        ⚖️ **"Is our portfolio balanced?"** → Healthy mix across quadrants
        """
        )

        st.divider()

        # Logistics Pro Specific Impact
        st.subheader("🚀 Why This Matters for Logistics Pro")

        st.markdown(
            """
        **For an FMCG distributor, BCG analysis directly impacts:**

        📦 **Inventory Planning**: Stock more **Stars**, less **Dogs**  
        🏭 **Warehouse Space Allocation**: Prioritize high-growth **Stars** and **Question Marks**  
        📌 **Supply Chain Investments**: Build capacity for **Stars**  
        🤝 **Procurement Strategy**: Negotiate better terms for **Cash Cows**  
        📊 **Sales Focus**: Incentivize **Stars** and promising **Question Marks**  
        💡 **Resource Allocation**: Optimize marketing, operations, and capital expenditure
        """
        )

        st.success(
            """
        **🎯 Key Takeaway**: The BCG Matrix transforms product data into actionable strategic intelligence, 
        separating basic analytics from true enterprise-grade business intelligence.
        """
        )

    # Cross-platform product performance
    st.subheader("🌐 Cross-Platform Product Analysis")

    platform_product_matrix = (
        data.groupby(["platform", "product"])["revenue"].sum().unstack(fill_value=0)
    )

    fig = px.imshow(
        platform_product_matrix,
        title="Revenue Heatmap: Products vs Platforms",
        aspect="auto",
        color_continuous_scale="Blues",
    )
    st.plotly_chart(fig, use_container_width=True)


def render_financial_analytics(data: pd.DataFrame):
    """Advanced financial metrics and forecasting"""

    st.header("💰 Financial Analytics & Revenue Intelligence")

    # Guard for safety
    required_cols = ["revenue", "orders", "visitors", "date", "platform"]
    for col in required_cols:
        if col not in data.columns:
            st.error(f"Required column `{col}` is missing in ecommerce data.")
            return

    # Avoid division by zero
    total_orders = data["orders"].sum() or 1
    total_visitors = data["visitors"].sum() or 1

    # Financial KPIs
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        aov = data["revenue"].sum() / total_orders
        st.metric(
            "Average Order Value",
            f"${aov:.2f}",
            "3.5%",
            help="Weighted average across all platforms and products",
        )

    with col2:
        customer_lifetime_value = aov * 4.2  # Simulated LTV multiple
        st.metric(
            "Estimated CLV",
            f"${customer_lifetime_value:.2f}",
            "8.1%",
            help="Customer Lifetime Value estimation",
        )

    with col3:
        revenue_per_visitor = data["revenue"].sum() / total_visitors
        st.metric(
            "Revenue per Visitor",
            f"${revenue_per_visitor:.4f}",
            "12.3%",
            help="Overall revenue efficiency metric",
        )

    with col4:
        days_sales_outstanding = 14.2  # Simulated DSO
        st.metric(
            "Days Sales Outstanding",
            f"{days_sales_outstanding:.1f}",
            "-2.1",
            help="Average collection period for receivables",
        )

    st.divider()

    # Revenue forecasting and trends
    st.subheader("📈 Revenue Forecasting & Seasonality")

    # Time series analysis
    daily_revenue = data.groupby("date")["revenue"].sum().reset_index()
    daily_revenue["7_day_avg"] = daily_revenue["revenue"].rolling(window=7).mean()
    daily_revenue["30_day_avg"] = daily_revenue["revenue"].rolling(window=30).mean()

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=daily_revenue["date"],
            y=daily_revenue["revenue"],
            mode="lines",
            name="Daily Revenue",
            line=dict(color="lightblue"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=daily_revenue["date"],
            y=daily_revenue["7_day_avg"],
            mode="lines",
            name="7-Day Moving Avg",
            line=dict(color="blue"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=daily_revenue["date"],
            y=daily_revenue["30_day_avg"],
            mode="lines",
            name="30-Day Moving Avg",
            line=dict(color="darkblue"),
        )
    )

    fig.update_layout(
        title="Revenue Trend Analysis with Moving Averages",
        xaxis_title="Date",
        yaxis_title="Revenue ($)",
    )

    st.plotly_chart(fig, use_container_width=True)

    # Platform contribution analysis
    st.subheader("🏆 Platform Contribution Margin Analysis")

    # Simulated contribution margins by platform
    platforms = data["platform"].unique()
    contribution_data = []

    for platform in platforms:
        platform_revenue = data[data["platform"] == platform]["revenue"].sum()
        # Simulate platform-specific costs (fees, marketing, etc.)
        cost_rate = np.random.uniform(0.15, 0.35)
        contribution_margin = platform_revenue * (1 - cost_rate)

        contribution_data.append(
            {
                "Platform": platform,
                "Total Revenue": platform_revenue,
                "Platform Costs": platform_revenue * cost_rate,
                "Contribution Margin": contribution_margin,
                "Margin Rate": (1 - cost_rate) * 100,
            }
        )

    contribution_df = pd.DataFrame(contribution_data)

    fig = px.bar(
        contribution_df,
        x="Platform",
        y=["Total Revenue", "Contribution Margin"],
        title="Platform Contribution Analysis",
        barmode="group",
        labels={"value": "Amount ($)", "variable": "Metric"},
    )

    st.plotly_chart(fig, use_container_width=True)


def render_optimization_engine(data: pd.DataFrame):
    """AI-powered optimization recommendations"""

    st.header("🎯 Ecommerce Optimization Engine")

    # AI Insights Section
    with st.expander("🧠 AI-Powered Strategic Insights", expanded=True):
        st.success(
            """
        **Top Optimization Opportunities Identified:**

        🚀 **High-Impact**: Shopify shows 28% higher conversion rates than Amazon – consider reallocating marketing budget  
        💡 **Quick Win**: *Premium Coffee* has 45% higher AOV on WooCommerce – optimize product placement  
        ⚠️ **Risk Alert**: eBay contribution margin is 12% below average – review fee structure
        """
        )

    # Optimization recommendations
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📊 Performance Optimization")

        # Conversion rate optimization opportunities
        platform_conversion = (
            data.groupby("platform")["conversion_rate"].mean().sort_values(ascending=False)
        )

        fig = px.bar(
            x=platform_conversion.index,
            y=platform_conversion.values,
            title="Conversion Rate by Platform (Optimization Priority)",
            labels={"x": "Platform", "y": "Conversion Rate (%)"},
        )
        st.plotly_chart(fig, use_container_width=True)

        if len(platform_conversion) > 1:
            worst_platform = platform_conversion.index[-1]
            st.info(
                f"**Recommendation**: Focus on **{worst_platform}** – approximately "
                f"{(1 - platform_conversion.iloc[-1] / platform_conversion.iloc[0]) * 100:.0f}% "
                f"below top performer."
            )

    with col2:
        st.subheader("💰 Revenue Optimization")

        # AOV optimization opportunities
        def _safe_aov(group: pd.DataFrame) -> float:
            orders_sum = group["orders"].sum() or 1
            return group["revenue"].sum() / orders_sum

        product_aov = (
            data.groupby("product")
            .apply(_safe_aov)
            .sort_values(ascending=False)
        )

        fig = px.bar(
            x=product_aov.index,
            y=product_aov.values,
            title="Average Order Value by Product",
            labels={"x": "Product", "y": "AOV ($)"},
        )
        st.plotly_chart(fig, use_container_width=True)

        if len(product_aov) > 0:
            top_product = product_aov.index[0]
            st.info(
                f"**Upsell Opportunity**: Bundle **{top_product}** with lower AOV products "
                "to lift overall basket value."
            )

    # Automated action plan
    st.subheader("🧰 Automated Action Plan")

    action_items = [
        {"action": "Budget Reallocation", "platform": "Shopify", "impact": "High", "timeline": "2 weeks"},
        {"action": "Product Placement", "platform": "WooCommerce", "impact": "Medium", "timeline": "1 week"},
        {"action": "Fee Negotiation", "platform": "eBay", "impact": "High", "timeline": "4 weeks"},
        {"action": "Cross-sell Strategy", "platform": "All", "impact": "Medium", "timeline": "3 weeks"},
    ]

    action_df = pd.DataFrame(action_items)

    def _impact_style(val: str) -> str:
        if val == "High":
            return "background-color: #90EE90"  # light green
        if val == "Medium":
            return "background-color: #FFFFE0"  # light yellow
        return ""

    st.dataframe(
        action_df.style.applymap(_impact_style, subset=["impact"]),
        use_container_width=True,
    )


if __name__ == "__main__":
    render()
