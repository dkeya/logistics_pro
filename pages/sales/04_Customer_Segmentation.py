# logistics_pro/pages/04_Customer_Segmentation.py
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import numpy as np


def safe_get_column(df, column, default_value=None):
    """Safely get column with fallback."""
    if column in df.columns:
        return df[column]
    if default_value is None:
        return pd.Series([f"Default_{i}" for i in range(len(df))])
    return pd.Series([default_value] * len(df))


def compute_churn_risk_label(days_since_last: int):
    """
    Simple heuristic churn risk label + score (0–1).
    Shared across Behavior Analytics + Customer 360.
    """
    if days_since_last <= 30:
        return "Low", 0.2
    elif days_since_last <= 60:
        return "Medium", 0.5
    elif days_since_last <= 90:
        return "High", 0.8
    else:
        return "Critical", 1.0


def build_rfm_segments(sales_data: pd.DataFrame) -> pd.DataFrame:
    """
    Shared RFM builder so RFM Segmentation + Growth Strategies use
    the SAME definitions.
    """
    if sales_data.empty or "date" not in sales_data.columns:
        return pd.DataFrame()

    current_date = sales_data["date"].max()

    rfm_data = (
        sales_data.groupby("customer_id")
        .agg(
            recency_days=("date", lambda x: (current_date - x.max()).days),
            frequency_count=("customer_id", "count"),
            monetary_value=("revenue", "sum"),
        )
        .reset_index()
    )

    if rfm_data.empty:
        return rfm_data

    # Thresholds
    monetary_threshold_80 = rfm_data["monetary_value"].quantile(0.8)
    monetary_threshold_60 = rfm_data["monetary_value"].quantile(0.6)
    frequency_threshold_80 = rfm_data["frequency_count"].quantile(0.8)

    conditions = [
        (rfm_data["recency_days"] <= 30)
        & (rfm_data["monetary_value"] > monetary_threshold_80)
        & (rfm_data["frequency_count"] > frequency_threshold_80),
        (rfm_data["recency_days"] <= 30)
        & (rfm_data["monetary_value"] > monetary_threshold_60),
        (rfm_data["recency_days"] <= 60)
        & (rfm_data["monetary_value"] > monetary_threshold_60),
        (rfm_data["recency_days"] > 90),
    ]

    choices = ["VIP Champions", "Loyal High-Value", "Regular Loyal", "At Risk"]
    rfm_data["segment"] = np.select(conditions, choices, default="Regular")

    return rfm_data


def render():
    """👥 CUSTOMER SEGMENTATION - Enhanced Customer Intelligence"""

    st.title("👥 Customer Segmentation")

    # 🌈 Gradient hero header (aligned with Executive Cockpit style)
    st.markdown(
        f"""
    <div style="
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px 24px;
        border-radius: 12px;
        color: white;
        margin-bottom: 12px;
    ">
        <h3 style="margin: 0; color: white;">
            🎯 Customer Intelligence & Segmentation
        </h3>
        <p style="margin: 6px 0 0 0; opacity: 0.95; color: white; font-size: 0.9rem;">
            <strong>📍</strong> Sales Intelligence &gt; Customer Segmentation |
            <strong>🏢</strong> {st.session_state.get('current_tenant', 'ELORA Holding')} |
            <strong>📅</strong> {datetime.now().strftime('%Y-%m-%d')}
        </p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # 💚 Soft green marquee strip – same styling family as 01_Dashboard
    st.markdown(
        """
    <div style="background: #dcfce7; padding: 10px 16px; border-radius: 10px;
                margin-bottom: 24px; border-left: 4px solid #16a34a;">
        <marquee behavior="scroll" direction="left" scrollamount="4"
                 style="font-size: 0.9rem; font-weight: 500; color: #166534;">
            👥 Customer Intelligence Radar: High-value VIPs • At-Risk and Lost Customers •
            CLV and RFM Segments • Portfolio Churn Risk • Targeted Retention & Growth Playbooks
        </marquee>
    </div>
    """,
        unsafe_allow_html=True,
    )

    if "analytics" not in st.session_state:
        st.error("❌ Please visit the main dashboard first to initialize data")
        return

    analytics = st.session_state.analytics

    try:
        # Get sales data safely
        sales_data = analytics.sales_data.copy()

        # 🔐 Ensure date is a proper pandas datetime for all .dt operations
        if "date" in sales_data.columns:
            sales_data["date"] = pd.to_datetime(sales_data["date"], errors="coerce")
            # Drop rows where date could not be parsed
            sales_data = sales_data.dropna(subset=["date"])
        else:
            raise ValueError("Missing required 'date' column in sales_data")

        # Ensure key columns exist so downstream logic never hard-crashes
        if "customer_id" not in sales_data.columns:
            sales_data["customer_id"] = [f"CUST_{i+1:03d}" for i in range(len(sales_data))]

        if "quantity" not in sales_data.columns:
            sales_data["quantity"] = 1

        if "sku_id" not in sales_data.columns:
            sales_data["sku_id"] = [f"SKU_{i+1:03d}" for i in range(len(sales_data))]

        # Safely create revenue (handles missing unit_price)
        sales_data["revenue"] = sales_data["quantity"] * safe_get_column(
            sales_data, "unit_price", 100
        )

        # ❌ Old banner intentionally removed to keep cockpit clean
        # st.success("✅ Customer Segmentation Intelligence loaded successfully!")

    except Exception as e:
        st.error(f"❌ Error loading data: {e}")
        return

    # Enhanced Tab Structure with Strategic Frameworks
    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        [
            "📊 Customer Overview",
            "🎯 RFM Segmentation",
            "📈 Behavior Analytics",
            "💡 Customer 360°",
            "🚀 Growth Strategies",
        ]
    )

    with tab1:
        render_customer_overview(sales_data)

    with tab2:
        render_rfm_analysis(sales_data)

    with tab3:
        render_behavior_insights(sales_data)

    with tab4:
        render_customer_360(sales_data)

    with tab5:
        render_growth_strategies(sales_data)


def render_customer_overview(sales_data: pd.DataFrame):
    """Render enhanced customer overview with strategic insights."""
    st.header("📊 Customer Portfolio Overview")

    # Calculate basic metrics
    total_customers = sales_data["customer_id"].nunique()
    total_revenue = sales_data["revenue"].sum()
    total_orders = len(sales_data)
    avg_order_value = total_revenue / total_orders if total_orders > 0 else 0

    # Enhanced KPI Cards with Strategic Context
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Customers", total_customers)
        st.caption("👥 Customer Base Size")

    with col2:
        st.metric("Total Revenue", f"KES {total_revenue:,.0f}")
        st.caption("💰 Total Customer Value")

    with col3:
        st.metric("Total Orders", total_orders)
        st.caption("📦 Transaction Volume")

    with col4:
        st.metric("Avg Order Value", f"KES {avg_order_value:,.0f}")
        st.caption("🎯 Customer Spending Power")

    # Customer Value Distribution Analysis
    st.subheader("💰 Customer Value Distribution")

    customer_revenue = (
        sales_data.groupby("customer_id")
        .agg(
            total_revenue=("revenue", "sum"),
            total_quantity=("quantity", "sum"),
            order_count=("date", "count"),
        )
        .reset_index()
    )

    col1, col2 = st.columns(2)

    with col1:
        # Top customers chart
        top_customers = customer_revenue.nlargest(10, "total_revenue")

        fig = px.bar(
            top_customers,
            x="customer_id",
            y="total_revenue",
            title="🏆 Top 10 Customers by Revenue",
            labels={"customer_id": "Customer ID", "total_revenue": "Revenue (KES)"},
            color="total_revenue",
            color_continuous_scale="Viridis",
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

        # Strategic insight
        top_10_revenue = top_customers["total_revenue"].sum()
        top_10_percentage = (
            top_10_revenue / total_revenue * 100 if total_revenue > 0 else 0
        )
        st.info(
            f"**Strategic Insight:** Top 10 customers contribute "
            f"{top_10_percentage:.1f}% of total revenue"
        )

    with col2:
        # Customer revenue distribution with segmentation
        fig = px.histogram(
            customer_revenue,
            x="total_revenue",
            nbins=20,
            title="📈 Customer Revenue Distribution",
            labels={"total_revenue": "Revenue (KES)"},
            color_discrete_sequence=["#00cc96"],
        )

        # Add segmentation lines
        high_value_threshold = customer_revenue["total_revenue"].quantile(0.8)
        medium_value_threshold = customer_revenue["total_revenue"].quantile(0.5)

        fig.add_vline(
            x=high_value_threshold,
            line_dash="dash",
            line_color="red",
            annotation_text="High Value",
            annotation_position="top",
        )
        fig.add_vline(
            x=medium_value_threshold,
            line_dash="dash",
            line_color="orange",
            annotation_text="Medium Value",
            annotation_position="top",
        )

        st.plotly_chart(fig, use_container_width=True)

    # Customer Lifetime Value Analysis
    st.subheader("🎯 Customer Lifetime Value (CLV) Insights")

    customer_ltv = (
        sales_data.groupby("customer_id")
        .agg(
            total_revenue=("revenue", "sum"),
            first_purchase=("date", "min"),
            last_purchase=("date", "max"),
            order_count=("date", "count"),
        )
        .reset_index()
    )

    customer_ltv["customer_lifetime"] = (
        customer_ltv["last_purchase"] - customer_ltv["first_purchase"]
    ).dt.days

    # Avoid division by zero
    customer_ltv["customer_lifetime"] = customer_ltv["customer_lifetime"].replace(0, 1)

    customer_ltv["avg_order_value"] = (
        customer_ltv["total_revenue"] / customer_ltv["order_count"]
    )
    customer_ltv["purchase_frequency"] = (
        customer_ltv["order_count"] / (customer_ltv["customer_lifetime"] / 30)
    )

    # CLV segments based on revenue thresholds
    high_value_threshold = customer_ltv["total_revenue"].quantile(0.8)
    medium_value_threshold = customer_ltv["total_revenue"].quantile(0.5)

    high_clv = customer_ltv[customer_ltv["total_revenue"] > high_value_threshold]
    medium_clv = customer_ltv[
        (customer_ltv["total_revenue"] > medium_value_threshold)
        & (customer_ltv["total_revenue"] <= high_value_threshold)
    ]
    low_clv = customer_ltv[customer_ltv["total_revenue"] <= medium_value_threshold]

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("High CLV Customers", len(high_clv))
        if not high_clv.empty:
            st.caption(f"KES {high_clv['total_revenue'].mean():,.0f} avg value")
        else:
            st.caption("No data")

    with col2:
        st.metric("Medium CLV Customers", len(medium_clv))
        if not medium_clv.empty:
            st.caption(f"KES {medium_clv['total_revenue'].mean():,.0f} avg value")
        else:
            st.caption("No data")

    with col3:
        st.metric("Low CLV Customers", len(low_clv))
        if not low_clv.empty:
            st.caption(f"KES {low_clv['total_revenue'].mean():,.0f} avg value")
        else:
            st.caption("No data")


def render_rfm_analysis(sales_data: pd.DataFrame):
    """Render enhanced RFM analysis with strategic frameworks."""
    st.header("🎯 Customer Value Segmentation (RFM)")

    try:
        rfm_data = build_rfm_segments(sales_data)

        if rfm_data.empty:
            st.info("Not enough data to compute RFM segments.")
            return

        col1, col2 = st.columns(2)

        with col1:
            # Segment distribution
            segment_counts = rfm_data["segment"].value_counts()
            fig = px.pie(
                values=segment_counts.values,
                names=segment_counts.index,
                title="🎯 Customer Segments Distribution",
                color=segment_counts.index,
                color_discrete_map={
                    "VIP Champions": "#FFD700",
                    "Loyal High-Value": "#00CC96",
                    "Regular Loyal": "#1F77B4",
                    "At Risk": "#EF553B",
                    "Regular": "#636EFA",
                },
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Scatter RFM
            fig = px.scatter(
                rfm_data,
                x="frequency_count",
                y="monetary_value",
                color="segment",
                size="recency_days",
                title="📊 RFM Analysis: Frequency vs Monetary Value",
                hover_data=["customer_id"],
                color_discrete_map={
                    "VIP Champions": "#FFD700",
                    "Loyal High-Value": "#00CC96",
                    "Regular Loyal": "#1F77B4",
                    "At Risk": "#EF553B",
                    "Regular": "#636EFA",
                },
            )
            fig.update_layout(
                xaxis_title="Purchase Frequency",
                yaxis_title="Monetary Value (KES)",
            )
            st.plotly_chart(fig, use_container_width=True)

        # Segment metrics
        st.subheader("💡 Strategic Segment Analysis")

        segment_metrics = (
            rfm_data.groupby("segment")
            .agg(
                customer_count=("customer_id", "count"),
                avg_recency=("recency_days", "mean"),
                avg_frequency=("frequency_count", "mean"),
                avg_monetary=("monetary_value", "mean"),
            )
            .reset_index()
            .rename(
                columns={
                    "segment": "Segment",
                    "customer_count": "Customer Count",
                    "avg_recency": "Avg Recency (days)",
                    "avg_frequency": "Avg Frequency",
                    "avg_monetary": "Avg Monetary (KES)",
                }
            )
            .round(2)
        )

        st.dataframe(segment_metrics, use_container_width=True)

        # Strategies
        st.subheader("🚀 Segment-Specific Growth Strategies")

        strategies = {
            "VIP Champions": {
                "strategy": "🏆 Elite Retention & Advocacy",
                "investment": "High",
                "roi": "Very High",
                "actions": [
                    "Dedicated account management",
                    "Exclusive VIP loyalty rewards",
                    "Early access to new products",
                    "Personalized premium support",
                ],
            },
            "Loyal High-Value": {
                "strategy": "💎 Strategic Upsell & Expansion",
                "investment": "Medium-High",
                "roi": "High",
                "actions": [
                    "Personalized product recommendations",
                    "Premium bundle offers",
                    "Volume-based loyalty tiers",
                    "Cross-sell complementary products",
                ],
            },
            "Regular Loyal": {
                "strategy": "📊 Engagement & Value Growth",
                "investment": "Medium",
                "roi": "Medium-High",
                "actions": [
                    "Targeted email campaigns",
                    "Seasonal promotion participation",
                    "Referral program enrollment",
                    "Educational content delivery",
                ],
            },
            "At Risk": {
                "strategy": "⚠️ Strategic Win-back & Recovery",
                "investment": "Medium",
                "roi": "Variable",
                "actions": [
                    "Personalized win-back offers (20–30% discount)",
                    "Customer satisfaction deep-dive surveys",
                    "Reactivate with new product announcements",
                    "Loyalty program re-engagement",
                ],
            },
            "Regular": {
                "strategy": "🎯 Activation & Loyalty Building",
                "investment": "Low-Medium",
                "roi": "Medium",
                "actions": [
                    "Welcome and onboarding sequences",
                    "Entry-level loyalty program",
                    "Educational content on product value",
                    "Regular engagement communications",
                ],
            },
        }

        for segment, info in strategies.items():
            with st.expander(
                f"{segment} - {info['strategy']} | Investment: {info['investment']} | ROI: {info['roi']}"
            ):
                segment_customers = rfm_data[rfm_data["segment"] == segment]
                if not segment_customers.empty:
                    st.write(f"**Customer Count:** {len(segment_customers)}")
                    st.write(
                        f"**Average Customer Value:** "
                        f"KES {segment_customers['monetary_value'].mean():,.0f}"
                    )
                    st.write(
                        f"**Total Segment Value:** "
                        f"KES {segment_customers['monetary_value'].sum():,.0f}"
                    )

                st.write("**Strategic Actions:**")
                for action in info["actions"]:
                    st.write(f"✅ {action}")

                if st.button(
                    f"🚀 Execute {segment} Strategy", key=f"btn_{segment}"
                ):
                    st.success(f"{segment} strategy execution initiated!")

    except Exception as e:
        st.error(f"❌ RFM analysis error: {e}")
        st.info(
            "This might be due to insufficient customer data. "
            "Try generating more sales data."
        )


def render_behavior_insights(sales_data: pd.DataFrame):
    """Render enhanced customer behavior analytics."""
    st.header("📈 Customer Behavior Analytics")

    try:
        st.subheader("🔍 Advanced Purchase Behavior Analysis")

        customer_behavior = (
            sales_data.groupby("customer_id")
            .agg(
                first_purchase=("date", "min"),
                last_purchase=("date", "max"),
                order_count=("date", "count"),
                total_revenue=("revenue", "sum"),
            )
            .reset_index()
        )

        customer_behavior["customer_lifetime_days"] = (
            customer_behavior["last_purchase"] - customer_behavior["first_purchase"]
        ).dt.days
        customer_behavior["customer_lifetime_days"] = customer_behavior[
            "customer_lifetime_days"
        ].replace(0, 1)

        customer_behavior["purchase_frequency"] = customer_behavior[
            "order_count"
        ] / (customer_behavior["customer_lifetime_days"] / 30)

        col1, col2 = st.columns(2)

        with col1:
            fig = px.histogram(
                customer_behavior,
                x="purchase_frequency",
                nbins=20,
                title="📊 Purchase Frequency Distribution",
                labels={"purchase_frequency": "Orders per Month"},
                color_discrete_sequence=["#1f77b4"],
            )

            high_freq = customer_behavior["purchase_frequency"].quantile(0.8)
            fig.add_vline(
                x=high_freq,
                line_dash="dash",
                line_color="green",
                annotation_text="High Frequency",
                annotation_position="top",
            )

            st.plotly_chart(fig, use_container_width=True)

            high_freq_count = len(
                customer_behavior[
                    customer_behavior["purchase_frequency"] > high_freq
                ]
            )
            st.info(
                f"**Insight:** {high_freq_count} customers are high-frequency purchasers"
            )

        with col2:
            fig = px.scatter(
                customer_behavior,
                x="customer_lifetime_days",
                y="total_revenue",
                size="order_count",
                color="purchase_frequency",
                title="🎯 Customer Value Matrix: Lifetime vs Revenue",
                hover_data=["customer_id"],
                color_continuous_scale="Viridis",
                labels={
                    "customer_lifetime_days": "Customer Lifetime (Days)",
                    "total_revenue": "Total Revenue (KES)",
                    "purchase_frequency": "Purchase Frequency",
                },
            )

            lifetime_median = customer_behavior["customer_lifetime_days"].median()
            revenue_median = customer_behavior["total_revenue"].median()

            fig.add_hline(y=revenue_median, line_dash="dash", line_color="red")
            fig.add_vline(x=lifetime_median, line_dash="dash", line_color="red")

            st.plotly_chart(fig, use_container_width=True)

        # Retention analytics
        st.subheader("📊 Advanced Retention Analytics")

        current_date = sales_data["date"].max()
        days_since_last = (current_date - customer_behavior["last_purchase"]).dt.days

        active_customers = len(days_since_last[days_since_last <= 30])
        warm_customers = len(
            days_since_last[(days_since_last > 30) & (days_since_last <= 60)]
        )
        at_risk_customers = len(
            days_since_last[(days_since_last > 60) & (days_since_last <= 90)]
        )
        lost_customers = len(days_since_last[days_since_last > 90])
        total_analyzed = len(days_since_last)

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Active Customers (30d)", active_customers, delta="Core Revenue")
            st.caption("High engagement")

        with col2:
            st.metric("Warm Customers", warm_customers, delta="Needs attention")
            st.caption("Moderate risk")

        with col3:
            st.metric(
                "At Risk Customers",
                at_risk_customers,
                delta="High priority",
                delta_color="inverse",
            )
            st.caption("60–90 days inactive")

        with col4:
            retention_rate = (
                active_customers / total_analyzed * 100 if total_analyzed > 0 else 0
            )
            industry_benchmark = 75
            delta_retention = retention_rate - industry_benchmark
            st.metric(
                "Retention Rate",
                f"{retention_rate:.1f}%",
                delta=f"{delta_retention:+.1f}% vs benchmark",
            )
            st.caption(f"Industry: {industry_benchmark}%")

        retention_data = pd.DataFrame(
            {
                "Segment": ["Active", "Warm", "At Risk", "Lost"],
                "Count": [
                    active_customers,
                    warm_customers,
                    at_risk_customers,
                    lost_customers,
                ],
            }
        )

        fig = px.bar(
            retention_data,
            x="Segment",
            y="Count",
            title="📈 Customer Retention Status Dashboard",
            color="Segment",
            color_discrete_map={
                "Active": "green",
                "Warm": "orange",
                "At Risk": "red",
                "Lost": "gray",
            },
        )
        st.plotly_chart(fig, use_container_width=True)

        # 🔥 Portfolio Churn Risk Index (simple heuristic)
        st.subheader("🔥 Portfolio Churn Risk Index")

        if total_analyzed > 0:
            # weights: Active (0.1), Warm (0.4), At Risk (0.7), Lost (1.0)
            churn_raw = (
                active_customers * 0.1
                + warm_customers * 0.4
                + at_risk_customers * 0.7
                + lost_customers * 1.0
            ) / total_analyzed
            churn_index_pct = churn_raw * 100
        else:
            churn_index_pct = 0.0

        st.metric(
            "Churn Risk Index",
            f"{churn_index_pct:.1f} / 100",
            help="0 = very low churn risk, 100 = very high churn risk (heuristic).",
        )
        st.progress(min(1.0, max(0.0, churn_index_pct / 100.0)))

        # Retention playbook
        st.subheader("🎯 Customer Retention Playbook")

        retention_strategies = {
            "Active Customers": {
                "focus": "Retention & Loyalty",
                "actions": [
                    "Implement tiered loyalty rewards",
                    "Provide exclusive early access to new products",
                    "Offer premium customer support",
                    "Create VIP community engagement",
                ],
                "kpi": "Increase lifetime value by 25%",
            },
            "Warm Customers": {
                "focus": "Re-engagement & Value Demonstration",
                "actions": [
                    "Send personalized product recommendations",
                    "Offer limited-time reactivation discounts",
                    "Share customer success stories",
                    "Provide educational content on advanced features",
                ],
                "kpi": "Reactivate 40% within 30 days",
            },
            "At Risk Customers": {
                "focus": "Win-back & Relationship Repair",
                "actions": [
                    "Conduct satisfaction deep-dive surveys",
                    "Offer strategic win-back incentives (25–30% off)",
                    "Assign dedicated recovery specialist",
                    "Create personalized comeback offers",
                ],
                "kpi": "Recover 20% of at-risk customers",
            },
            "Lost Customers": {
                "focus": "Learning & Future Prevention",
                "actions": [
                    "Analyze churn reasons systematically",
                    "Implement exit interview process",
                    "Create cannibalization prevention strategies",
                    "Develop win-back campaign for 90+ day lapsed",
                ],
                "kpi": "Reduce overall churn by 15%",
            },
        }

        for segment, strategy in retention_strategies.items():
            with st.expander(f"{segment} - {strategy['focus']}"):
                st.write(f"**Strategic Focus:** {strategy['focus']}")
                st.write(f"**Target KPI:** {strategy['kpi']}")
                st.write("**Action Plan:**")
                for action in strategy["actions"]:
                    st.write(f"🎯 {action}")

                if st.button(
                    f"Execute {segment} Strategy", key=f"retain_{segment}"
                ):
                    st.success(f"{segment} retention strategy launched!")

    except Exception as e:
        st.error(f"❌ Behavior insights error: {e}")
        st.info(
            "This analysis requires sufficient customer transaction history."
        )


def render_customer_360(sales_data: pd.DataFrame):
    """Render enhanced Customer 360° view with strategic insights."""
    st.header("💡 Customer 360° Intelligence")

    try:
        customer_list = sales_data["customer_id"].unique()

        if len(customer_list) == 0:
            st.warning("No customer data available for 360° analysis")
            return

        col1, col2 = st.columns([2, 1])

        with col1:
            selected_customer = st.selectbox(
                "Select Customer for Deep Analysis", customer_list
            )

        with col2:
            analysis_depth = st.selectbox(
                "Analysis Depth",
                ["Quick Overview", "Standard Analysis", "Deep Dive"],
            )

        if selected_customer:
            customer_data = sales_data[
                sales_data["customer_id"] == selected_customer
            ]

            if customer_data.empty:
                st.warning("No data found for selected customer")
                return

            st.subheader(f"🎯 Customer Intelligence: {selected_customer}")

            col1, col2, col3, col4 = st.columns(4)

            total_spend = customer_data["revenue"].sum()
            order_count = len(customer_data)
            avg_order_value = (
                total_spend / order_count if order_count > 0 else 0
            )

            # ✅ Use pandas Timestamps for date math to avoid .date vs Timestamp issues
            current_date = sales_data["date"].max()  # Timestamp
            last_purchase_date = customer_data["date"].max()  # Timestamp

            if pd.isna(current_date) or pd.isna(last_purchase_date):
                days_since_last = 999  # fallback
            else:
                days_since_last = (current_date - last_purchase_date).days

            churn_label, churn_score = compute_churn_risk_label(days_since_last)

            with col1:
                st.metric("Total Customer Value", f"KES {total_spend:,.0f}")
                st.caption("Lifetime revenue")

            with col2:
                st.metric("Order Count", order_count)
                st.caption("Transaction volume")

            with col3:
                st.metric("Avg Order Value", f"KES {avg_order_value:,.0f}")
                st.caption("Spending power")

            with col4:
                if days_since_last <= 30:
                    status = "Active"
                elif days_since_last <= 60:
                    status = "Warm"
                else:
                    status = "At Risk"

                st.metric("Engagement Status", status)
                st.caption(
                    f"{days_since_last} days since last order | "
                    f"Churn Risk: {churn_label}"
                )

            # Purchase timeline
            st.subheader("📆 Strategic Purchase Timeline")

            customer_trend = (
                customer_data.groupby("date")
                .agg(revenue=("revenue", "sum"), quantity=("quantity", "sum"))
                .reset_index()
            )

            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=customer_trend["date"],
                    y=customer_trend["revenue"],
                    mode="lines+markers",
                    name="Daily Revenue",
                    line=dict(color="#1f77b4", width=3),
                    marker=dict(size=8),
                    fill="tozeroy",
                    fillcolor="rgba(31, 119, 180, 0.1)",
                )
            )
            fig.update_layout(
                title=f"Revenue Timeline: {selected_customer}",
                xaxis_title="Date",
                yaxis_title="Revenue (KES)",
                height=400,
            )
            st.plotly_chart(fig, use_container_width=True)

            # Recent transactions
            st.subheader("🧠 Transaction Intelligence")

            recent_transactions = (
                customer_data.sort_values("date", ascending=False)
                .head(10)[["date", "sku_id", "quantity", "unit_price", "revenue"]]
                .copy()
            )

            recent_transactions["date"] = recent_transactions["date"].dt.strftime(
                "%Y-%m-%d"
            )

            styled_df = (
                recent_transactions.rename(
                    columns={
                        "date": "Date",
                        "sku_id": "Product ID",
                        "quantity": "Quantity",
                        "unit_price": "Unit Price",
                        "revenue": "Revenue",
                    }
                )
                .style.format(
                    {
                        "Revenue": "KES {:,.0f}",
                        "Unit Price": "KES {:,.0f}",
                    }
                )
            )

            st.dataframe(styled_df, use_container_width=True)

            # Strategic insights
            st.subheader("🎯 Advanced Customer Intelligence")

            insights = generate_customer_insights(
                customer_data,
                selected_customer,
                total_spend,
                order_count,
                avg_order_value,
                days_since_last,
            )

            for insight in insights:
                if insight["priority"] == "high":
                    st.error(f"🚨 **{insight['title']}** - {insight['message']}")
                elif insight["priority"] == "medium":
                    st.warning(f"⚠️ **{insight['title']}** - {insight['message']}")
                else:
                    st.success(f"✅ **{insight['title']}** - {insight['message']}")

                st.write("**Strategic Actions:**")
                for action in insight["actions"]:
                    st.write(f"🎯 {action}")

                if "roi_impact" in insight:
                    st.info(f"**Estimated Impact:** {insight['roi_impact']}")
                st.write("")

            # Strategic action center
            st.subheader("🚀 Strategic Action Center")

            col1, col2, col3 = st.columns(3)

            with col1:
                if st.button(
                    "📧 Execute Email Campaign",
                    key=f"email_campaign_{selected_customer}",
                    use_container_width=True,
                ):
                    st.success(
                        f"Targeted email campaign launched for {selected_customer}"
                    )

            with col2:
                if st.button(
                    "🎯 Create Strategic Offer",
                    key=f"offer_{selected_customer}",
                    use_container_width=True,
                ):
                    st.success(
                        f"Personalized strategic offer created for {selected_customer}"
                    )

            with col3:
                if st.button(
                    "📊 Export Intelligence Report",
                    key=f"export_{selected_customer}",
                    use_container_width=True,
                ):
                    st.success(
                        f"Comprehensive customer intelligence report exported for {selected_customer}"
                    )

    except Exception as e:
        st.error(f"❌ Customer 360° error: {e}")
        st.info(
            "Please ensure you have selected a valid customer with transaction history."
        )


def render_growth_strategies(sales_data: pd.DataFrame):
    """Render customer growth strategies and opportunity analysis."""
    st.header("🚀 Customer Growth Strategies")

    st.markdown(
        """
    <div style="background: #f0f8ff; padding: 20px; border-radius: 8px;
                border-left: 4px solid #007bff; margin-bottom: 20px;">
        <strong>💡 Strategic Growth Framework</strong><br>
        Identify and execute growth opportunities across your customer portfolio using data-driven strategies.
    </div>
    """,
        unsafe_allow_html=True,
    )

    st.subheader("📈 Portfolio Growth Opportunities")

    customer_growth = (
        sales_data.groupby("customer_id")
        .agg(
            total_revenue=("revenue", "sum"),
            first_purchase=("date", "min"),
            last_purchase=("date", "max"),
            order_count=("date", "count"),
        )
        .reset_index()
    )

    total_customers = len(customer_growth)
    high_value_customers = len(
        customer_growth[
            customer_growth["total_revenue"]
            > customer_growth["total_revenue"].quantile(0.8)
        ]
    )
    growth_potential = total_customers - high_value_customers

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Total Customer Base", total_customers)

    with col2:
        st.metric("High-Value Customers", high_value_customers)

    with col3:
        st.metric("Growth Potential", f"{growth_potential} customers")

    # 🔗 Segment-driven wiring from RFM into Growth Strategies
    st.subheader("🔗 Segment-Driven Action Queues")

    rfm_data = build_rfm_segments(sales_data)

    if not rfm_data.empty:
        seg_counts = (
            rfm_data["segment"]
            .value_counts()
            .reset_index()
            .rename(columns={"index": "Segment", "segment": "Customer Count"})
        )
        st.dataframe(seg_counts, use_container_width=True)

        # Build a simple action queue: focus on VIP + At Risk
        action_queue = rfm_data[
            rfm_data["segment"].isin(["VIP Champions", "At Risk"])
        ].copy()

        if not action_queue.empty:
            # Attach recommended action
            def map_action(seg):
                if seg == "VIP Champions":
                    return "VIP nurture: premium offers & advocacy"
                elif seg == "At Risk":
                    return "Win-back: discount + outreach"
                else:
                    return "Standard engagement"

            action_queue["recommended_action"] = action_queue["segment"].apply(
                map_action
            )

            # Prioritize: At Risk (highest recency_days) then VIP by monetary_value
            action_queue = action_queue.sort_values(
                by=["segment", "recency_days", "monetary_value"],
                ascending=[True, False, False],
            )

            st.markdown("**Priority Action Queue (VIP & At Risk customers)**")
            st.dataframe(
                action_queue[
                    [
                        "customer_id",
                        "segment",
                        "recency_days",
                        "frequency_count",
                        "monetary_value",
                        "recommended_action",
                    ]
                ]
                .rename(
                    columns={
                        "customer_id": "Customer ID",
                        "recency_days": "Days Since Last Purchase",
                        "frequency_count": "Frequency",
                        "monetary_value": "Total Revenue (KES)",
                    }
                )
                .head(30),
                use_container_width=True,
            )
        else:
            st.info("No VIP or At Risk customers found for action queue.")
    else:
        st.info("RFM segments not available yet for action queues.")

    st.subheader("🎯 Strategic Growth Initiatives")

    growth_initiatives = [
        {
            "name": "Customer Value Expansion",
            "description": "Increase spending from existing high-value customers",
            "potential_impact": "15–25% revenue growth",
            "effort": "Medium",
            "timeline": "60–90 days",
        },
        {
            "name": "Portfolio Penetration",
            "description": "Increase purchase frequency across customer base",
            "potential_impact": "10–20% revenue growth",
            "effort": "High",
            "timeline": "90–120 days",
        },
        {
            "name": "At-Risk Recovery",
            "description": "Reactivate dormant and at-risk customers",
            "potential_impact": "5–15% revenue recovery",
            "effort": "Medium",
            "timeline": "30–60 days",
        },
        {
            "name": "Customer Advocacy",
            "description": "Leverage satisfied customers for referrals",
            "potential_impact": "8–12% new customer acquisition",
            "effort": "Low",
            "timeline": "45–75 days",
        },
    ]

    for initiative in growth_initiatives:
        with st.expander(f"🚀 {initiative['name']} | Impact: {initiative['potential_impact']}"):
            st.write(f"**Description:** {initiative['description']}")
            st.write(f"**Effort Level:** {initiative['effort']}")
            st.write(f"**Implementation Timeline:** {initiative['timeline']}")

            if st.button(
                f"Launch {initiative['name']}",
                key=f"growth_{initiative['name']}",
                use_container_width=True,
            ):
                st.success(f"{initiative['name']} initiative launched!")


def generate_customer_insights(
    customer_data: pd.DataFrame,
    customer_id: str,
    total_spend: float,
    order_count: int,
    avg_order_value: float,
    days_since_last: int,
):
    """Generate strategic customer insights."""
    insights = []

    # High-value customer insight
    if total_spend > 100_000 and order_count >= 10:
        insights.append(
            {
                "priority": "high",
                "title": "🏆 VIP Customer Identified",
                "message": (
                    f"This customer has generated KES {total_spend:,.0f} "
                    f"across {order_count} orders with high loyalty."
                ),
                "actions": [
                    "Assign dedicated account manager",
                    "Create exclusive VIP loyalty program",
                    "Offer premium support services",
                    "Provide early access to new products",
                ],
                "roi_impact": (
                    "Potential 25–40% revenue growth through premium services"
                ),
            }
        )

    # At-risk customer insight
    elif days_since_last > 90:
        insights.append(
            {
                "priority": "high",
                "title": "⚠️ High Churn Risk Customer",
                "message": (
                    f"No purchases in {days_since_last} days. "
                    "Immediate action required to prevent churn."
                ),
                "actions": [
                    "Execute personalized win-back campaign",
                    "Offer strategic discount (20–30%)",
                    "Conduct satisfaction survey",
                    "Schedule proactive outreach call",
                ],
                "roi_impact": (
                    "15–25% recovery probability with proper intervention"
                ),
            }
        )

    # Growth opportunity insight
    elif order_count >= 5 and avg_order_value < 8000:
        insights.append(
            {
                "priority": "medium",
                "title": "📈 Upsell Opportunity",
                "message": (
                    f"Loyal customer with {order_count} orders but below-average "
                    f"order value of KES {avg_order_value:,.0f}."
                ),
                "actions": [
                    "Create premium product bundle offers",
                    "Implement volume-based discount tiers",
                    "Offer complementary product recommendations",
                    "Provide educational content on product value",
                ],
                "roi_impact": (
                    "Potential 20–35% increase in average order value"
                ),
            }
        )

    # New customer insight
    elif order_count == 1:
        insights.append(
            {
                "priority": "medium",
                "title": "🆕 New Customer Onboarding",
                "message": "First-time purchaser with significant future value potential.",
                "actions": [
                    "Send welcome and onboarding sequence",
                    "Offer second purchase incentive",
                    "Provide educational content",
                    "Enroll in entry-level loyalty program",
                ],
                "roi_impact": (
                    "40–60% probability of becoming repeat customer with proper "
                    "onboarding"
                ),
            }
        )

    # Default strategic insight
    else:
        insights.append(
            {
                "priority": "low",
                "title": "📊 Strategic Engagement Opportunity",
                "message": "Solid customer with consistent engagement patterns.",
                "actions": [
                    "Increase communication frequency",
                    "Offer seasonal promotions",
                    "Provide product usage tips",
                    "Enhance customer experience",
                ],
                "roi_impact": "10–20% potential value growth through engagement",
            }
        )

    return insights


if __name__ == "__main__":
    render()
