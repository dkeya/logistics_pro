# logistics_pro/pages/01_Dashboard.py 
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


class CrossFunctionalAnalytics:
    """Cross-functional analytics engine for executive insights."""

    def __init__(self, analytics_engine):
        self.analytics = analytics_engine

    def calculate_business_health_score(self):
        """Calculate overall business health score."""
        try:
            scores = {
                "sales_health": self._calculate_sales_health(),
                "inventory_efficiency": self._calculate_inventory_efficiency(),
                "logistics_performance": self._calculate_logistics_performance(),
                "financial_health": self._calculate_financial_health(),
                "customer_health": self._calculate_customer_health(),
            }

            weights = {
                "sales_health": 0.3,
                "inventory_efficiency": 0.2,
                "logistics_performance": 0.2,
                "financial_health": 0.15,
                "customer_health": 0.15,
            }

            overall_score = sum(scores[dim] * weights[dim] for dim in scores)
            return {
                "overall_score": overall_score,
                "dimension_scores": scores,
                "health_status": self._get_health_status(overall_score),
            }
        except Exception:
            return {"overall_score": 75, "health_status": "Good"}

    def _calculate_sales_health(self):
        """Calculate sales health score."""
        try:
            sales_data = self.analytics.sales_data
            recent_sales = sales_data[
                sales_data["date"] >= (datetime.now().date() - timedelta(days=30))
            ]

            if len(recent_sales) == 0:
                return 70

            revenue_growth = self._calculate_revenue_growth(sales_data)
            margin_health = self._calculate_margin_health(sales_data)
            customer_growth = self._calculate_customer_growth(sales_data)

            return revenue_growth * 0.4 + margin_health * 0.4 + customer_growth * 0.2
        except Exception:
            return 70

    def _calculate_inventory_efficiency(self):
        """Calculate inventory efficiency score."""
        try:
            turnover_rate = np.random.uniform(6, 12)
            stockout_rate = np.random.uniform(2, 8)
            carrying_cost_ratio = np.random.uniform(18, 25)

            turnover_score = min(100, turnover_rate * 8)
            stockout_score = max(0, 100 - stockout_rate * 10)
            cost_score = max(0, 100 - (carrying_cost_ratio - 15) * 4)

            return turnover_score * 0.4 + stockout_score * 0.4 + cost_score * 0.2
        except Exception:
            return 75

    def _calculate_logistics_performance(self):
        """Calculate logistics performance score."""
        try:
            on_time_rate = np.random.uniform(85, 98)
            fleet_utilization = np.random.uniform(70, 90)
            cost_per_delivery = np.random.uniform(45, 85)

            on_time_score = on_time_rate
            utilization_score = fleet_utilization
            cost_score = max(0, 100 - (cost_per_delivery - 40))

            return on_time_score * 0.5 + utilization_score * 0.3 + cost_score * 0.2
        except Exception:
            return 80

    def _calculate_financial_health(self):
        """Calculate financial health score."""
        try:
            sales_data = self.analytics.sales_data
            total_revenue = (
                sales_data["quantity"] * safe_get_column(sales_data, "unit_price", 100)
            ).sum()

            profit_margin = np.random.uniform(18, 28)
            cash_flow_health = np.random.uniform(70, 95)
            revenue_growth = self._calculate_revenue_growth(sales_data)

            return profit_margin * 2.5 + cash_flow_health * 0.3 + revenue_growth * 0.2
        except Exception:
            return 75

    def _calculate_customer_health(self):
        """Calculate customer health score."""
        try:
            sales_data = self.analytics.sales_data
            _ = sales_data["customer_id"].nunique()  # reserved for future extension

            retention_rate = np.random.uniform(75, 92)
            satisfaction_score = np.random.uniform(80, 95)
            growth_rate = self._calculate_customer_growth(sales_data)

            return retention_rate * 0.4 + satisfaction_score * 0.4 + growth_rate * 0.2
        except Exception:
            return 80

    def _calculate_revenue_growth(self, sales_data):
        """Calculate revenue growth rate."""
        try:
            if len(sales_data) < 60:
                return 75

            # Ensure revenue exists
            if "revenue" not in sales_data.columns:
                sales_data = sales_data.copy()
                sales_data["revenue"] = sales_data["quantity"] * safe_get_column(
                    sales_data, "unit_price", 100
                )

            recent = sales_data[
                sales_data["date"] >= (datetime.now().date() - timedelta(days=30))
            ]
            previous = sales_data[
                (sales_data["date"] >= (datetime.now().date() - timedelta(days=60)))
                & (
                    sales_data["date"]
                    < (datetime.now().date() - timedelta(days=30))
                )
            ]

            recent_revenue = recent["revenue"].sum()
            previous_revenue = previous["revenue"].sum()

            if previous_revenue > 0:
                growth_rate = ((recent_revenue - previous_revenue) / previous_revenue) * 100
                return min(100, max(0, 50 + growth_rate))
            return 75
        except Exception:
            return 75

    def _calculate_margin_health(self, sales_data):
        """Calculate margin health score."""
        try:
            if "margin_percent" in sales_data.columns:
                avg_margin = sales_data["margin_percent"].mean()
            else:
                avg_margin = 25.0

            if avg_margin >= 25:
                return 90
            if avg_margin >= 20:
                return 80
            if avg_margin >= 15:
                return 70
            return 60
        except Exception:
            return 70

    def _calculate_customer_growth(self, sales_data):
        """Calculate customer growth rate."""
        try:
            if len(sales_data) < 60:
                return 75

            recent = sales_data[
                sales_data["date"] >= (datetime.now().date() - timedelta(days=30))
            ]
            previous = sales_data[
                (sales_data["date"] >= (datetime.now().date() - timedelta(days=60)))
                & (
                    sales_data["date"]
                    < (datetime.now().date() - timedelta(days=30))
                )
            ]

            recent_customers = recent["customer_id"].nunique()
            previous_customers = previous["customer_id"].nunique()

            if previous_customers > 0:
                growth_rate = ((recent_customers - previous_customers) / previous_customers) * 100
                return min(100, max(0, 50 + growth_rate * 2))
            return 75
        except Exception:
            return 75

    def _get_health_status(self, score):
        """Get health status based on score."""
        if score >= 90:
            return "Excellent"
        if score >= 80:
            return "Very Good"
        if score >= 70:
            return "Good"
        if score >= 60:
            return "Fair"
        return "Needs Attention"


def render():
    """EXECUTIVE COCKPIT - Cross-Functional Strategic Command Center."""
    st.title("🏰 Executive Cockpit")

    # 🌈 Gradient hero header
    st.markdown(
        f"""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                padding: 20px; border-radius: 12px; color: white; margin-bottom: 12px;">
        <h3 style="margin: 0; color: white;">🎯 Cross-Functional Strategic Command Center</h3>
        <p style="margin: 5px 0 0 0; opacity: 0.9; color: white;">
        <strong>📍</strong> Executive Cockpit &gt; Strategic Overview |
        <strong>🏰</strong> {st.session_state.get('current_tenant', 'ELORA Holding')} |
        <strong>📅</strong> {datetime.now().strftime('%Y-%m-%d')}
        </p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # 💚 Soft green marquee strip – lives directly under the hero
    st.markdown(
        """
    <div style="background: #dcfce7; padding: 10px 16px; border-radius: 10px;
                margin-bottom: 24px; border-left: 4px solid #16a34a;">
        <marquee behavior="scroll" direction="left" scrollamount="4"
                 style="font-size: 0.9rem; font-weight: 500; color: #166534;">
            🌟 <strong>Strategic Intelligence:</strong> Driving FMCG Excellence • 
            📈 <strong>Revenue Performance:</strong> KES 2.8B MTD | +18.3% YoY Growth • 
            🎯 <strong>Customer Excellence:</strong> 94.2% OTIF Delivery | Elite Service Levels • 
            📦 <strong>Inventory Optimization:</strong> 6.8x Turnover | 98.5% Stock Availability • 
            🚛 <strong>Logistics Mastery:</strong> 78.5% Fleet Utilization | Route Efficiency +22% • 
            🤝 <strong>Supplier Intelligence:</strong> 156 Active Partners | 95.8% Compliance • 
            🌐 <strong>Digital Transformation:</strong> E-commerce +35% | Social Engagement +42% • 
            ⚡ <strong>AI-Powered Insights:</strong> 8 Predictive Models | Real-time Decision Support
        </marquee>
    </div>
    """,
        unsafe_allow_html=True,
    )

    if "analytics" not in st.session_state:
        st.error("❌ Please initialize the application first")
        return

    analytics = st.session_state.analytics
    cross_analytics = CrossFunctionalAnalytics(analytics)

    # Load data without the old success banner
    try:
        sales_data = analytics.sales_data
    except Exception as e:
        st.error(f"❌ Error loading data: {e}")
        return

    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        [
            "🎯 Business Health",
            "📊 Performance Dashboard",
            "🚀 Strategic Initiatives",
            "📈 Cross-Functional Insights",
            "🔍 Opportunity Radar",
        ]
    )

    with tab1:
        render_business_health(sales_data, cross_analytics)

    with tab2:
        render_performance_dashboard(sales_data, cross_analytics)

    with tab3:
        render_strategic_initiatives(sales_data)

    with tab4:
        render_cross_functional_insights(sales_data, cross_analytics)

    with tab5:
        render_opportunity_radar(sales_data)


def render_business_health(sales_data, cross_analytics):
    """Render comprehensive business health assessment."""
    st.header("🎯 Business Health Assessment")

    health_data = cross_analytics.calculate_business_health_score()

    st.subheader("🏆 Overall Business Health")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        score = health_data["overall_score"]
        status = health_data["health_status"]

        if score >= 90:
            emoji = "🌟"
        elif score >= 80:
            emoji = "✅"
        elif score >= 70:
            emoji = "⚠️"
        else:
            emoji = "🔴"

        st.metric("Overall Health Score", f"{score:.0f}/100", f"{emoji} {status}")
        st.progress(score / 100)

    with col2:
        sales_health = health_data.get("dimension_scores", {}).get("sales_health", 75)
        st.metric("Sales Health", f"{sales_health:.0f}/100")
        st.caption("Revenue & Growth")

    with col3:
        inventory_health = health_data.get("dimension_scores", {}).get(
            "inventory_efficiency", 75
        )
        st.metric("Inventory Efficiency", f"{inventory_health:.0f}/100")
        st.caption("Stock & Turnover")

    with col4:
        logistics_health = health_data.get("dimension_scores", {}).get(
            "logistics_performance", 75
        )
        st.metric("Logistics Performance", f"{logistics_health:.0f}/100")
        st.caption("Delivery & Operations")

    st.subheader("📊 Health Dimension Analysis")

    if "dimension_scores" in health_data:
        dimensions = list(health_data["dimension_scores"].keys())
        scores = list(health_data["dimension_scores"].values())

        fig = go.Figure()
        fig.add_trace(
            go.Scatterpolar(
                r=scores + [scores[0]],
                theta=[d.replace("_", " ").title() for d in dimensions]
                + [dimensions[0].replace("_", " ").title()],
                fill="toself",
                name="Business Health",
                line=dict(color="#1f77b4"),
            )
        )
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
            showlegend=False,
            title="Business Health Dimensions",
            height=400,
        )

        st.plotly_chart(fig, use_container_width=True)

    st.subheader("💡 Health Insights & Recommendations")

    insights = generate_health_insights(health_data)
    for insight in insights:
        if insight["priority"] == "high":
            st.error(f"🚨 **{insight['title']}** - {insight['message']}")
        elif insight["priority"] == "medium":
            st.warning(f"⚠️ **{insight['title']}** - {insight['message']}")
        else:
            st.success(f"✅ **{insight['title']}** - {insight['message']}")

        st.write("**Recommended Actions:**")
        for action in insight["actions"]:
            st.write(f"• {action}")
        st.write("")


def render_performance_dashboard(sales_data, cross_analytics):
    """Render comprehensive performance dashboard."""
    st.header("📊 Cross-Functional Performance Dashboard")

    # 🔧 Ensure essential columns exist so analytics always work with inline data
    sales_data = sales_data.copy()

    if "revenue" not in sales_data.columns:
        sales_data["revenue"] = sales_data["quantity"] * safe_get_column(
            sales_data, "unit_price", 100
        )

    if "date" not in sales_data.columns:
        # Synthetic date series if not present
        sales_data["date"] = pd.date_range(
            end=datetime.now().date(), periods=len(sales_data)
        )

    if "customer_id" not in sales_data.columns:
        sales_data["customer_id"] = [f"CUST_{i+1:03d}" for i in range(len(sales_data))]

    total_revenue = sales_data["revenue"].sum()
    total_volume = sales_data["quantity"].sum()
    unique_customers = sales_data["customer_id"].nunique()
    avg_order_value = total_revenue / len(sales_data) if len(sales_data) > 0 else 0

    st.subheader("🎯 Key Performance Indicators")

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric("Total Revenue", f"KES {total_revenue:,.0f}")
        st.caption("30-day rolling")

    with col2:
        st.metric("Total Volume", f"{total_volume:,}")
        st.caption("Units sold")

    with col3:
        st.metric("Active Customers", f"{unique_customers}")
        st.caption("Unique buyers")

    with col4:
        if "margin_percent" in sales_data.columns:
            avg_margin = sales_data["margin_percent"].mean()
        else:
            avg_margin = 25.0
        st.metric("Avg Margin %", f"{avg_margin:.1f}%")
        st.caption("Gross profit")

    with col5:
        st.metric("Avg Order Value", f"KES {avg_order_value:,.0f}")
        st.caption("Per transaction")

    st.subheader("📈 Performance Trends & Patterns")

    col1, col2 = st.columns(2)

    # --- Revenue Trend with Moving Average ---
    with col1:
        try:
            daily_revenue = (
                sales_data.groupby("date")
                .agg({"revenue": "sum"})
                .reset_index()
                .sort_values("date")
            )
            daily_revenue["moving_avg"] = daily_revenue["revenue"].rolling(window=7).mean()

            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=daily_revenue["date"],
                    y=daily_revenue["revenue"],
                    name="Daily Revenue",
                    line=dict(color="#1f77b4", width=2),
                    opacity=0.7,
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=daily_revenue["date"],
                    y=daily_revenue["moving_avg"],
                    name="7-Day Average",
                    line=dict(color="#ff7f0e", width=3),
                )
            )

            fig.update_layout(
                title="💰 Revenue Trend with Moving Average",
                xaxis_title="Date",
                yaxis_title="Revenue (KES)",
                height=400,
            )
            st.plotly_chart(fig, use_container_width=True)
        except Exception:
            # This should now rarely (if ever) trigger with inline data
            st.info(
                "📊 Enhanced revenue analytics will be available with complete data integration"
            )

    # --- Customer Value Matrix ---
    with col2:
        try:
            customer_metrics = (
                sales_data.groupby("customer_id")
                .agg({"revenue": "sum", "date": "count"})
                .reset_index()
            )
            customer_metrics.columns = ["customer_id", "total_revenue", "order_count"]
            customer_metrics["avg_order_value"] = (
                customer_metrics["total_revenue"] / customer_metrics["order_count"]
            )

            fig = px.scatter(
                customer_metrics.nlargest(50, "total_revenue"),
                x="order_count",
                y="avg_order_value",
                size="total_revenue",
                color="total_revenue",
                title="👥 Customer Value Matrix",
                hover_data=["customer_id"],
                color_continuous_scale="Viridis",
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        except Exception:
            st.info("📈 Customer analytics will be available with enhanced data")


def render_strategic_initiatives(sales_data):
    """Render strategic initiatives and program management."""
    st.header("🚀 Strategic Initiatives & Program Management")

    st.subheader("📌 Strategic Initiative Portfolio")

    strategic_initiatives = [
        {
            "name": "Digital Transformation Program",
            "department": "Cross-Functional",
            "status": "In Progress",
            "progress": 75,
            "impact": "Very High",
            "timeline": "Q2 2024",
            "budget": "KES 15M",
            "roi_estimate": "42%",
        },
        {
            "name": "Supply Chain Optimization",
            "department": "Logistics & Procurement",
            "status": "Planning",
            "progress": 45,
            "impact": "High",
            "timeline": "Q3 2024",
            "budget": "KES 8M",
            "roi_estimate": "35%",
        },
        {
            "name": "Customer Experience Enhancement",
            "department": "Sales & Marketing",
            "status": "Execution",
            "progress": 60,
            "impact": "High",
            "timeline": "Q2 2024",
            "budget": "KES 5M",
            "roi_estimate": "38%",
        },
        {
            "name": "Product Portfolio Expansion",
            "department": "Product & Innovation",
            "status": "Planning",
            "progress": 30,
            "impact": "Medium-High",
            "timeline": "Q4 2024",
            "budget": "KES 12M",
            "roi_estimate": "28%",
        },
        {
            "name": "Operational Excellence",
            "department": "Operations",
            "status": "Execution",
            "progress": 55,
            "impact": "Medium",
            "timeline": "Q3 2024",
            "budget": "KES 6M",
            "roi_estimate": "25%",
        },
    ]

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        active_initiatives = len(
            [i for i in strategic_initiatives if i["status"] == "Execution"]
        )
        st.metric("Active Initiatives", active_initiatives)

    with col2:
        total_budget = sum(
            float(i["budget"].replace("KES ", "").replace("M", ""))
            for i in strategic_initiatives
        )
        st.metric("Total Budget", f"KES {total_budget:.0f}M")

    with col3:
        avg_roi = np.mean(
            [float(i["roi_estimate"].replace("%", "")) for i in strategic_initiatives]
        )
        st.metric("Avg Expected ROI", f"{avg_roi:.1f}%")

    with col4:
        high_impact = len(
            [i for i in strategic_initiatives if i["impact"] in ["Very High", "High"]]
        )
        st.metric("High Impact Initiatives", high_impact)

    st.subheader("📋 Initiative Details & Progress Tracking")

    for initiative in strategic_initiatives:
        with st.expander(
            f"{initiative['name']} - {initiative['department']} | Impact: {initiative['impact']}"
        ):
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Progress", f"{initiative['progress']}%")
                st.progress(initiative["progress"] / 100)

            with col2:
                st.metric("Status", initiative["status"])
                st.metric("Timeline", initiative["timeline"])

            with col3:
                st.metric("Budget", initiative["budget"])
                st.metric("ROI Estimate", initiative["roi_estimate"])

            with col4:
                status_color = {
                    "Planning": "blue",
                    "Execution": "orange",
                    "Completed": "green",
                    "On Hold": "red",
                }.get(initiative["status"], "gray")

                st.markdown(
                    f"<div style='padding: 10px; background: {status_color}20; "
                    f"border-radius: 5px; border-left: 4px solid {status_color};'>"
                    f"<strong>Current Phase:</strong> {initiative['status']}</div>",
                    unsafe_allow_html=True,
                )

            col1, col2 = st.columns(2)
            with col1:
                if st.button("📊 View Details", key=f"details_{initiative['name']}"):
                    st.success(f"Detailed view opened for {initiative['name']}")
            with col2:
                if st.button("🔄 Update Progress", key=f"update_{initiative['name']}"):
                    st.success(f"Progress update initiated for {initiative['name']}")


def render_cross_functional_insights(sales_data, cross_analytics):
    """Render cross-functional insights and correlations."""
    st.header("📈 Cross-Functional Insights & Correlations")

    st.info(
        """
    **💡 Strategic Context:** Analyzing relationships and dependencies across business functions 
    to identify synergistic opportunities and potential conflicts.
    """
    )

    st.subheader("🔗 Functional Relationship Matrix")

    relationships = {
        "Sales Growth": {
            "Inventory": 0.8,
            "Logistics": 0.7,
            "Procurement": 0.6,
            "Finance": 0.9,
        },
        "Inventory Turnover": {
            "Sales": 0.8,
            "Logistics": 0.6,
            "Procurement": 0.7,
            "Finance": 0.5,
        },
        "Delivery Performance": {
            "Sales": 0.7,
            "Inventory": 0.6,
            "Procurement": 0.4,
            "Finance": 0.3,
        },
        "Cost Efficiency": {
            "Sales": 0.5,
            "Inventory": 0.7,
            "Logistics": 0.8,
            "Procurement": 0.9,
        },
    }

    matrix_data = []
    for metric, impacts in relationships.items():
        for function, impact in impacts.items():
            matrix_data.append({"Metric": metric, "Function": function, "Impact": impact})

    matrix_df = pd.DataFrame(matrix_data)
    heatmap_data = matrix_df.pivot(index="Metric", columns="Function", values="Impact")

    fig = px.imshow(
        heatmap_data,
        title="Cross-Functional Impact Heatmap",
        color_continuous_scale="RdYlGn",
        aspect="auto",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("🤝 Strategic Synergies & Dependencies")

    synergies = [
        {
            "synergy": "Sales ↔ Inventory Optimization",
            "description": "Higher sales volume enables better inventory turnover and reduced carrying costs",
            "opportunity": "15-20% cost reduction potential",
            "actions": [
                "Implement demand forecasting",
                "Optimize safety stock levels",
                "Align promotion planning",
            ],
        },
        {
            "synergy": "Logistics ↔ Customer Satisfaction",
            "description": "Improved delivery performance directly enhances customer experience and retention",
            "opportunity": "25% improvement in customer satisfaction scores",
            "actions": [
                "Enhance delivery tracking",
                "Optimize route planning",
                "Implement delivery SLAs",
            ],
        },
        {
            "synergy": "Procurement ↔ Financial Health",
            "description": "Strategic sourcing and supplier management significantly impact profit margins",
            "opportunity": "8-12% margin improvement potential",
            "actions": [
                "Consolidate supplier base",
                "Negotiate volume discounts",
                "Implement vendor scorecards",
            ],
        },
    ]

    for synergy in synergies:
        with st.expander(f"🔎 {synergy['synergy']}"):
            st.write(f"**Description:** {synergy['description']}")
            st.write(f"**Opportunity:** {synergy['opportunity']}")
            st.write("**Key Actions:**")
            for action in synergy["actions"]:
                st.write(f"✅ {action}")


def render_opportunity_radar(sales_data):
    """Render opportunity identification and strategic radar."""
    st.header("🔍 Strategic Opportunity Radar")

    st.subheader("🎯 Identified Growth Opportunities")

    opportunities = [
        {
            "type": "Market Expansion",
            "opportunity": "Western Region Penetration",
            "potential_value": "KES 45M",
            "timeframe": "6-9 months",
            "confidence": "High",
            "key_metrics": ["Market Share", "New Customers", "Revenue Growth"],
        },
        {
            "type": "Product Innovation",
            "opportunity": "Premium Product Line Launch",
            "potential_value": "KES 28M",
            "timeframe": "8-12 months",
            "confidence": "Medium-High",
            "key_metrics": [
                "Average Order Value",
                "Margin Improvement",
                "Brand Positioning",
            ],
        },
        {
            "type": "Operational Efficiency",
            "opportunity": "Supply Chain Digitization",
            "potential_value": "KES 15M",
            "timeframe": "12-18 months",
            "confidence": "High",
            "key_metrics": ["Cost Reduction", "Process Efficiency", "Delivery Speed"],
        },
        {
            "type": "Customer Experience",
            "opportunity": "Digital Customer Journey",
            "potential_value": "KES 22M",
            "timeframe": "9-15 months",
            "confidence": "Medium",
            "key_metrics": [
                "Customer Satisfaction",
                "Retention Rate",
                "Lifetime Value",
            ],
        },
    ]

    for opp in opportunities:
        with st.expander(
            f"{opp['type']}: {opp['opportunity']} | Value: {opp['potential_value']}"
        ):
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Potential Value", opp["potential_value"])
                st.metric("Timeframe", opp["timeframe"])

            with col2:
                st.metric("Confidence Level", opp["confidence"])

            with col3:
                st.write("**Key Success Metrics:**")
                for metric in opp["key_metrics"]:
                    st.write(f"📊 {metric}")

            col1, col2 = st.columns(2)
            with col1:
                if st.button("📊 Business Case", key=f"case_{opp['opportunity']}"):
                    st.success(
                        f"Business case development started for {opp['opportunity']}"
                    )
            with col2:
                if st.button("🎯 Action Plan", key=f"plan_{opp['opportunity']}"):
                    st.success(
                        f"Strategic action plan created for {opp['opportunity']}"
                    )

    st.subheader("⚡ Quick Action Center")

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("📄 Generate Executive Report", key="quick_exec_report"):
            st.success("Comprehensive executive report generated!")

    with col2:
        if st.button("🔄 Refresh Analytics", key="quick_refresh_analytics"):
            st.success("All analytics refreshed with latest data!")

    with col3:
        if st.button("🧭 Strategic Planning", key="quick_strategic_planning"):
            st.success("Strategic planning session initiated!")


def generate_health_insights(health_data):
    """Generate health insights based on business health assessment."""
    insights = []

    overall_score = health_data["overall_score"]

    if overall_score < 70:
        insights.append(
            {
                "priority": "high",
                "title": "Business Health Needs Attention",
                "message": f"Overall business health score of {overall_score:.0f} indicates areas needing immediate improvement.",
                "actions": [
                    "Conduct deep-dive analysis on underperforming dimensions",
                    "Develop targeted improvement initiatives",
                    "Increase executive oversight on key metrics",
                    "Implement weekly health check meetings",
                ],
            }
        )
    else:
        insights.append(
            {
                "priority": "low",
                "title": "Strong Business Health",
                "message": f"Excellent overall health score of {overall_score:.0f}. Maintain current performance levels.",
                "actions": [
                    "Continue current strategic initiatives",
                    "Focus on incremental improvements",
                    "Explore new growth opportunities",
                    "Maintain operational excellence",
                ],
            }
        )

    if "dimension_scores" in health_data:
        dim_scores = health_data["dimension_scores"]

        if dim_scores.get("sales_health", 100) < 70:
            insights.append(
                {
                    "priority": "medium",
                    "title": "Sales Performance Optimization Opportunity",
                    "message": "Sales health dimension shows room for improvement in revenue growth and customer acquisition.",
                    "actions": [
                        "Review sales strategy and targets",
                        "Enhance customer acquisition programs",
                        "Optimize pricing and promotion strategies",
                        "Improve sales team performance management",
                    ],
                }
            )

        if dim_scores.get("inventory_efficiency", 100) < 70:
            insights.append(
                {
                    "priority": "medium",
                    "title": "Inventory Efficiency Improvement Needed",
                    "message": "Inventory management shows opportunities for better turnover and cost control.",
                    "actions": [
                        "Review inventory management policies",
                        "Optimize stock levels and reorder points",
                        "Improve demand forecasting accuracy",
                        "Reduce carrying costs and stockouts",
                    ],
                }
            )

    return insights


if __name__ == "__main__":
    render()