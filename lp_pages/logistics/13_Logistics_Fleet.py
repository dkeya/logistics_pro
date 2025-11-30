# logistics_pro/pages/logistics/13_Logistics_Fleet.py
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import numpy as np


def render():
    """🚚 FLEET INTELLIGENCE - Strategic Asset Optimization & Performance"""

    st.title("🚚 Fleet Intelligence")

    # 🌈 Gradient hero header – aligned with 01_Dashboard.py
    st.markdown(
        f"""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                padding: 20px; border-radius: 12px; color: white; margin-bottom: 12px;">
        <h3 style="margin: 0; color: white;">🎯 Strategic Fleet Optimization & Asset Intelligence</h3>
        <p style="margin: 5px 0 0 0; opacity: 0.9; color: white;">
        <strong>📍</strong> Logistics Intelligence &gt; Fleet Optimization |
        <strong>🏢</strong> {st.session_state.get('current_tenant', 'ELORA Holding')} |
        <strong>📅</strong> {datetime.now().strftime('%Y-%m-%d')}
        </p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # 💚 Soft green marquee strip – just like Executive Cockpit
    st.markdown(
        """
    <div style="background: #dcfce7; padding: 10px 16px; border-radius: 10px;
                margin-bottom: 24px; border-left: 4px solid #16a34a;">
        <marquee behavior="scroll" direction="left" scrollamount="4"
                 style="font-size: 0.9rem; font-weight: 500; color: #166534;">
            🚚 <strong>Fleet Pulse:</strong> 78.5% Utilization | 92.3% On-Time Dispatch • 
            🔧 <strong>Maintenance Intelligence:</strong> 40% Reduction Target for Unplanned Downtime • 
            ⛽ <strong>Fuel Performance:</strong> 6.9 km/L Avg | 14% Optimization Headroom • 
            🌿 <strong>Sustainability:</strong> CO₂ Intensity Trending Down | Green Fleet Pilots Underway • 
            🛰️ <strong>Telematics:</strong> Live Location, Idling & Harsh Driving Insights • 
            🧠 <strong>AI Fleet Intelligence:</strong> Predictive Maintenance & Right-Sizing Scenarios Active
        </marquee>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Check if analytics is available
    if "analytics" not in st.session_state:
        st.error("❌ Please initialize the application from the main dashboard first")
        return

    analytics = st.session_state.analytics
    data_gen = st.session_state.data_gen

    # Enhanced Tab Structure with Strategic Frameworks
    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        [
            "🏆 Fleet Intelligence",
            "🔧 Maintenance Strategy",
            "⛽ Fuel Optimization",
            "🚀 Asset Optimization",
            "🌿 Sustainability",
        ]
    )

    with tab1:
        render_fleet_intelligence(analytics, data_gen)

    with tab2:
        render_maintenance_strategy(analytics, data_gen)

    with tab3:
        render_fuel_optimization(analytics, data_gen)

    with tab4:
        render_asset_optimization(analytics, data_gen)

    with tab5:
        render_sustainability(analytics, data_gen)


class FleetIntelligenceEngine:
    """AI-powered fleet intelligence engine for strategic asset optimization"""

    def __init__(self, fleet_data, maintenance_data, fuel_data):
        self.fleet_data = fleet_data
        self.maintenance_data = maintenance_data
        self.fuel_data = fuel_data
        self.frameworks = self._initialize_strategic_frameworks()

    def _initialize_strategic_frameworks(self):
        """Initialize strategic fleet management frameworks"""
        return {
            "asset_utilization_framework": {
                "high_performance": ">85% utilization",
                "optimal_range": "75-85% utilization",
                "improvement_needed": "60-75% utilization",
                "critical_attention": "<60% utilization",
            },
            "maintenance_strategy": {
                "predictive": "IoT sensors + AI forecasting",
                "preventive": "Scheduled maintenance",
                "reactive": "Breakdown maintenance",
            },
            "sustainability_tiers": {
                "leader": "<200g CO2/km",
                "performer": "200-250g CO2/km",
                "improver": "250-300g CO2/km",
                "laggard": ">300g CO2/km",
            },
        }

    def calculate_fleet_health_score(self):
        """Calculate comprehensive fleet health score (0-100)"""
        utilization_score = self.fleet_data["utilization_rate"].mean() * 0.3
        reliability_score = self.fleet_data["reliability_score"].mean() * 0.25
        maintenance_score = (
            100
            - (self.maintenance_data["cost"].sum() / len(self.fleet_data) / 100)
        ) * 0.25
        fuel_score = (self.fleet_data["fuel_efficiency"].mean() / 10) * 0.2

        return min(100, utilization_score + reliability_score + maintenance_score + fuel_score)

    def generate_strategic_insights(self):
        """Generate AI-powered strategic insights"""
        insights = []

        # Utilization insights
        avg_utilization = self.fleet_data["utilization_rate"].mean()
        if avg_utilization < 70:
            insights.append(
                {
                    "type": "🚨 Critical Opportunity",
                    "title": "Low Fleet Utilization",
                    "description": (
                        f"Current utilization ({avg_utilization:.1f}%) is below optimal range "
                        "(75-85%). Consider right-sizing fleet or optimizing routing."
                    ),
                    "impact": "Potential 15-25% cost savings",
                    "action": "Run fleet optimization simulation",
                }
            )

        # Maintenance insights
        high_cost_vehicles = self.fleet_data[
            self.fleet_data["maintenance_cost_ytd"] > 4000
        ]
        if len(high_cost_vehicles) > 0:
            insights.append(
                {
                    "type": "🔧 Efficiency Opportunity",
                    "title": "High Maintenance Vehicles",
                    "description": (
                        f"{len(high_cost_vehicles)} vehicles exceeding "
                        "maintenance cost thresholds"
                    ),
                    "impact": f"KES {high_cost_vehicles['maintenance_cost_ytd'].sum():,.0f} annual excess cost",
                    "action": "Review maintenance strategy for high-cost vehicles",
                }
            )

        # Fuel efficiency insights
        low_efficiency = self.fleet_data[self.fleet_data["fuel_efficiency"] < 5.5]
        if len(low_efficiency) > 0:
            insights.append(
                {
                    "type": "⛽ Optimization Opportunity",
                    "title": "Fuel Inefficient Vehicles",
                    "description": (
                        f"{len(low_efficiency)} vehicles with fuel efficiency below 5.5 km/L"
                    ),
                    "impact": "15-20% higher fuel costs",
                    "action": "Implement driver training and route optimization",
                }
            )

        return insights

    def predict_maintenance_risks(self):
        """Predict maintenance risks using AI patterns"""
        risks = []

        for _, vehicle in self.fleet_data.iterrows():
            risk_score = (
                (100 - vehicle["reliability_score"]) * 0.4
                + (vehicle["maintenance_cost_ytd"] / 1000) * 0.3
                + (
                    (datetime.now().date() - vehicle["last_maintenance"].date()).days
                    / 30
                )
                * 0.3
            )

            if risk_score > 70:
                risks.append(
                    {
                        "vehicle_id": vehicle["vehicle_id"],
                        "risk_level": "🔴 High",
                        "risk_score": risk_score,
                        "primary_concern": "Maintenance overdue & high costs",
                        "recommended_action": "Immediate maintenance scheduling",
                        "estimated_cost": "KES 15,000-25,000",
                    }
                )
            elif risk_score > 50:
                risks.append(
                    {
                        "vehicle_id": vehicle["vehicle_id"],
                        "risk_level": "🟠 Medium",
                        "risk_score": risk_score,
                        "primary_concern": "Elevated maintenance risk",
                        "recommended_action": "Schedule within 2 weeks",
                        "estimated_cost": "KES 8,000-15,000",
                    }
                )

        return risks


def render_fleet_intelligence(analytics, data_gen):
    """Render comprehensive fleet intelligence dashboard"""
    st.header("🏆 Fleet Intelligence Dashboard")

    # Generate enhanced data
    fleet_data = generate_enhanced_fleet_data(data_gen)
    utilization_data = generate_enhanced_utilization_data(data_gen)
    maintenance_data = generate_enhanced_maintenance_data(data_gen)
    fuel_data = generate_enhanced_fuel_data(data_gen)

    # Initialize intelligence engine
    intelligence_engine = FleetIntelligenceEngine(fleet_data, maintenance_data, fuel_data)

    # AI-Powered Fleet Insights
    with st.expander("🤖 AI FLEET INTELLIGENCE SUMMARY", expanded=True):
        col1, col2, col3, col4 = st.columns(4)

        fleet_health = intelligence_engine.calculate_fleet_health_score()

        with col1:
            st.metric(
                "🏆 Fleet Health Score",
                f"{fleet_health:.0f}/100",
                delta="+5.2" if fleet_health > 75 else "-3.8",
                delta_color="normal" if fleet_health > 75 else "inverse",
            )
            st.caption("Comprehensive asset performance")

        with col2:
            utilization = fleet_data["utilization_rate"].mean()
            st.metric(
                "🚚 Fleet Utilization",
                f"{utilization:.1f}%",
                "+4.2%" if utilization > 75 else "-2.1%",
            )
            st.caption("Asset utilization efficiency")

        with col3:
            maintenance_efficiency = calculate_maintenance_efficiency(maintenance_data)
            st.metric("🔧 Maintenance Efficiency", f"{maintenance_efficiency:.1f}%", "+3.8%")
            st.caption("Maintenance planning effectiveness")

        with col4:
            fuel_optimization = calculate_fuel_optimization_potential(fuel_data)
            st.metric("⛽ Fuel Optimization", f"{fuel_optimization:.1f}%", "+2.1%")
            st.caption("Fuel efficiency improvement potential")

        strategic_insights = intelligence_engine.generate_strategic_insights()
        if strategic_insights:
            st.success(
                """
            **💡 Strategic Insight:** 
            - **Asset Optimization**: Right-size fleet by replacing underutilized large trucks  
            - **Predictive Maintenance**: Implement IoT sensors for 40% reduction in unplanned downtime  
            - **Fuel Efficiency**: Driver training program can reduce fuel costs by 12-15%  
            - **Sustainability**: Electric vehicle transition for urban routes shows 25% TCO reduction
            """
            )

    # Strategic Analysis Parameters
    st.subheader("🎯 Strategic Analysis Parameters")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        time_period = st.selectbox(
            "Analysis Period",
            ["Last 30 Days", "Last 90 Days", "Year to Date", "All Time"],
            key="fleet_time",
        )
    with col2:
        vehicle_type_filter = st.selectbox(
            "Vehicle Type",
            ["All"] + sorted(fleet_data["vehicle_type"].unique().tolist()),
            key="fleet_type",
        )
    with col3:
        performance_view = st.selectbox(
            "Performance Metric",
            ["Utilization", "Reliability", "Cost Efficiency", "Fuel Efficiency"],
            key="fleet_view",
        )
    with col4:
        analysis_depth = st.selectbox(
            "Analysis Depth",
            ["Strategic Overview", "Detailed Analysis", "Executive Summary"],
            key="fleet_depth",
        )

    # Enhanced Fleet Performance KPIs
    st.subheader("📊 Fleet Performance Scorecard")

    # Calculate comprehensive metrics
    total_vehicles = len(fleet_data)
    active_vehicles = len(fleet_data[fleet_data["status"] == "Active"])
    avg_utilization = fleet_data["utilization_rate"].mean()
    total_capacity = fleet_data["capacity_kg"].sum()
    avg_reliability = fleet_data["reliability_score"].mean()
    total_maintenance_cost = maintenance_data["cost"].sum()
    avg_fuel_efficiency = fleet_data["fuel_efficiency"].mean()

    # Enhanced KPI Dashboard
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric("Total Vehicles", total_vehicles)
        st.caption("Fleet Size")

    with col2:
        st.metric("Active Vehicles", f"{active_vehicles}")
        st.caption("Operational Readiness")

    with col3:
        st.metric("Avg Utilization", f"{avg_utilization:.1f}%")
        st.caption("Asset Efficiency")

    with col4:
        st.metric("Reliability Score", f"{avg_reliability:.1f}%")
        st.caption("Operational Reliability")

    with col5:
        st.metric("Fuel Efficiency", f"{avg_fuel_efficiency:.1f} km/L")
        st.caption("Fuel Performance")

    # Strategic Fleet Intelligence
    st.subheader("📈 Strategic Fleet Intelligence")

    col1, col2 = st.columns(2)

    with col1:
        # Enhanced utilization trends with strategic context
        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=utilization_data["month"],
                y=utilization_data["utilization_rate"],
                name="Utilization Rate",
                line=dict(color="#1f77b4", width=4),
                fill="tozeroy",
                fillcolor="rgba(31, 119, 180, 0.1)",
            )
        )

        # Add strategic targets
        fig.add_hline(y=85, line_dash="dash", line_color="green", annotation_text="Optimal Target")
        fig.add_hline(y=75, line_dash="dash", line_color="orange", annotation_text="Minimum Target")

        fig.update_layout(
            title="🚀 Fleet Utilization Trends with Strategic Targets",
            xaxis_title="Month",
            yaxis_title="Utilization Rate (%)",
            height=400,
            showlegend=True,
        )
        st.plotly_chart(fig, use_container_width=True)

        # Strategic insight
        current_vs_target = avg_utilization - 75
        performance_status = "above" if current_vs_target >= 0 else "below"
        st.info(
            f"**Performance Gap:** Current utilization is {abs(current_vs_target):.1f}% "
            f"{performance_status} minimum target of 75%"
        )

    with col2:
        # Enhanced vehicle type performance matrix
        type_performance = (
            fleet_data.groupby("vehicle_type")
            .agg(
                {
                    "utilization_rate": "mean",
                    "reliability_score": "mean",
                    "maintenance_cost_ytd": "mean",
                    "fuel_efficiency": "mean",
                }
            )
            .reset_index()
        )

        fig = px.scatter(
            type_performance,
            x="utilization_rate",
            y="reliability_score",
            size="maintenance_cost_ytd",
            color="fuel_efficiency",
            title="🎯 Vehicle Type Performance Matrix",
            hover_data=["vehicle_type"],
            size_max=40,
            color_continuous_scale="RdYlGn",
        )

        # Add performance quadrants
        fig.add_hline(
            y=type_performance["reliability_score"].mean(),
            line_dash="dash",
            line_color="red",
        )
        fig.add_vline(
            x=type_performance["utilization_rate"].mean(),
            line_dash="dash",
            line_color="red",
        )

        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    # Advanced Fleet Analytics
    st.subheader("🔍 Advanced Fleet Analytics")

    col1, col2 = st.columns(2)

    with col1:
        # Strategic vehicle age analysis
        fleet_data["vehicle_age"] = datetime.now().year - fleet_data["year"]
        age_analysis = (
            fleet_data.groupby("vehicle_age")
            .agg(
                {
                    "maintenance_cost_ytd": "mean",
                    "reliability_score": "mean",
                    "utilization_rate": "mean",
                }
            )
            .reset_index()
        )

        fig = px.scatter(
            age_analysis,
            x="vehicle_age",
            y="maintenance_cost_ytd",
            size="reliability_score",
            color="utilization_rate",
            title="📊 Vehicle Age vs Maintenance Cost Analysis",
            color_continuous_scale="RdYlGn_r",
            size_max=30,
        )

        # Add trend line
        z = np.polyfit(age_analysis["vehicle_age"], age_analysis["maintenance_cost_ytd"], 1)
        p = np.poly1d(z)
        fig.add_trace(
            go.Scatter(
                x=age_analysis["vehicle_age"],
                y=p(age_analysis["vehicle_age"]),
                name="Cost Trend",
                line=dict(color="red", dash="dash"),
            )
        )

        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Fleet composition and capacity analysis
        composition = fleet_data["vehicle_type"].value_counts()

        fig = px.pie(
            values=composition.values,
            names=composition.index,
            title="🚚 Fleet Composition by Vehicle Type",
            hole=0.4,
            color_discrete_sequence=px.colors.qualitative.Set3,
        )
        st.plotly_chart(fig, use_container_width=True)

    # Real-time Fleet Status with Strategic Context
    st.subheader("📍 Real-time Fleet Status & Strategic Monitoring")

    status_counts = fleet_data["status"].value_counts()

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        active = status_counts.get("Active", 0)
        st.metric("🟢 Active", active, "Operational")

    with col2:
        maintenance = status_counts.get("Maintenance", 0)
        st.metric("🔧 Maintenance", maintenance, delta_color="inverse")

    with col3:
        available = status_counts.get("Available", 0)
        st.metric("🟦 Available", available, "Ready for deployment")

    with col4:
        reserved = status_counts.get("Reserved", 0)
        st.metric("🟡 Reserved", reserved)

    with col5:
        utilization_rate = (active / total_vehicles) * 100 if total_vehicles > 0 else 0
        st.metric("📊 Utilization", f"{utilization_rate:.1f}%")

    # Strategic Fleet Alerts
    st.subheader("🚨 Strategic Fleet Alerts & Opportunities")

    # Generate intelligent alerts
    alerts = generate_strategic_fleet_alerts(fleet_data, maintenance_data)

    for alert in alerts:
        if alert["priority"] == "🔴 Critical":
            st.error(f"{alert['priority']} {alert['vehicle_id']} - {alert['issue']}")
        elif alert["priority"] == "🟠 High":
            st.warning(f"{alert['priority']} {alert['vehicle_id']} - {alert['issue']}")
        else:
            st.info(f"{alert['priority']} {alert['vehicle_id']} - {alert['issue']}")

        st.caption(f"Impact: {alert['impact']} | Action: {alert['action']}")

        col1, col2 = st.columns([1, 4])
        with col1:
            if st.button("🛠️ Resolve", key=f"resolve_{alert['vehicle_id']}"):
                st.success(f"Action initiated for {alert['vehicle_id']}")


def render_maintenance_strategy(analytics, data_gen):
    """Render advanced maintenance strategy and predictive analytics"""
    st.header("🔧 Maintenance Strategy & Predictive Intelligence")

    st.info(
        """
    **💡 Strategic Context:** Transform maintenance from reactive cost center to strategic asset 
    optimization through predictive analytics, reliability engineering, and intelligent scheduling.
    """
    )

    # Generate enhanced data
    fleet_data = generate_enhanced_fleet_data(data_gen)
    maintenance_data = generate_enhanced_maintenance_data(data_gen)

    # Initialize intelligence engine
    intelligence_engine = FleetIntelligenceEngine(fleet_data, maintenance_data, None)

    # Maintenance Performance Intelligence
    st.subheader("📊 Maintenance Performance Intelligence")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        total_maintenance_cost = maintenance_data["cost"].sum()
        st.metric("Monthly Maintenance Cost", f"KES {total_maintenance_cost:,.0f}")
        st.caption("Total maintenance expenditure")

    with col2:
        downtime_percentage = calculate_downtime_percentage(fleet_data)
        st.metric("Vehicle Downtime", f"{downtime_percentage}%", delta_color="inverse")
        st.caption("Operational availability impact")

    with col3:
        predictive_accuracy = calculate_predictive_accuracy(maintenance_data)
        st.metric("Predictive Accuracy", f"{predictive_accuracy}%", "+8.2%")
        st.caption("AI prediction reliability")

    with col4:
        maintenance_backlog = calculate_maintenance_backlog(maintenance_data)
        st.metric("Maintenance Backlog", f"{maintenance_backlog} vehicles")
        st.caption("Scheduled work pending")

    # Advanced Maintenance Analytics
    st.subheader("🔍 Advanced Maintenance Analytics")

    col1, col2 = st.columns(2)

    with col1:
        # Enhanced maintenance cost analysis
        cost_by_type = (
            maintenance_data.groupby("maintenance_type")["cost"].sum().reset_index()
        )

        fig = px.sunburst(
            cost_by_type,
            path=["maintenance_type"],
            values="cost",
            title="💰 Maintenance Cost Distribution by Type",
            color="cost",
            color_continuous_scale="RdYlGn_r",
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Maintenance frequency and impact analysis
        frequency_by_vehicle = (
            maintenance_data.groupby("vehicle_id")
            .agg({"maintenance_id": "count", "cost": "sum"})
            .reset_index()
        )
        frequency_by_vehicle = frequency_by_vehicle.merge(
            fleet_data[["vehicle_id", "vehicle_type"]], on="vehicle_id"
        )

        fig = px.scatter(
            frequency_by_vehicle,
            x="maintenance_id",
            y="cost",
            color="vehicle_type",
            size="cost",
            title="📈 Maintenance Frequency vs Cost Impact",
            hover_data=["vehicle_id"],
            size_max=30,
        )
        st.plotly_chart(fig, use_container_width=True)

    # Predictive Maintenance Intelligence
    st.subheader("🛠️ Predictive Maintenance Intelligence")

    # Generate predictive alerts
    predictive_risks = intelligence_engine.predict_maintenance_risks()

    if predictive_risks:
        st.write("**⚠️ Predictive Maintenance Alerts:**")

        for risk in predictive_risks:
            with st.expander(
                f"{risk['risk_level']} {risk['vehicle_id']} - Risk Score: {risk['risk_score']:.0f}"
            ):
                col1, col2 = st.columns(2)

                with col1:
                    st.write(f"**Primary Concern:** {risk['primary_concern']}")
                    st.write(f"**Recommended Action:** {risk['recommended_action']}")

                with col2:
                    st.write(f"**Estimated Cost:** {risk['estimated_cost']}")
                    st.write("**Confidence Level:** 85%")

                # Action buttons
                col1, col2, col3 = st.columns(3)
                with col1:
                    if st.button(
                        "📅 Schedule Maintenance",
                        key=f"schedule_{risk['vehicle_id']}",
                    ):
                        st.success(f"Maintenance scheduled for {risk['vehicle_id']}")
                with col2:
                    if st.button(
                        "📊 Detailed Analysis", key=f"analyze_{risk['vehicle_id']}"
                    ):
                        st.success(
                            f"Detailed analysis generated for {risk['vehicle_id']}"
                        )
                with col3:
                    if st.button(
                        "⏰ Remind Later", key=f"remind_{risk['vehicle_id']}"
                    ):
                        st.info(
                            f"Reminder set for {risk['vehicle_id']} in 7 days"
                        )
    else:
        st.success("✅ No critical predictive maintenance alerts")

    # Maintenance Strategy Optimization
    st.subheader("🎯 Maintenance Strategy Optimization")

    maintenance_strategies = [
        {
            "strategy": "🛠️ Predictive Maintenance Program",
            "current_state": "Reactive maintenance causing 12% unplanned downtime",
            "recommendation": "Implement IoT sensors and AI-powered predictive analytics",
            "investment": "KES 2.5M",
            "savings_potential": "KES 1.8M annually",
            "roi_period": "17 months",
            "implementation_timeline": "6-9 months",
            "risk_level": "Low",
        },
        {
            "strategy": "⚡ Maintenance Process Automation",
            "current_state": "Manual scheduling causing 15% efficiency loss",
            "recommendation": "Implement automated maintenance management system",
            "investment": "KES 1.2M",
            "savings_potential": "KES 850K annually",
            "roi_period": "14 months",
            "implementation_timeline": "3-4 months",
            "risk_level": "Low",
        },
        {
            "strategy": "🤖 Technician Efficiency Program",
            "current_state": "Variable technician performance impacting service quality",
            "recommendation": "Implement skill development and performance monitoring",
            "investment": "KES 800K",
            "savings_potential": "KES 1.2M annually",
            "roi_period": "8 months",
            "implementation_timeline": "4-6 months",
            "risk_level": "Low",
        },
    ]

    for strategy in maintenance_strategies:
        with st.expander(
            f"{strategy['strategy']} - ROI: {strategy['roi_period']}"
        ):
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Investment", strategy["investment"])

            with col2:
                st.metric("Annual Savings", strategy["savings_potential"])

            with col3:
                st.metric("ROI Period", strategy["roi_period"])

            with col4:
                st.metric("Risk Level", strategy["risk_level"])

            st.write(f"**Current State:** {strategy['current_state']}")
            st.write(f"**Recommended Action:** {strategy['recommendation']}")
            st.write(
                f"**Implementation Timeline:** {strategy['implementation_timeline']}"
            )

            # Strategy implementation actions
            col1, col2 = st.columns(2)
            with col1:
                if st.button(
                    "📄 Develop Business Case", key=f"case_{strategy['strategy']}"
                ):
                    st.success(
                        f"Business case development started for {strategy['strategy']}"
                    )
            with col2:
                if st.button(
                    "🚀 Implement Strategy",
                    key=f"implement_{strategy['strategy']}",
                ):
                    st.success(
                        f"Implementation initiated for {strategy['strategy']}"
                    )


def render_fuel_optimization(analytics, data_gen):
    """Render advanced fuel optimization and efficiency analytics"""
    st.header("⛽ Fuel Optimization & Efficiency Intelligence")

    st.info(
        """
    **💡 Strategic Context:** Transform fuel management from cost control to strategic advantage 
    through AI-powered optimization, driver behavior analytics, and route efficiency intelligence.
    """
    )

    # Generate enhanced data
    fleet_data = generate_enhanced_fleet_data(data_gen)
    fuel_data = generate_enhanced_fuel_data(data_gen)

    # Fuel Performance Intelligence
    st.subheader("📊 Fuel Performance Intelligence")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        total_fuel_cost = fuel_data["fuel_cost"].sum()
        st.metric("Monthly Fuel Cost", f"KES {total_fuel_cost:,.0f}")
        st.caption("Total fuel expenditure")

    with col2:
        avg_fuel_efficiency = fleet_data["fuel_efficiency"].mean()
        st.metric("Avg Fuel Efficiency", f"{avg_fuel_efficiency:.1f} km/L")
        st.caption("Fleet-wide efficiency")

    with col3:
        fuel_wastage = calculate_fuel_wastage(fuel_data)
        st.metric("Fuel Wastage", f"{fuel_wastage}%", delta_color="inverse")
        st.caption("Optimization opportunity")

    with col4:
        co2_emissions = calculate_co2_emissions(fuel_data)
        st.metric("CO2 Emissions", f"{co2_emissions:,.0f} kg")
        st.caption("Environmental impact")

    # Advanced Fuel Analytics
    st.subheader("🔍 Advanced Fuel Analytics")

    col1, col2 = st.columns(2)

    with col1:
        # Enhanced fuel efficiency trends
        monthly_efficiency = (
            fuel_data.groupby(fuel_data["date"].dt.to_period("M"))[
                "fuel_efficiency"
            ]
            .mean()
            .reset_index()
        )
        monthly_efficiency["date"] = monthly_efficiency["date"].astype(str)

        fig = px.line(
            monthly_efficiency,
            x="date",
            y="fuel_efficiency",
            title="📈 Monthly Fuel Efficiency Trend Analysis",
            markers=True,
            line_shape="spline",
        )

        # Add efficiency targets
        fig.add_hline(
            y=7.0, line_dash="dash", line_color="green", annotation_text="Target"
        )
        fig.add_hline(
            y=6.0, line_dash="dash", line_color="orange", annotation_text="Minimum"
        )

        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Vehicle efficiency comparison
        efficiency_by_type = (
            fleet_data.groupby("vehicle_type")["fuel_efficiency"]
            .mean()
            .reset_index()
        )

        fig = px.bar(
            efficiency_by_type,
            x="vehicle_type",
            y="fuel_efficiency",
            title="🚚 Fuel Efficiency by Vehicle Type",
            color="fuel_efficiency",
            color_continuous_scale="RdYlGn",
        )
        st.plotly_chart(fig, use_container_width=True)

    # Fuel Optimization Opportunities
    st.subheader("🎯 Fuel Optimization Opportunities")

    fuel_optimizations = [
        {
            "opportunity": "🚀 Driver Training Program",
            "current_performance": "15% efficiency variation between drivers",
            "target_performance": "8% efficiency variation",
            "key_actions": [
                "Eco-driving training",
                "Performance monitoring",
                "Incentive programs",
            ],
            "savings_potential": "KES 1.2M annually",
            "implementation_cost": "KES 450K",
            "roi_period": "5 months",
            "environmental_impact": "12% CO2 reduction",
        },
        {
            "opportunity": "🚛 Route Optimization Integration",
            "current_performance": "Static routing without fuel considerations",
            "target_performance": "Fuel-optimized dynamic routing",
            "key_actions": [
                "Integrate fuel efficiency into route planning",
                "Real-time traffic optimization",
            ],
            "savings_potential": "KES 1.8M annually",
            "implementation_cost": "KES 850K",
            "roi_period": "7 months",
            "environmental_impact": "18% CO2 reduction",
        },
        {
            "opportunity": "🔧 Vehicle Maintenance Optimization",
            "current_performance": "Irregular maintenance affecting fuel efficiency",
            "target_performance": "Optimal maintenance scheduling",
            "key_actions": [
                "Tire pressure monitoring",
                "Engine tuning optimization",
                "Aerodynamic improvements",
            ],
            "savings_potential": "KES 950K annually",
            "implementation_cost": "KES 320K",
            "roi_period": "4 months",
            "environmental_impact": "8% CO2 reduction",
        },
    ]

    for optimization in fuel_optimizations:
        with st.expander(
            f"{optimization['opportunity']} - Savings: {optimization['savings_potential']}"
        ):
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Savings Potential", optimization["savings_potential"])

            with col2:
                st.metric("Implementation Cost", optimization["implementation_cost"])

            with col3:
                st.metric("ROI Period", optimization["roi_period"])

            st.write(f"**Current Performance:** {optimization['current_performance']}")
            st.write(f"**Target Performance:** {optimization['target_performance']}")
            st.write(f"**Environmental Impact:** {optimization['environmental_impact']}")

            st.write("**Key Actions:**")
            for action in optimization["key_actions"]:
                st.write(f"✅ {action}")

            # Optimization implementation
            col1, col2 = st.columns(2)
            with col1:
                if st.button(
                    "📊 Analyze Impact",
                    key=f"analyze_{optimization['opportunity']}",
                ):
                    st.success(
                        f"Impact analysis started for {optimization['opportunity']}"
                    )
            with col2:
                if st.button(
                    "🚀 Implement", key=f"implement_{optimization['opportunity']}"
                ):
                    st.success(
                        f"Implementation initiated for {optimization['opportunity']}"
                    )


def render_asset_optimization(analytics, data_gen):
    """Render advanced asset optimization and strategic planning"""
    st.header("🚀 Asset Optimization & Strategic Planning")

    st.info(
        """
    **💡 Strategic Context:** Transform fleet management from operational function to strategic asset 
    optimization through right-sizing, lifecycle management, and total cost of ownership optimization.
    """
    )

    # Generate enhanced data
    fleet_data = generate_enhanced_fleet_data(data_gen)
    optimization_data = generate_enhanced_optimization_data(data_gen)

    # Optimization Intelligence
    st.subheader("📊 Optimization Intelligence")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        optimization_potential = calculate_optimization_potential(fleet_data)
        st.metric("Optimization Potential", f"{optimization_potential}%")
        st.caption("Fleet improvement opportunity")

    with col2:
        cost_savings = (
            optimization_data["current_fleet_cost"]
            - optimization_data["optimized_fleet_cost"]
        )
        st.metric("Potential Savings", f"KES {cost_savings:,.0f}")
        st.caption("Annual cost reduction")

    with col3:
        capacity_improvement = calculate_capacity_improvement(fleet_data)
        st.metric("Capacity Improvement", f"{capacity_improvement}%")
        st.caption("Operational capacity gain")

    with col4:
        roi_period = calculate_fleet_roi(optimization_data)
        st.metric("ROI Period", f"{roi_period} months")
        st.caption("Investment payback period")

    # Strategic Fleet Optimization
    st.subheader("🎯 Strategic Fleet Optimization")

    col1, col2 = st.columns(2)

    with col1:
        # Fleet right-sizing analysis
        sizing_scenarios = {
            "Scenario": ["Current", "Optimized", "Expanded"],
            "Vehicle Count": [
                len(fleet_data),
                len(fleet_data) - 4,
                len(fleet_data) + 3,
            ],
            "Utilization Rate": [
                fleet_data["utilization_rate"].mean(),
                88.5,
                82.0,
            ],
            "Total Cost": [1_850_000, 1_520_000, 2_100_000],
        }

        sizing_df = pd.DataFrame(sizing_scenarios)

        fig = px.bar(
            sizing_df,
            x="Scenario",
            y="Total Cost",
            title="💰 Fleet Sizing Scenarios - Cost Comparison",
            color="Utilization Rate",
            color_continuous_scale="RdYlGn",
            text=[f"KES {x:,.0f}" for x in sizing_scenarios["Total Cost"]],
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Capacity utilization analysis
        capacity_utilization = (
            fleet_data.groupby("vehicle_type")
            .agg({"capacity_kg": "sum", "utilization_rate": "mean"})
            .reset_index()
        )

        fig = px.scatter(
            capacity_utilization,
            x="capacity_kg",
            y="utilization_rate",
            size="utilization_rate",
            color="vehicle_type",
            title="📊 Capacity vs Utilization by Vehicle Type",
            size_max=40,
        )
        st.plotly_chart(fig, use_container_width=True)

    # Strategic Recommendations
    st.subheader("💡 Strategic Fleet Recommendations")

    strategic_recommendations = [
        {
            "category": "🚀 Fleet Right-Sizing",
            "impact": "18% cost reduction",
            "current_state": "4 underutilized large trucks, capacity mismatch",
            "recommendation": "Replace 3 large trucks with 5 medium trucks",
            "benefits": "Better capacity utilization, lower operating costs, improved flexibility",
            "investment": "KES 4.2M",
            "timeline": "3-6 months",
            "risk_level": "Low",
            "expected_roi": "42%",
        },
        {
            "category": "⚡ Electric Vehicle Transition",
            "impact": "25% operating cost reduction",
            "current_state": "100% diesel fleet, high fuel costs",
            "recommendation": "Pilot 2 electric trucks for urban routes",
            "benefits": "Lower fuel costs, environmental compliance, brand enhancement",
            "investment": "KES 8.5M",
            "timeline": "12-18 months",
            "risk_level": "Medium",
            "expected_roi": "35%",
        },
        {
            "category": "🔧 Predictive Maintenance Implementation",
            "impact": "40% reduction in unplanned downtime",
            "current_state": "Reactive maintenance causing operational disruptions",
            "recommendation": "Implement IoT sensors and AI-powered predictive maintenance",
            "benefits": "Higher reliability, lower repair costs, better planning",
            "investment": "KES 2.8M",
            "timeline": "6-9 months",
            "risk_level": "Low",
            "expected_roi": "55%",
        },
    ]

    for recommendation in strategic_recommendations:
        with st.expander(
            f"{recommendation['category']} - Impact: {recommendation['impact']}"
        ):
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Investment", recommendation["investment"])

            with col2:
                st.metric("Expected ROI", recommendation["expected_roi"])

            with col3:
                st.metric("Timeline", recommendation["timeline"])

            with col4:
                st.metric("Risk Level", recommendation["risk_level"])

            st.write(f"**Current State:** {recommendation['current_state']}")
            st.write(f"**Recommended Action:** {recommendation['recommendation']}")
            st.write(f"**Expected Benefits:** {recommendation['benefits']}")

            # Action buttons
            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button(
                    "📄 Accept Recommendation",
                    key=f"accept_{recommendation['category']}",
                ):
                    st.success(
                        f"Recommendation accepted: {recommendation['category']}"
                    )
            with col2:
                if st.button(
                    "💰 Create Business Case",
                    key=f"case_{recommendation['category']}",
                ):
                    st.success(
                        f"Business case created for {recommendation['category']}"
                    )
            with col3:
                if st.button(
                    "📅 Schedule Review",
                    key=f"review_{recommendation['category']}",
                ):
                    st.success(
                        f"Review scheduled for {recommendation['category']}"
                    )


def render_sustainability(analytics, data_gen):
    """Render sustainability and environmental impact analytics"""
    st.header("🌿 Sustainability & Environmental Intelligence")

    st.info(
        """
    **💡 Strategic Context:** Transform environmental compliance into competitive advantage through 
    carbon footprint reduction, sustainable operations, and green fleet initiatives.
    """
    )

    # Generate enhanced data
    fleet_data = generate_enhanced_fleet_data(data_gen)
    fuel_data = generate_enhanced_fuel_data(data_gen)

    # Sustainability Performance
    st.subheader("📊 Sustainability Performance")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        total_emissions = calculate_co2_emissions(fuel_data)
        st.metric("Total CO2 Emissions", f"{total_emissions:,.0f} kg")
        st.caption("Monthly carbon footprint")

    with col2:
        emission_intensity = (
            total_emissions / fleet_data["capacity_kg"].sum()
            if fleet_data["capacity_kg"].sum() > 0
            else 0
        )
        st.metric("Emission Intensity", f"{emission_intensity:.2f} kg/ton-km")
        st.caption("Operational efficiency")

    with col3:
        fuel_efficiency = fleet_data["fuel_efficiency"].mean()
        st.metric("Avg Fuel Efficiency", f"{fuel_efficiency:.1f} km/L")
        st.caption("Fleet efficiency")

    with col4:
        electric_ready = len(fleet_data[fleet_data["vehicle_type"] == "Electric"])
        st.metric("Electric Vehicles", f"{electric_ready}/35")
        st.caption("Green fleet adoption")

    # Sustainability Analytics
    st.subheader("🔍 Sustainability Analytics")

    col1, col2 = st.columns(2)

    with col1:
        # Emission trends
        monthly_emissions = (
            fuel_data.groupby(fuel_data["date"].dt.to_period("M"))
            .apply(lambda x: x["fuel_volume_liters"].sum() * 2.68)
            .reset_index(name="emissions")
        )
        monthly_emissions["date"] = monthly_emissions["date"].astype(str)

        fig = px.line(
            monthly_emissions,
            x="date",
            y="emissions",
            title="📈 Monthly CO2 Emission Trends",
            markers=True,
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Vehicle type emission analysis
        type_emissions = (
            fleet_data.groupby("vehicle_type")
            .agg({"fuel_efficiency": "mean", "vehicle_id": "count"})
            .reset_index()
        )
        type_emissions["emission_score"] = (
            10 - type_emissions["fuel_efficiency"]
        ) * 10

        fig = px.bar(
            type_emissions,
            x="vehicle_type",
            y="emission_score",
            title="🚚 Emission Impact by Vehicle Type",
            color="emission_score",
            color_continuous_scale="RdYlGn_r",
        )
        st.plotly_chart(fig, use_container_width=True)

    # Green Fleet Initiatives
    st.subheader("💡 Green Fleet Initiatives")

    sustainability_initiatives = [
        {
            "initiative": "⚡ Electric Vehicle Transition",
            "description": "Gradual replacement of diesel vehicles with electric alternatives",
            "environmental_impact": "60-70% emission reduction",
            "cost_impact": "25% lower operating costs",
            "implementation_timeline": "24-36 months",
            "investment_required": "KES 25M",
            "payback_period": "42 months",
        },
        {
            "initiative": "🌱 Biofuel Pilot Program",
            "description": "Test biodiesel blends in existing fleet vehicles",
            "environmental_impact": "15-20% emission reduction",
            "cost_impact": "5-8% higher fuel costs",
            "implementation_timeline": "6-9 months",
            "investment_required": "KES 1.2M",
            "payback_period": "N/A - Environmental focus",
        },
        {
            "initiative": "🔧 Aerodynamic Optimization",
            "description": "Install aerodynamic devices and optimize vehicle configurations",
            "environmental_impact": "8-12% fuel consumption reduction",
            "cost_impact": "3-5% lower fuel costs",
            "implementation_timeline": "3-4 months",
            "investment_required": "KES 850K",
            "payback_period": "18 months",
        },
    ]

    for initiative in sustainability_initiatives:
        with st.expander(
            f"{initiative['initiative']} - {initiative['environmental_impact']} impact"
        ):
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Investment", initiative["investment_required"])

            with col2:
                st.metric(
                    "Environmental Impact", initiative["environmental_impact"]
                )

            with col3:
                st.metric(
                    "Timeline", initiative["implementation_timeline"]
                )

            st.write(f"**Description:** {initiative['description']}")
            st.write(f"**Cost Impact:** {initiative['cost_impact']}")

            if (
                initiative["payback_period"]
                != "N/A - Environmental focus"
            ):
                st.write(
                    f"**Payback Period:** {initiative['payback_period']}"
                )

            # Initiative actions
            col1, col2 = st.columns(2)
            with col1:
                if st.button(
                    "📄 Develop Plan",
                    key=f"plan_{initiative['initiative']}",
                ):
                    st.success(
                        f"Development plan created for {initiative['initiative']}"
                    )
            with col2:
                if st.button(
                    "🌱 Implement",
                    key=f"implement_{initiative['initiative']}",
                ):
                    st.success(
                        f"Implementation started for {initiative['initiative']}"
                    )


# Enhanced Data Generation Functions
def generate_enhanced_fleet_data(data_gen):
    """Generate comprehensive enhanced fleet data"""
    np.random.seed(42)
    n_vehicles = 35

    fleet = []
    vehicle_types = [
        "Small Truck",
        "Medium Truck",
        "Large Truck",
        "Van",
        "Refrigerated Truck",
        "Electric Van",
    ]

    for i in range(n_vehicles):
        vehicle_type = np.random.choice(
            vehicle_types, p=[0.20, 0.30, 0.15, 0.15, 0.05, 0.15]
        )

        # Enhanced capacity mapping
        capacity_map = {
            "Van": 1500,
            "Electric Van": 1200,
            "Small Truck": 3500,
            "Medium Truck": 7500,
            "Large Truck": 12000,
            "Refrigerated Truck": 8000,
        }

        # Enhanced fuel efficiency based on type
        efficiency_map = {
            "Van": (6.0, 8.0),
            "Electric Van": (8.0, 12.0),  # km/kWh equivalent
            "Small Truck": (5.5, 7.5),
            "Medium Truck": (4.5, 6.5),
            "Large Truck": (3.5, 5.5),
            "Refrigerated Truck": (3.0, 5.0),
        }

        vehicle = {
            "vehicle_id": f"VH{1000 + i}",
            "license_plate": f"K{chr(65 + i % 26)}{np.random.randint(100, 999)}",
            "vehicle_type": vehicle_type,
            "capacity_kg": capacity_map[vehicle_type],
            "year": np.random.randint(2018, 2024),
            "purchase_price": np.random.randint(25000, 120000),
            "current_value": np.random.randint(15000, 80000),
            "utilization_rate": np.random.uniform(65, 92),
            "idle_time_percent": np.random.uniform(8, 30),
            "fuel_efficiency": np.random.uniform(*efficiency_map[vehicle_type]),
            "maintenance_cost_ytd": np.random.uniform(800, 6000),
            "reliability_score": np.random.uniform(75, 98),
            "last_maintenance": datetime.now()
            - timedelta(days=np.random.randint(1, 120)),
            "next_maintenance": datetime.now()
            + timedelta(days=np.random.randint(7, 90)),
            "status": np.random.choice(
                ["Active", "Maintenance", "Available", "Reserved"],
                p=[0.72, 0.08, 0.15, 0.05],
            ),
            "driver_id": f"DRV{np.random.randint(100, 125)}",
            "current_location": np.random.choice(
                ["Warehouse A", "Warehouse B", "Warehouse C", "On Route", "Depot"]
            ),
            "odometer_reading": np.random.randint(5000, 150000),
        }

        fleet.append(vehicle)

    return pd.DataFrame(fleet)


def generate_enhanced_utilization_data(data_gen):
    """Generate enhanced fleet utilization trend data"""
    np.random.seed(42)
    months = 12

    utilization_data = []

    for month in range(months):
        month_date = datetime.now() - timedelta(days=30 * (months - month - 1))
        utilization_data.append(
            {
                "month": month_date.strftime("%Y-%m"),
                "utilization_rate": np.random.uniform(70, 85),
                "active_vehicles": np.random.randint(25, 32),
                "total_distance": np.random.uniform(18000, 28000),
                "deliveries_completed": np.random.randint(1200, 2000),
                "fuel_consumption": np.random.uniform(4500, 6500),
                "maintenance_hours": np.random.uniform(120, 220),
            }
        )

    return pd.DataFrame(utilization_data)


def generate_enhanced_maintenance_data(data_gen):
    """Generate enhanced maintenance data"""
    np.random.seed(42)
    n_maintenance_records = 200

    maintenance_data = []
    maintenance_types = [
        "Routine Service",
        "Brake Repair",
        "Engine Maintenance",
        "Tire Replacement",
        "Electrical Repair",
    ]

    for i in range(n_maintenance_records):
        record = {
            "maintenance_id": f"MT{10000 + i}",
            "vehicle_id": f"VH{1000 + np.random.randint(0, 35)}",
            "maintenance_type": np.random.choice(maintenance_types),
            "date": datetime.now() - timedelta(days=np.random.randint(1, 365)),
            "cost": np.random.uniform(150, 2500),
            "duration_hours": np.random.uniform(2, 16),
            "technician": f"TECH{np.random.randint(1, 8)}",
            "parts_used": np.random.choice(
                ["Brake Pads", "Oil Filter", "Tires", "Battery", "Spark Plugs"],
                np.random.randint(1, 4),
            ),
            "status": np.random.choice(
                ["Completed", "In Progress", "Scheduled"],
                p=[0.8, 0.1, 0.1],
            ),
            "predictive_flag": np.random.choice(
                [True, False], p=[0.3, 0.7]
            ),
        }

        maintenance_data.append(record)

    return pd.DataFrame(maintenance_data)


def generate_enhanced_fuel_data(data_gen):
    """Generate enhanced fuel consumption data"""
    np.random.seed(42)
    n_fuel_records = 500

    fuel_data = []

    for i in range(n_fuel_records):
        record = {
            "fuel_id": f"FL{10000 + i}",
            "vehicle_id": f"VH{1000 + np.random.randint(0, 35)}",
            "date": datetime.now() - timedelta(days=np.random.randint(1, 90)),
            "fuel_volume_liters": np.random.uniform(30, 150),
            "fuel_cost": np.random.uniform(50, 400),
            "odometer_reading": np.random.randint(1000, 150000),
            "fuel_efficiency": np.random.uniform(5.0, 8.5),
            "driver_id": f"DRV{np.random.randint(100, 125)}",
            "route_id": f"RT{np.random.randint(100, 108)}",
            "fuel_station": np.random.choice(
                ["Station A", "Station B", "Station C", "Station D"]
            ),
        }

        fuel_data.append(record)

    return pd.DataFrame(fuel_data)


def generate_enhanced_optimization_data(data_gen):
    """Generate enhanced fleet optimization data"""
    np.random.seed(42)

    return {
        "current_fleet_cost": 1_850_000,
        "optimized_fleet_cost": 1_520_000,
        "vehicle_reduction": 4,
        "utilization_improvement": 18.5,
        "maintenance_savings": 45_000,
        "fuel_savings": 68_000,
        "sustainability_impact": 25,
        "implementation_timeline": "6-9 months",
    }


# Enhanced Analytical Functions
def calculate_maintenance_efficiency(maintenance_data):
    """Calculate maintenance efficiency score"""
    completed_on_time = len(
        maintenance_data[
            (maintenance_data["status"] == "Completed")
            & (maintenance_data["duration_hours"] < 8)
        ]
    )
    total_completed = len(maintenance_data[maintenance_data["status"] == "Completed"])

    return (completed_on_time / total_completed * 100) if total_completed > 0 else 0


def calculate_fuel_optimization_potential(fuel_data):
    """Calculate fuel optimization potential"""
    avg_efficiency = fuel_data["fuel_efficiency"].mean()
    max_efficiency = fuel_data["fuel_efficiency"].max()

    return (
        (max_efficiency - avg_efficiency) / avg_efficiency * 100
        if avg_efficiency > 0
        else 0
    )


def generate_strategic_fleet_alerts(fleet_data, maintenance_data):
    """Generate strategic fleet alerts with enhanced intelligence"""
    alerts = []

    # Low utilization alerts
    low_utilization = fleet_data[fleet_data["utilization_rate"] < 70]
    for _, vehicle in low_utilization.iterrows():
        alerts.append(
            {
                "vehicle_id": vehicle["vehicle_id"],
                "issue": f"Low Utilization ({vehicle['utilization_rate']:.1f}%)",
                "impact": f"KES {calculate_underutilization_cost(vehicle):,.0f} annual opportunity cost",
                "action": "Consider reassignment or route optimization",
                "priority": "🟠 High",
            }
        )

    # High maintenance cost alerts
    high_maintenance = fleet_data[fleet_data["maintenance_cost_ytd"] > 4000]
    for _, vehicle in high_maintenance.iterrows():
        alerts.append(
            {
                "vehicle_id": vehicle["vehicle_id"],
                "issue": f"High Maintenance Costs (KES {vehicle['maintenance_cost_ytd']:,.0f})",
                "impact": "Exceeds maintenance budget by 25-40%",
                "action": "Review maintenance strategy and consider replacement",
                "priority": "🔴 Critical",
            }
        )

    # Reliability concerns
    low_reliability = fleet_data[fleet_data["reliability_score"] < 80]
    for _, vehicle in low_reliability.iterrows():
        alerts.append(
            {
                "vehicle_id": vehicle["vehicle_id"],
                "issue": f"Low Reliability ({vehicle['reliability_score']:.1f}%)",
                "impact": "Increased risk of operational disruptions",
                "action": "Schedule comprehensive inspection",
                "priority": "🟠 High",
            }
        )

    return alerts[:5]  # Return top 5 alerts


def calculate_underutilization_cost(vehicle):
    """Calculate cost of vehicle underutilization"""
    capacity_value = vehicle["capacity_kg"] * 0.5  # KES per kg capacity
    utilization_gap = 75 - vehicle["utilization_rate"]  # Target 75%

    return max(0, capacity_value * utilization_gap * 0.01 * 365)


# Preserve existing analytical functions with enhancements
def calculate_avg_utilization(fleet_data):
    """Calculate average fleet utilization with enhanced logic"""
    return round(fleet_data["utilization_rate"].mean(), 1)


def calculate_total_capacity(fleet_data):
    """Calculate total fleet capacity"""
    return fleet_data["capacity_kg"].sum()


def calculate_total_maintenance_cost(maintenance_data):
    """Calculate total maintenance cost"""
    return round(maintenance_data["cost"].sum())


def calculate_downtime_percentage(fleet_data):
    """Calculate vehicle downtime percentage"""
    in_maintenance = len(fleet_data[fleet_data["status"] == "Maintenance"])
    return (
        round((in_maintenance / len(fleet_data)) * 100, 1)
        if len(fleet_data) > 0
        else 0
    )


def calculate_predictive_accuracy(maintenance_data):
    """Calculate predictive maintenance accuracy"""
    predictive_records = maintenance_data[maintenance_data["predictive_flag"] == True]
    if len(predictive_records) == 0:
        return 0
    accurate_predictions = len(predictive_records[predictive_records["cost"] < 1000])
    return round((accurate_predictions / len(predictive_records)) * 100, 1)


def calculate_maintenance_backlog(maintenance_data):
    """Calculate maintenance backlog"""
    scheduled = maintenance_data[maintenance_data["status"] == "Scheduled"]
    return len(scheduled)


def calculate_total_fuel_cost(fuel_data):
    """Calculate total fuel cost"""
    return round(fuel_data["fuel_cost"].sum())


def calculate_avg_fuel_efficiency(fleet_data):
    """Calculate average fuel efficiency"""
    return round(fleet_data["fuel_efficiency"].mean(), 1)


def calculate_fuel_wastage(fuel_data):
    """Calculate fuel wastage percentage (simulated)"""
    return round(np.random.uniform(8, 15), 1)


def calculate_co2_emissions(fuel_data):
    """Calculate CO2 emissions (kg)"""
    total_fuel = fuel_data["fuel_volume_liters"].sum()
    return round(total_fuel * 2.68)


def calculate_optimization_potential(fleet_data):
    """Calculate fleet optimization potential"""
    current_utilization = calculate_avg_utilization(fleet_data)
    return min(95 - current_utilization, 25)


def calculate_fleet_cost_savings(optimization_data):
    """Calculate potential fleet cost savings"""
    return (
        optimization_data["current_fleet_cost"]
        - optimization_data["optimized_fleet_cost"]
    )


def calculate_capacity_improvement(fleet_data):
    """Calculate capacity improvement potential (simulated)"""
    return round(np.random.uniform(15, 30), 1)


def calculate_fleet_roi(optimization_data):
    """Calculate fleet optimization ROI period (months)"""
    savings = calculate_fleet_cost_savings(optimization_data)
    investment = savings * 0.4
    monthly_savings = savings / 12 if savings > 0 else 1
    return max(6, round(investment / monthly_savings))


if __name__ == "__main__":
    render()
