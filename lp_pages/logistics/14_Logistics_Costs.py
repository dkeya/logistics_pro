# logistics_pro/pages/logistics/14_Logistics_Costs.py
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import numpy as np


def render():
    """💰 LOGISTICS COST INTELLIGENCE - Strategic Cost Optimization & Analysis"""

    st.title("💰 Logistics Cost Intelligence")
    st.markdown(
        f"""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                padding: 20px; border-radius: 12px; color: white; margin-bottom: 12px;">
        <h3 style="margin: 0; color: white;">🎯 Strategic Cost Optimization & Financial Intelligence</h3>
        <p style="margin: 5px 0 0 0; opacity: 0.9; color: white;">
        <strong>📍</strong> Logistics Intelligence &gt; Cost Analysis |
        <strong>🏢</strong> {st.session_state.get('current_tenant', 'ELORA Holding')}
        </p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # 💚 Soft green marquee strip – aligned with 01_Dashboard pattern
    st.markdown(
        """
    <div style="background: #dcfce7; padding: 10px 16px; border-radius: 10px;
                margin-bottom: 24px; border-left: 4px solid #16a34a;">
        <marquee behavior="scroll" direction="left" scrollamount="4"
                 style="font-size: 0.9rem; font-weight: 500; color: #166534;">
            💰 <strong>Logistics Cost Pulse:</strong> $2.8M Annual Spend • 
            🚚 <strong>Cost per Delivery:</strong> $18.45 | Trending ↓ 10.4% • 
            🛣️ <strong>Cost per KM:</strong> $1.82 | Route Efficiency +14% • 
            ⚙️ <strong>Fuel & Labor Share:</strong> 80% of Total Cost • 
            📦 <strong>Cost per KG:</strong> $0.85 | Network-Optimized • 
            📉 <strong>Optimization Headroom:</strong> ~15–18% Savings Potential • 
            🤖 <strong>AI Recommendations:</strong> Route Optimization • Fleet Right-Sizing • Shift & Crew Planning
        </marquee>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Data initialization and checks
    if "analytics" not in st.session_state:
        st.error("❌ Please initialize the application first")
        return

    analytics = st.session_state.analytics
    data_gen = st.session_state.data_gen

    # Main Tab Structure (4 tabs standard)
    tab1, tab2, tab3, tab4 = st.tabs(
        [
            "📊 Cost Dashboard",
            "🔍 Cost Analytics",
            "📈 Trend Analysis",
            "🎯 Cost Optimization",
        ]
    )

    with tab1:
        render_cost_dashboard(analytics, data_gen)
    with tab2:
        render_cost_analytics(analytics, data_gen)
    with tab3:
        render_trend_analysis(analytics, data_gen)
    with tab4:
        render_cost_optimization(analytics, data_gen)


def render_cost_dashboard(analytics, data_gen):
    """Tab 1: Comprehensive logistics cost dashboard"""

    # AI Insights expander
    with st.expander("🤖 AI COST OPTIMIZATION INSIGHTS", expanded=True):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "💰 Total Logistics Cost",
                "$2.8M",
                "-$245K",
                help="Annual logistics cost with optimization savings",
            )
        with col2:
            st.metric(
                "📦 Cost per Delivery",
                "$18.45",
                "-$2.15",
                help="Average cost per delivery with improvement",
            )
        with col3:
            st.metric(
                "🚚 Cost per KM",
                "$1.82",
                "-$0.23",
                help="Cost per kilometer with route optimization",
            )

        st.info(
            "💡 **AI Recommendation**: Focus on fuel cost optimization in Warehouse B "
            "routes. Implementing dynamic routing could save $85,000 annually with ~92% confidence."
        )

    # Generate cost data
    cost_data = generate_cost_data(data_gen)
    kpi_data = generate_kpi_data(data_gen)

    # Top KPIs
    st.subheader("🎯 Logistics Cost Performance Dashboard")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        total_annual_cost = calculate_total_annual_cost(cost_data)
        st.metric("Total Annual Cost", f"${total_annual_cost:,.0f}")

    with col2:
        cost_per_kg = calculate_cost_per_kg(cost_data)
        st.metric("Cost per KG", f"${cost_per_kg:.2f}")

    with col3:
        cost_vs_revenue = calculate_cost_vs_revenue(cost_data)
        st.metric("Cost/Revenue Ratio", f"{cost_vs_revenue}%")

    with col4:
        optimization_potential = calculate_optimization_potential(cost_data)
        st.metric("Optimization Potential", f"${optimization_potential:,.0f}")

    # Cost Distribution Overview
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("📊 Logistics Cost Distribution")
        distribution_fig = create_cost_distribution_chart(cost_data)
        st.plotly_chart(distribution_fig, use_container_width=True)

    with col2:
        st.subheader("🚨 Cost Alerts")
        display_cost_alerts(cost_data)

    # Cost Performance by Category
    st.subheader("📈 Cost Performance by Category")

    col1, col2 = st.columns(2)

    with col1:
        category_fig = create_category_cost_analysis(cost_data)
        st.plotly_chart(category_fig, use_container_width=True)

    with col2:
        warehouse_fig = create_warehouse_cost_analysis(cost_data)
        st.plotly_chart(warehouse_fig, use_container_width=True)

    # Real-time Cost Monitoring
    st.subheader("🔍 Real-time Cost Monitoring")
    display_realtime_cost_monitor(cost_data)


def render_cost_analytics(analytics, data_gen):
    """Tab 2: Detailed cost analytics and root cause analysis"""

    st.subheader("🔍 Detailed Cost Analytics & Root Cause Analysis")

    cost_data = generate_cost_data(data_gen)
    detailed_data = generate_detailed_cost_data(data_gen)

    # Cost Analysis Metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        fuel_cost_ratio = calculate_fuel_cost_ratio(cost_data)
        st.metric("Fuel Cost Ratio", f"{fuel_cost_ratio}%")

    with col2:
        labor_cost_ratio = calculate_labor_cost_ratio(cost_data)
        st.metric("Labor Cost Ratio", f"{labor_cost_ratio}%")

    with col3:
        maintenance_ratio = calculate_maintenance_ratio(cost_data)
        st.metric("Maintenance Ratio", f"{maintenance_ratio}%")

    with col4:
        overhead_ratio = calculate_overhead_ratio(cost_data)
        st.metric("Overhead Ratio", f"{overhead_ratio}%")

    # Cost Driver Analysis
    st.subheader("📊 Cost Driver Analysis")

    col1, col2 = st.columns(2)

    with col1:
        driver_analysis_fig = create_cost_driver_analysis(detailed_data)
        st.plotly_chart(driver_analysis_fig, use_container_width=True)

    with col2:
        variance_analysis_fig = create_cost_variance_analysis(detailed_data)
        st.plotly_chart(variance_analysis_fig, use_container_width=True)

    # Root Cause Analysis
    st.subheader("🔍 Cost Root Cause Analysis")

    root_causes = analyze_cost_root_causes(cost_data)
    for cause, impact in root_causes.items():
        st.progress(impact / 100, text=f"{cause}: {impact}% impact")

    # Cost Benchmarking
    st.subheader("🏆 Cost Performance Benchmarking")

    benchmarks = generate_cost_benchmarks()

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Your Cost/KG", f"${calculate_cost_per_kg(cost_data):.2f}")
    with col2:
        st.metric("Industry Average", f"${benchmarks['industry_avg']:.2f}")
    with col3:
        st.metric("Best in Class", f"${benchmarks['best_in_class']:.2f}")

    # Detailed Cost Breakdown
    st.subheader("📋 Detailed Cost Breakdown")

    cost_breakdown = generate_detailed_breakdown(cost_data)
    display_detailed_cost_table(cost_breakdown)


def render_trend_analysis(analytics, data_gen):
    """Tab 3: Cost trend analysis and forecasting"""

    st.subheader("📈 Cost Trend Analysis & Forecasting")

    cost_data = generate_cost_data(data_gen)
    trend_data = generate_trend_data(data_gen)

    # Trend Analysis Metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        yoy_growth = calculate_yoy_growth(trend_data)
        st.metric("YoY Cost Growth", f"{yoy_growth}%")

    with col2:
        mom_change = calculate_mom_change(trend_data)
        st.metric("MoM Change", f"{mom_change}%")

    with col3:
        inflation_impact = calculate_inflation_impact(trend_data)
        st.metric("Inflation Impact", f"{inflation_impact}%")

    with col4:
        efficiency_gain = calculate_efficiency_gain(trend_data)
        st.metric("Efficiency Gain", f"{efficiency_gain}%")

    # Cost Trends Over Time
    st.subheader("📊 Cost Trends Over Time")

    col1, col2 = st.columns(2)

    with col1:
        trend_fig = create_cost_trend_chart(trend_data)
        st.plotly_chart(trend_fig, use_container_width=True)

    with col2:
        forecast_fig = create_cost_forecast_chart(trend_data)
        st.plotly_chart(forecast_fig, use_container_width=True)

    # Seasonal Analysis
    st.subheader("🌍 Seasonal Cost Patterns")

    seasonal_fig = create_seasonal_analysis(trend_data)
    st.plotly_chart(seasonal_fig, use_container_width=True)

    # Cost Forecasting
    st.subheader("🔮 Cost Forecasting & Projections")

    with st.form("cost_forecasting"):
        col1, col2 = st.columns(2)

        with col1:
            forecast_period = st.selectbox(
                "Forecast Period", ["3 months", "6 months", "12 months", "24 months"]
            )
            confidence_level = st.slider("Confidence Level", 80, 95, 90)
            include_inflation = st.checkbox(
                "Include Inflation Projection", value=True
            )

        with col2:
            growth_assumption = st.slider(
                "Growth Assumption (%)", -10, 20, 5
            )
            efficiency_improvement = st.slider(
                "Efficiency Improvement (%)", 0, 15, 5
            )
            scenario_analysis = st.selectbox(
                "Scenario Analysis",
                ["Base Case", "Optimistic", "Pessimistic"],
            )

        forecast = st.form_submit_button("🚀 Generate Cost Forecast")

        if forecast:
            forecast_results = generate_cost_forecast(
                trend_data,
                forecast_period,
                confidence_level,
                include_inflation,
                growth_assumption,
                efficiency_improvement,
                scenario_analysis,
            )
            display_forecast_results(forecast_results)


def render_cost_optimization(analytics, data_gen):
    """Tab 4: Cost optimization strategies and implementation"""

    st.subheader("🎯 Cost Optimization Strategies & Implementation")

    cost_data = generate_cost_data(data_gen)
    optimization_data = generate_optimization_data(data_gen)

    # Optimization Metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        total_savings_potential = calculate_total_savings_potential(optimization_data)
        st.metric("Total Savings Potential", f"${total_savings_potential:,.0f}")

    with col2:
        quick_wins = calculate_quick_wins(optimization_data)
        st.metric("Quick Wins", f"${quick_wins:,.0f}")

    with col3:
        strategic_initiatives = calculate_strategic_initiatives(optimization_data)
        st.metric("Strategic Initiatives", f"${strategic_initiatives:,.0f}")

    with col4:
        implementation_timeline = calculate_implementation_timeline(optimization_data)
        st.metric("Implementation Timeline", f"{implementation_timeline} months")

    # Optimization Opportunities
    st.subheader("💡 Cost Optimization Opportunities")

    optimization_opportunities = identify_optimization_opportunities(cost_data)

    for opportunity in optimization_opportunities:
        with st.expander(
            f"🎯 {opportunity['category']} - {opportunity['savings_potential']}"
        ):
            col1, col2 = st.columns(2)

            with col1:
                st.write(f"**Current Cost**: {opportunity['current_cost']}")
                st.write(f"**Target Cost**: {opportunity['target_cost']}")
                st.write(f"**Key Initiatives**: {opportunity['initiatives']}")

            with col2:
                st.write(
                    f"**Implementation Complexity**: {opportunity['complexity']}"
                )
                st.write(f"**Confidence Level**: {opportunity['confidence']}")
                st.write(f"**ROI Period**: {opportunity['roi_period']}")

            # Progress tracking
            if "progress" in opportunity:
                st.write(
                    f"**Implementation Progress**: {opportunity['progress']}%"
                )
                st.progress(opportunity["progress"] / 100)

            # Action buttons
            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button(
                    "Start Implementation",
                    key=f"start_{opportunity['category']}",
                ):
                    st.success(
                        f"Implementation started for {opportunity['category']}"
                    )
            with col2:
                if st.button(
                    "Create Business Case",
                    key=f"case_{opportunity['category']}",
                ):
                    st.info(
                        f"Business case created for {opportunity['category']}"
                    )
            with col3:
                if st.button(
                    "Schedule Review",
                    key=f"review_{opportunity['category']}",
                ):
                    st.info(
                        f"Review scheduled for {opportunity['category']}"
                    )

    # Cost Reduction Initiatives
    st.subheader("🚀 Active Cost Reduction Initiatives")

    active_initiatives = get_active_initiatives(optimization_data)

    for initiative in active_initiatives:
        with st.expander(
            f"📋 {initiative['name']} - {initiative['status']}"
        ):
            col1, col2 = st.columns(2)

            with col1:
                st.write(f"**Objective**: {initiative['objective']}")
                st.write(f"**Scope**: {initiative['scope']}")
                st.write(
                    f"**Target Savings**: {initiative['target_savings']}"
                )
                st.write(f"**Timeline**: {initiative['timeline']}")

            with col2:
                st.write(f"**Budget**: {initiative['budget']}")
                st.write(f"**Resources**: {initiative['resources']}")
                st.write(f"**Risks**: {initiative['risks']}")
                st.write(f"**Progress**: {initiative['progress']}%")

            st.progress(initiative["progress"] / 100)

    # Cost Optimization Simulation
    st.subheader("🛠️ Cost Optimization Simulation")

    with st.form("optimization_simulation"):
        col1, col2 = st.columns(2)

        with col1:
            fuel_cost_reduction = st.slider(
                "Fuel Cost Reduction (%)", 0, 30, 15
            )
            labor_optimization = st.slider(
                "Labor Optimization (%)", 0, 25, 10
            )
            route_efficiency = st.slider(
                "Route Efficiency Gain (%)", 0, 35, 20
            )

        with col2:
            maintenance_savings = st.slider(
                "Maintenance Savings (%)", 0, 20, 8
            )
            technology_investment = st.number_input(
                "Technology Investment ($)", 0, 500_000, 150_000
            )
            implementation_period = st.selectbox(
                "Implementation Period",
                ["6 months", "12 months", "18 months", "24 months"],
            )

        simulate = st.form_submit_button("🚀 Run Optimization Simulation")

        if simulate:
            simulation_results = run_optimization_simulation(
                cost_data,
                fuel_cost_reduction,
                labor_optimization,
                route_efficiency,
                maintenance_savings,
                technology_investment,
                implementation_period,
            )
            display_simulation_results(simulation_results)


# Data Generation Functions
def generate_cost_data(data_gen):
    """Generate comprehensive logistics cost data"""

    np.random.seed(42)

    # Annual cost data
    annual_costs = {
        "fuel_cost": 985_000,
        "labor_cost": 1_250_000,
        "maintenance_cost": 320_000,
        "vehicle_cost": 185_000,
        "insurance_cost": 85_000,
        "overhead_cost": 150_000,
        "technology_cost": 75_000,
        "other_costs": 45_000,
    }

    annual_costs["total_cost"] = sum(annual_costs.values())

    # Add calculated metrics
    annual_costs["cost_per_kg"] = 0.85
    annual_costs["cost_per_delivery"] = 18.45
    annual_costs["cost_per_km"] = 1.82
    annual_costs["cost_vs_revenue_ratio"] = 8.2

    return annual_costs


def generate_kpi_data(data_gen):
    """Generate KPI performance data"""

    np.random.seed(42)
    months = 12

    kpi_data = []

    for month in range(months):
        kpi = {
            "month": (
                datetime.now()
                - timedelta(days=30 * (months - month - 1))
            ).strftime("%Y-%m"),
            "total_cost": np.random.uniform(200_000, 280_000),
            "deliveries": np.random.randint(12_000, 18_000),
            "distance_km": np.random.uniform(120_000, 180_000),
            "weight_kg": np.random.uniform(800_000, 1_200_000),
        }

        kpi["cost_per_delivery"] = kpi["total_cost"] / kpi["deliveries"]
        kpi["cost_per_km"] = kpi["total_cost"] / kpi["distance_km"]
        kpi["cost_per_kg"] = kpi["total_cost"] / kpi["weight_kg"]

        kpi_data.append(kpi)

    return pd.DataFrame(kpi_data)


def generate_detailed_cost_data(data_gen):
    """Generate detailed cost analysis data"""

    np.random.seed(42)

    cost_categories = [
        "Fuel",
        "Labor",
        "Maintenance",
        "Vehicles",
        "Insurance",
        "Overhead",
        "Technology",
        "Other",
    ]
    cost_drivers = [
        "Price Increase",
        "Volume Growth",
        "Inefficiency",
        "External Factors",
        "Process Issues",
    ]

    detailed_data = []

    for category in cost_categories:
        for driver in cost_drivers:
            detailed_data.append(
                {
                    "category": category,
                    "cost_driver": driver,
                    "impact_percentage": np.random.uniform(5, 25),
                    "variance": np.random.uniform(-15, 15),
                    "controllable": np.random.choice(
                        [True, False], p=[0.7, 0.3]
                    ),
                }
            )

    return pd.DataFrame(detailed_data)


def generate_trend_data(data_gen):
    """Generate cost trend data"""

    np.random.seed(42)
    months = 24  # 2 years of data

    trend_data = []
    base_cost = 220_000

    for month in range(months):
        trend = {
            "month": (
                datetime.now()
                - timedelta(days=30 * (months - month - 1))
            ).strftime("%Y-%m"),
            "total_cost": base_cost
            * (1 + month * 0.02 + np.random.uniform(-0.05, 0.05)),
            "fuel_cost": base_cost
            * 0.35
            * (1 + month * 0.025 + np.random.uniform(-0.08, 0.08)),
            "labor_cost": base_cost
            * 0.45
            * (1 + month * 0.015 + np.random.uniform(-0.03, 0.03)),
            "maintenance_cost": base_cost
            * 0.12
            * (1 + month * 0.01 + np.random.uniform(-0.1, 0.1)),
            "inflation_rate": 2.5 + np.random.uniform(-1, 1),
        }

        trend_data.append(trend)

    return pd.DataFrame(trend_data)


def generate_optimization_data(data_gen):
    """Generate optimization data"""

    np.random.seed(42)

    return {
        "total_savings_potential": 485_000,
        "quick_wins": 125_000,
        "strategic_initiatives": 360_000,
        "implementation_timeline": 18,
        "initiatives": [
            {
                "name": "Route Optimization Program",
                "status": "In Progress",
                "objective": "Reduce fuel and labor costs through optimized routing",
                "scope": "All delivery routes",
                "target_savings": "$85,000",
                "timeline": "6 months",
                "budget": "$45,000",
                "resources": "Project team, routing software",
                "risks": "Driver adoption, technology integration",
                "progress": 65,
            },
            {
                "name": "Fleet Right-Sizing Initiative",
                "status": "Planning",
                "objective": "Optimize vehicle mix for better utilization",
                "scope": "Entire fleet",
                "target_savings": "$120,000",
                "timeline": "12 months",
                "budget": "$180,000",
                "resources": "Fleet managers, analysts",
                "risks": "Capital investment, implementation timing",
                "progress": 25,
            },
        ],
    }


# Analytical Functions
def calculate_total_annual_cost(cost_data):
    """Calculate total annual logistics cost"""
    return cost_data["total_cost"]


def calculate_cost_per_kg(cost_data):
    """Calculate cost per kilogram"""
    return cost_data["cost_per_kg"]


def calculate_cost_vs_revenue(cost_data):
    """Calculate cost vs revenue ratio"""
    return cost_data["cost_vs_revenue_ratio"]


def calculate_optimization_potential(cost_data):
    """Calculate optimization potential"""
    return round(cost_data["total_cost"] * 0.15)  # 15% savings potential


def calculate_fuel_cost_ratio(cost_data):
    """Calculate fuel cost ratio"""
    return round((cost_data["fuel_cost"] / cost_data["total_cost"]) * 100, 1)


def calculate_labor_cost_ratio(cost_data):
    """Calculate labor cost ratio"""
    return round((cost_data["labor_cost"] / cost_data["total_cost"]) * 100, 1)


def calculate_maintenance_ratio(cost_data):
    """Calculate maintenance cost ratio"""
    return round(
        (cost_data["maintenance_cost"] / cost_data["total_cost"]) * 100, 1
    )


def calculate_overhead_ratio(cost_data):
    """Calculate overhead cost ratio"""
    return round(
        (cost_data["overhead_cost"] / cost_data["total_cost"]) * 100, 1
    )


def calculate_yoy_growth(trend_data):
    """Calculate year-over-year growth"""
    latest = trend_data.iloc[-1]["total_cost"]
    if len(trend_data) >= 12:
        year_ago = trend_data.iloc[-12]["total_cost"]
    else:
        year_ago = trend_data.iloc[0]["total_cost"]
    return round(((latest - year_ago) / year_ago) * 100, 1)


def calculate_mom_change(trend_data):
    """Calculate month-over-month change"""
    if len(trend_data) < 2:
        return 0
    latest = trend_data.iloc[-1]["total_cost"]
    previous = trend_data.iloc[-2]["total_cost"]
    return round(((latest - previous) / previous) * 100, 1)


def calculate_inflation_impact(trend_data):
    """Calculate inflation impact (simulated)"""
    return round(np.random.uniform(1.5, 3.5), 1)


def calculate_efficiency_gain(trend_data):
    """Calculate efficiency gain (simulated)"""
    return round(np.random.uniform(2.5, 8.5), 1)


def calculate_total_savings_potential(optimization_data):
    """Calculate total savings potential"""
    return optimization_data["total_savings_potential"]


def calculate_quick_wins(optimization_data):
    """Calculate quick wins savings"""
    return optimization_data["quick_wins"]


def calculate_strategic_initiatives(optimization_data):
    """Calculate strategic initiatives savings"""
    return optimization_data["strategic_initiatives"]


def calculate_implementation_timeline(optimization_data):
    """Calculate implementation timeline"""
    return optimization_data["implementation_timeline"]


# Visualization Functions
def create_cost_distribution_chart(cost_data):
    """Create cost distribution chart"""
    categories = [
        "Fuel",
        "Labor",
        "Maintenance",
        "Vehicles",
        "Insurance",
        "Overhead",
        "Technology",
        "Other",
    ]
    values = [
        cost_data["fuel_cost"],
        cost_data["labor_cost"],
        cost_data["maintenance_cost"],
        cost_data["vehicle_cost"],
        cost_data["insurance_cost"],
        cost_data["overhead_cost"],
        cost_data["technology_cost"],
        cost_data["other_costs"],
    ]

    fig = px.pie(
        values=values,
        names=categories,
        title="Logistics Cost Distribution",
        color_discrete_sequence=px.colors.sequential.RdBu,
    )

    return fig


def create_category_cost_analysis(cost_data):
    """Create category cost analysis chart"""
    categories = [
        "Fuel",
        "Labor",
        "Maintenance",
        "Vehicles",
        "Insurance",
        "Overhead",
        "Technology",
        "Other",
    ]
    values = [
        cost_data["fuel_cost"],
        cost_data["labor_cost"],
        cost_data["maintenance_cost"],
        cost_data["vehicle_cost"],
        cost_data["insurance_cost"],
        cost_data["overhead_cost"],
        cost_data["technology_cost"],
        cost_data["other_costs"],
    ]

    fig = px.bar(
        x=categories,
        y=values,
        title="Cost by Category",
        color=values,
        color_continuous_scale="RdYlGn_r",
    )

    return fig


def create_warehouse_cost_analysis(cost_data):
    """Create warehouse cost analysis chart (simulated)"""
    warehouses = ["WH-A", "WH-B", "WH-C", "WH-D"]
    costs = [650_000, 720_000, 580_000, 850_000]
    efficiency = [88, 82, 91, 76]

    fig = px.scatter(
        x=warehouses,
        y=costs,
        size=efficiency,
        title="Cost vs Efficiency by Warehouse",
        color=efficiency,
        color_continuous_scale="RdYlGn",
        labels={"x": "Warehouse", "y": "Cost ($)"},
    )

    return fig


def create_cost_driver_analysis(detailed_data):
    """Create cost driver analysis chart"""
    driver_impact = (
        detailed_data.groupby("cost_driver")["impact_percentage"]
        .sum()
        .reset_index()
    )

    fig = px.bar(
        driver_impact,
        x="cost_driver",
        y="impact_percentage",
        title="Cost Driver Impact Analysis",
        color="impact_percentage",
        color_continuous_scale="RdYlGn_r",
        labels={"impact_percentage": "Impact (%)", "cost_driver": "Cost Driver"},
    )

    return fig


def create_cost_variance_analysis(detailed_data):
    """Create cost variance analysis chart"""
    controllable = detailed_data[detailed_data["controllable"] == True]
    non_controllable = detailed_data[detailed_data["controllable"] == False]

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            name="Controllable",
            x=controllable["category"],
            y=controllable["variance"],
        )
    )
    fig.add_trace(
        go.Bar(
            name="Non-Controllable",
            x=non_controllable["category"],
            y=non_controllable["variance"],
        )
    )

    fig.update_layout(
        title="Cost Variance Analysis",
        barmode="stack",
        xaxis_title="Category",
        yaxis_title="Variance (%)",
    )

    return fig


def create_cost_trend_chart(trend_data):
    """Create cost trend chart"""
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=trend_data["month"],
            y=trend_data["total_cost"],
            name="Total Cost",
            line=dict(color="#1f77b4", width=3),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=trend_data["month"],
            y=trend_data["fuel_cost"],
            name="Fuel Cost",
            line=dict(color="#ff7f0e", width=2),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=trend_data["month"],
            y=trend_data["labor_cost"],
            name="Labor Cost",
            line=dict(color="#2ca02c", width=2),
        )
    )

    fig.update_layout(
        title="Cost Trends Over Time",
        xaxis_title="Month",
        yaxis_title="Cost ($)",
        hovermode="x unified",
    )

    return fig


def create_cost_forecast_chart(trend_data):
    """Create cost forecast chart (6 months ahead)"""
    # Simulate forecast data
    last_12 = trend_data.tail(12)
    forecast_months = 6

    # Convert month strings to period then to timestamps
    last_month_str = last_12["month"].iloc[-1]
    last_month_period = pd.Period(last_month_str, freq="M")
    forecast_periods = [
        last_month_period + i for i in range(1, forecast_months + 1)
    ]
    forecast_dates = [p.to_timestamp() for p in forecast_periods]

    forecast_costs = []
    base_cost = last_12["total_cost"].iloc[-1]
    for i in range(forecast_months):
        forecast_cost = base_cost * (1 + 0.015) ** (i + 1)  # 1.5% monthly growth
        forecast_costs.append(forecast_cost)

    fig = go.Figure()

    # Historical data
    fig.add_trace(
        go.Scatter(
            x=last_12["month"],
            y=last_12["total_cost"],
            name="Historical",
            line=dict(color="#1f77b4"),
        )
    )

    # Forecast data
    fig.add_trace(
        go.Scatter(
            x=[d.strftime("%Y-%m") for d in forecast_dates],
            y=forecast_costs,
            name="Forecast",
            line=dict(color="#ff7f0e", dash="dash"),
        )
    )

    fig.update_layout(
        title="Cost Forecast (6 Months)",
        xaxis_title="Month",
        yaxis_title="Cost ($)",
    )

    return fig


def create_seasonal_analysis(trend_data):
    """Create seasonal analysis chart (simulated)"""
    months = [
        "Jan",
        "Feb",
        "Mar",
        "Apr",
        "May",
        "Jun",
        "Jul",
        "Aug",
        "Sep",
        "Oct",
        "Nov",
        "Dec",
    ]
    seasonal_factor = [
        1.05,
        1.02,
        1.08,
        1.1,
        1.12,
        1.15,
        1.18,
        1.16,
        1.12,
        1.08,
        1.05,
        1.1,
    ]

    fig = px.line(
        x=months,
        y=seasonal_factor,
        title="Seasonal Cost Pattern",
        markers=True,
        labels={"x": "Month", "y": "Seasonal Factor"},
    )

    return fig


# Display Functions
def display_cost_alerts(cost_data):
    """Display cost performance alerts"""
    high_cost_categories = []

    if cost_data["fuel_cost"] > 1_000_000:
        high_cost_categories.append("Fuel")
    if cost_data["labor_cost"] > 1_300_000:
        high_cost_categories.append("Labor")
    if cost_data["maintenance_cost"] > 350_000:
        high_cost_categories.append("Maintenance")

    if len(high_cost_categories) == 0:
        st.success("✅ No critical cost alerts")
        return

    for category in high_cost_categories:
        st.error(f"🔴 {category} Cost Alert")
        st.caption(
            "Above target threshold – review optimization opportunities."
        )


def display_realtime_cost_monitor(cost_data):
    """Display real-time cost monitoring (simulated)"""
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("MTD Cost", "$245,800")
    with col2:
        st.metric("vs Budget", "+2.8%", delta_color="inverse")
    with col3:
        st.metric("Forecast Variance", "-1.2%")
    with col4:
        st.metric("Cost Avoidance", "$18,500")


def display_detailed_cost_table(cost_breakdown):
    """Display detailed cost breakdown table"""
    st.dataframe(cost_breakdown, use_container_width=True)


def display_forecast_results(forecast_results):
    """Display cost forecast results"""
    st.success("🔮 Cost Forecast Generated!")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Projected Growth", f"{forecast_results['projected_growth']}%")
        st.metric(
            "Confidence Interval",
            f"±{forecast_results['confidence_interval']}%",
        )

    with col2:
        st.metric(
            "Total Projected Cost",
            f"${forecast_results['total_projected']:,.0f}",
        )
        st.metric(
            "Efficiency Impact",
            f"{forecast_results['efficiency_impact']}%",
        )

    with col3:
        st.metric("Risk Level", forecast_results["risk_level"])
        st.metric("Recommendation", forecast_results["recommendation"])


def display_simulation_results(simulation_results):
    """Display optimization simulation results"""
    st.success("🎯 Optimization Simulation Complete!")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Annual Savings",
            f"${simulation_results['annual_savings']:,.0f}",
        )
        st.metric("ROI Period", f"{simulation_results['roi_period']} months")

    with col2:
        st.metric(
            "Total Impact", f"{simulation_results['total_impact']}%"
        )
        st.metric(
            "Payback Period",
            f"{simulation_results['payback_period']} months",
        )

    with col3:
        st.metric(
            "Implementation Risk", simulation_results["implementation_risk"]
        )
        st.metric(
            "Confidence Level", f"{simulation_results['confidence']}%"
        )


# Analysis Functions
def analyze_cost_root_causes(cost_data):
    """Analyze cost root causes (simulated)"""
    return {
        "Fuel Price Volatility": 35,
        "Route Inefficiency": 25,
        "Labor Productivity": 18,
        "Vehicle Maintenance": 12,
        "Administrative Overhead": 10,
    }


def generate_cost_benchmarks():
    """Generate cost benchmark data (simulated)"""
    return {
        "industry_avg": 0.92,
        "best_in_class": 0.68,
        "top_quartile": 0.78,
        "bottom_quartile": 1.15,
    }


def generate_detailed_breakdown(cost_data):
    """Generate detailed cost breakdown"""
    breakdown = [
        {
            "Category": "Fuel",
            "Amount": f"${cost_data['fuel_cost']:,.0f}",
            "Percentage": f"{calculate_fuel_cost_ratio(cost_data)}%",
        },
        {
            "Category": "Labor",
            "Amount": f"${cost_data['labor_cost']:,.0f}",
            "Percentage": f"{calculate_labor_cost_ratio(cost_data)}%",
        },
        {
            "Category": "Maintenance",
            "Amount": f"${cost_data['maintenance_cost']:,.0f}",
            "Percentage": f"{calculate_maintenance_ratio(cost_data)}%",
        },
        {
            "Category": "Vehicles",
            "Amount": f"${cost_data['vehicle_cost']:,.0f}",
            "Percentage": f"{round((cost_data['vehicle_cost']/cost_data['total_cost'])*100, 1)}%",
        },
        {
            "Category": "Insurance",
            "Amount": f"${cost_data['insurance_cost']:,.0f}",
            "Percentage": f"{round((cost_data['insurance_cost']/cost_data['total_cost'])*100, 1)}%",
        },
        {
            "Category": "Overhead",
            "Amount": f"${cost_data['overhead_cost']:,.0f}",
            "Percentage": f"{calculate_overhead_ratio(cost_data)}%",
        },
        {
            "Category": "Technology",
            "Amount": f"${cost_data['technology_cost']:,.0f}",
            "Percentage": f"{round((cost_data['technology_cost']/cost_data['total_cost'])*100, 1)}%",
        },
        {
            "Category": "Other",
            "Amount": f"${cost_data['other_costs']:,.0f}",
            "Percentage": f"{round((cost_data['other_costs']/cost_data['total_cost'])*100, 1)}%",
        },
    ]

    return pd.DataFrame(breakdown)


def generate_cost_forecast(
    trend_data,
    forecast_period,
    confidence_level,
    include_inflation,
    growth_assumption,
    efficiency_improvement,
    scenario_analysis,
):
    """Generate cost forecast (simplified)"""
    latest_cost = trend_data["total_cost"].iloc[-1]
    total_projected = latest_cost * (1 + growth_assumption / 100)

    return {
        "projected_growth": growth_assumption,
        "confidence_interval": 100 - confidence_level,
        "total_projected": total_projected,
        "efficiency_impact": efficiency_improvement,
        "risk_level": "Medium",
        "recommendation": "Implement efficiency measures to offset growth.",
    }


def identify_optimization_opportunities(cost_data):
    """Identify optimization opportunities"""
    return [
        {
            "category": "Fuel Cost Optimization",
            "savings_potential": "$85,000",
            "current_cost": f"${cost_data['fuel_cost']:,.0f}",
            "target_cost": f"${cost_data['fuel_cost'] - 85_000:,.0f}",
            "initiatives": "Dynamic routing, driver training, fleet modernization",
            "complexity": "Medium",
            "confidence": "85%",
            "roi_period": "8 months",
            "progress": 45,
        },
        {
            "category": "Labor Productivity",
            "savings_potential": "$65,000",
            "current_cost": f"${cost_data['labor_cost']:,.0f}",
            "target_cost": f"${cost_data['labor_cost'] - 65_000:,.0f}",
            "initiatives": "Route optimization, shift planning, performance incentives",
            "complexity": "High",
            "confidence": "75%",
            "roi_period": "12 months",
            "progress": 25,
        },
    ]


def get_active_initiatives(optimization_data):
    """Get active cost reduction initiatives"""
    return optimization_data["initiatives"]


def run_optimization_simulation(
    cost_data,
    fuel_cost_reduction,
    labor_optimization,
    route_efficiency,
    maintenance_savings,
    technology_investment,
    implementation_period,
):
    """Run optimization simulation"""
    # route_efficiency is captured conceptually but not directly used in this simple model
    annual_savings = (
        cost_data["fuel_cost"] * (fuel_cost_reduction / 100)
        + cost_data["labor_cost"] * (labor_optimization / 100)
        + cost_data["maintenance_cost"] * (maintenance_savings / 100)
    )

    if annual_savings <= 0:
        # Avoid division by zero and return neutral outputs
        return {
            "annual_savings": 0,
            "roi_period": 0,
            "total_impact": 0.0,
            "payback_period": 0,
            "implementation_risk": "High",
            "confidence": 50,
        }

    roi_period_months = max(6, round(technology_investment / (annual_savings / 12)))
    payback_period = round(technology_investment / (annual_savings / 12))

    return {
        "annual_savings": round(annual_savings),
        "roi_period": roi_period_months,
        "total_impact": round((annual_savings / cost_data["total_cost"]) * 100, 1),
        "payback_period": payback_period,
        "implementation_risk": "Medium",
        "confidence": 80,
    }


if __name__ == "__main__":
    render()
