# -*- coding: utf-8 -*-
# logistics_pro/pages/procurement/16_Procurement_Costs.py

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import numpy as np


def render():
    """💰 PROCUREMENT COST INTELLIGENCE - Strategic Spend Optimization & Analysis"""

    st.title("💰 Procurement Cost Intelligence")

    # 🌈 Gradient hero header (aligned with 01_Dashboard style)
    st.markdown(
        f"""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                padding: 20px; border-radius: 12px; color: white; margin-bottom: 12px;">
        <h3 style="margin: 0; color: white;">🎯 Strategic Spend Optimization & Financial Intelligence</h3>
        <p style="margin: 5px 0 0 0; opacity: 0.9; color: white;">
            <strong>📍</strong> Procurement Intelligence &gt; Cost Analysis |
            <strong>🏢</strong> {st.session_state.get('current_tenant', 'ELORA Holding')} |
            <strong>📅</strong> {datetime.now().strftime('%Y-%m-%d')}
        </p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # 💚 Soft green marquee strip – aligned with Executive Cockpit pattern
    st.markdown(
        """
    <div style="background: #dcfce7; padding: 10px 16px; border-radius: 10px;
                margin-bottom: 24px; border-left: 4px solid #16a34a;">
        <marquee behavior="scroll" direction="left" scrollamount="4"
                 style="font-size: 0.9rem; font-weight: 500; color: #166534;">
            💰 <strong>Procurement Intelligence:</strong> Total Annual Spend ≈ $8.2M • 
            📉 <strong>Savings Rate:</strong> 14.8% realized vs target 12% • 
            🚨 <strong>Maverick Spend:</strong> Controlled under 8% of total spend • 
            📜 <strong>Contract Compliance:</strong> &gt; 87% across categories • 
            ⏱ <strong>Payment Terms:</strong> Blended average ≈ 45 days • 
            🤝 <strong>Supplier Portfolio:</strong> Optimizing consolidation & strategic partnerships • 
            🧠 <strong>AI Insights:</strong> Should-cost, benchmark gaps & scenario simulations powering decisions
        </marquee>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Data initialization and checks
    if "analytics" not in st.session_state or "data_gen" not in st.session_state:
        st.error("❌ Please go to the main dashboard first to initialize data")
        return

    analytics = st.session_state.analytics
    data_gen = st.session_state.data_gen

    # Main Tab Structure (4 tabs standard)
    tab1, tab2, tab3, tab4 = st.tabs(
        [
            "📊 Cost Dashboard",
            "🔍 Spend Analytics",
            "📈 Price Intelligence",
            "🎯 Cost Optimization",
        ]
    )

    with tab1:
        render_cost_dashboard(analytics, data_gen)
    with tab2:
        render_spend_analytics(analytics, data_gen)
    with tab3:
        render_price_intelligence(analytics, data_gen)
    with tab4:
        render_cost_optimization(analytics, data_gen)


# ---------- TAB 1: COST DASHBOARD ----------


def render_cost_dashboard(analytics, data_gen):
    """Tab 1: Comprehensive procurement cost dashboard"""

    # AI Insights expander
    with st.expander("🤖 AI PROCUREMENT COST INSIGHTS", expanded=True):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "💰 Total Procurement Spend",
                "$8.2M",
                "-$450K",
                help="Annual procurement spend with optimization savings",
            )
        with col2:
            st.metric(
                "📉 Cost Avoidance",
                "$1.2M",
                "+$180K",
                help="Cost avoidance through strategic sourcing",
            )
        with col3:
            st.metric(
                "🔄 Savings Rate",
                "14.8%",
                "2.3%",
                help="Overall savings rate across all categories",
            )

        st.info(
            "💡 **AI Recommendation**: Consolidate dairy suppliers and "
            "renegotiate contracts for ~25% savings. Implement strategic sourcing "
            "for packaging materials to achieve ~18% cost reduction with minimal risk."
        )

    # Generate procurement cost data
    cost_data = generate_procurement_cost_data(data_gen)
    _ = generate_procurement_kpi_data(data_gen)  # reserved for future use

    # Top KPIs
    st.subheader("🎯 Procurement Cost Performance Dashboard")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        total_spend = calculate_total_spend(cost_data)
        st.metric("Total Annual Spend", f"${total_spend:,.0f}")

    with col2:
        savings_achieved = calculate_savings_achieved(cost_data)
        st.metric("Savings Achieved", f"${savings_achieved:,.0f}")

    with col3:
        savings_rate = calculate_savings_rate(cost_data)
        st.metric("Savings Rate", f"{savings_rate}%")

    with col4:
        cost_avoidance = calculate_cost_avoidance(cost_data)
        st.metric("Cost Avoidance", f"${cost_avoidance:,.0f}")

    # Spend Distribution Overview
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("📊 Procurement Spend Distribution")
        distribution_fig = create_spend_distribution_chart(cost_data)
        st.plotly_chart(distribution_fig, use_container_width=True)

    with col2:
        st.subheader("🚨 Cost Alerts")
        display_procurement_alerts(cost_data)

    # Category Performance
    st.subheader("📈 Category Performance Analysis")

    col1, col2 = st.columns(2)

    with col1:
        category_savings_fig = create_category_savings_chart(cost_data)
        st.plotly_chart(category_savings_fig, use_container_width=True)

    with col2:
        supplier_performance_fig = create_supplier_cost_analysis(cost_data)
        st.plotly_chart(supplier_performance_fig, use_container_width=True)

    # Real-time Spend Monitoring
    st.subheader("🔍 Real-time Spend Monitoring")
    display_realtime_spend_monitor(cost_data)


# ---------- TAB 2: SPEND ANALYTICS ----------


def render_spend_analytics(analytics, data_gen):
    """Tab 2: Detailed spend analytics and pattern analysis"""

    st.subheader("🔍 Detailed Spend Analytics & Pattern Analysis")

    cost_data = generate_procurement_cost_data(data_gen)
    spend_data = generate_detailed_spend_data(data_gen)

    # Spend Analysis Metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        maverick_spend = calculate_maverick_spend(spend_data)
        st.metric("Maverick Spend", f"${maverick_spend:,.0f}", delta_color="inverse")

    with col2:
        contract_compliance = calculate_contract_compliance(spend_data)
        st.metric("Contract Compliance", f"{contract_compliance} %")

    with col3:
        tail_spend = calculate_tail_spend(spend_data)
        st.metric("Tail Spend", f"${tail_spend:,.0f}", delta_color="inverse")

    with col4:
        payment_terms = calculate_avg_payment_terms(spend_data)
        st.metric("Avg Payment Terms", f"{payment_terms} days")

    # Spend Pattern Analysis
    st.subheader("📊 Spend Pattern Analysis")

    col1, col2 = st.columns(2)

    with col1:
        pattern_fig = create_spend_pattern_analysis(spend_data)
        st.plotly_chart(pattern_fig, use_container_width=True)

    with col2:
        compliance_fig = create_compliance_analysis(spend_data)
        st.plotly_chart(compliance_fig, use_container_width=True)

    # Maverick Spend Analysis
    st.subheader("🎯 Maverick Spend Analysis")

    maverick_analysis = analyze_maverick_spend(spend_data)
    display_maverick_analysis(maverick_analysis)

    # Spend Optimization Opportunities
    st.subheader("💡 Spend Optimization Opportunities")

    optimization_opportunities = identify_spend_optimizations(spend_data)

    for opportunity in optimization_opportunities:
        with st.expander(f"💰 {opportunity['category']} - {opportunity['savings_potential']}"):
            col1, col2 = st.columns(2)

            with col1:
                st.write(f"**Current Spend**: {opportunity['current_spend']}")
                st.write(f"**Target Spend**: {opportunity['target_spend']}")
                st.write(f"**Primary Drivers**: {opportunity['drivers']}")

            with col2:
                st.write(f"**Optimization Strategy**: {opportunity['strategy']}")
                st.write(f"**Implementation Timeline**: {opportunity['timeline']}")
                st.write(f"**Confidence Level**: {opportunity['confidence']}")

            if st.button("Implement Strategy", key=f"impl_{opportunity['category']}"):
                st.success(f"Optimization strategy initiated for {opportunity['category']}")


# ---------- TAB 3: PRICE INTELLIGENCE ----------


def render_price_intelligence(analytics, data_gen):
    """Tab 3: Price intelligence and market analysis"""

    st.subheader("📈 Price Intelligence & Market Analysis")

    price_data = generate_price_intelligence_data(data_gen)
    market_data = generate_market_data(data_gen)

    # Price Intelligence Metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        price_variance = calculate_price_variance(price_data)
        st.metric("Price Variance", f"{price_variance} %", delta_color="inverse")

    with col2:
        market_intelligence = calculate_market_intelligence_score(market_data)
        st.metric("Market Intelligence", f"{market_intelligence}/10")

    with col3:
        should_cost_savings = calculate_should_cost_savings(price_data)
        st.metric("Should-Cost Savings", f"${should_cost_savings:,.0f}")

    with col4:
        benchmark_gap = calculate_benchmark_gap(price_data)
        st.metric("Benchmark Gap", f"{benchmark_gap} %", delta_color="inverse")

    # Price Analysis
    st.subheader("💰 Price Analysis & Benchmarking")

    col1, col2 = st.columns(2)

    with col1:
        price_trend_fig = create_price_trend_analysis(price_data)
        st.plotly_chart(price_trend_fig, use_container_width=True)

    with col2:
        benchmark_fig = create_benchmark_analysis(price_data)
        st.plotly_chart(benchmark_fig, use_container_width=True)

    # Market Intelligence
    st.subheader("🌍 Market Intelligence Dashboard")

    col1, col2 = st.columns(2)

    with col1:
        commodity_fig = create_commodity_analysis(market_data)
        st.plotly_chart(commodity_fig, use_container_width=True)

    with col2:
        supplier_market_fig = create_supplier_market_analysis(market_data)
        st.plotly_chart(supplier_market_fig, use_container_width=True)

    # Should-Cost Analysis
    st.subheader("🛠 Should-Cost Analysis Tool")

    with st.form("should_cost_analysis"):
        col1, col2 = st.columns(2)

        with col1:
            product_category = st.selectbox(
                "Product Category",
                ["Dairy", "Produce", "Meat", "Packaging", "Transportation", "Equipment"],
            )
            volume_tier = st.selectbox(
                "Volume Tier", ["Low (<$50K)", "Medium ($50K-$500K)", "High (>$500K)"]
            )
            complexity_level = st.selectbox(
                "Complexity Level", ["Simple", "Moderate", "Complex"]
            )

        with col2:
            current_price = st.number_input(
                "Current Price ($)", min_value=0.0, value=1000.0
            )
            market_conditions = st.selectbox(
                "Market Conditions",
                ["Stable", "Volatile", "Favorable", "Challenging"],
            )
            negotiation_leverage = st.slider("Negotiation Leverage", 1, 10, 7)

        analyze = st.form_submit_button("🔍 Run Should-Cost Analysis")

        if analyze:
            analysis_results = run_should_cost_analysis(
                product_category,
                volume_tier,
                complexity_level,
                current_price,
                market_conditions,
                negotiation_leverage,
            )
            display_should_cost_results(analysis_results)


# ---------- TAB 4: COST OPTIMIZATION ----------


def render_cost_optimization(analytics, data_gen):
    """Tab 4: Cost optimization strategies and implementation"""

    st.subheader("🎯 Cost Optimization Strategies & Implementation")

    cost_data = generate_procurement_cost_data(data_gen)
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
        implementation_rate = calculate_implementation_rate(optimization_data)
        st.metric("Implementation Rate", f"{implementation_rate} %")

    # Optimization Initiatives
    st.subheader("🚀 Active Optimization Initiatives")

    active_initiatives = get_active_initiatives(optimization_data)

    for initiative in active_initiatives:
        with st.expander(f"📄 {initiative['name']} - {initiative['status']}"):
            col1, col2 = st.columns(2)

            with col1:
                st.write(f"**Objective**: {initiative['objective']}")
                st.write(f"**Scope**: {initiative['scope']}")
                st.write(f"**Target Savings**: {initiative['target_savings']}")
                st.write(f"**Timeline**: {initiative['timeline']}")

            with col2:
                st.write(f"**Budget**: {initiative['budget']}")
                st.write(f"**Resources**: {initiative['resources']}")
                st.write(f"**Risks**: {initiative['risks']}")
                st.write(f"**Progress**: {initiative['progress']}%")

            st.progress(initiative["progress"] / 100)

            # Action buttons
            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button("Update Progress", key=f"progress_{initiative['name']}"):
                    st.success(f"Progress updated for {initiative['name']}")
            with col2:
                if st.button("Request Resources", key=f"resources_{initiative['name']}"):
                    st.info(f"Resource request submitted for {initiative['name']}")
            with col3:
                if st.button("Complete Initiative", key=f"complete_{initiative['name']}"):
                    st.success(f"Initiative {initiative['name']} marked as complete!")

    # Savings Tracking
    st.subheader("📈 Savings Tracking & Validation")

    savings_tracking = generate_savings_tracking(optimization_data)
    display_savings_tracking(savings_tracking)

    # Optimization Simulation
    st.subheader("🛠 Cost Optimization Simulation")

    with st.form("optimization_simulation"):
        col1, col2 = st.columns(2)

        with col1:
            supplier_consolidation = st.slider(
                "Supplier Consolidation (%)", 0, 50, 25
            )
            contract_negotiation = st.slider(
                "Contract Negotiation Savings (%)", 0, 30, 15
            )
            process_efficiency = st.slider("Process Efficiency (%)", 0, 20, 10)

        with col2:
            technology_investment = st.number_input(
                "Technology Investment ($)", 0, 500_000, 150_000
            )
            implementation_period = st.selectbox(
                "Implementation Period", ["6 months", "12 months", "18 months", "24 months"]
            )
            risk_tolerance = st.selectbox(
                "Risk Tolerance", ["Low", "Medium", "High"]
            )

        simulate = st.form_submit_button("🚀 Run Optimization Simulation")

        if simulate:
            simulation_results = run_optimization_simulation(
                cost_data,
                supplier_consolidation,
                contract_negotiation,
                process_efficiency,
                technology_investment,
                implementation_period,
                risk_tolerance,
            )
            display_optimization_simulation_results(simulation_results)


# ---------- DATA GENERATION FUNCTIONS ----------


def generate_procurement_cost_data(data_gen):
    """Generate comprehensive procurement cost data"""

    np.random.seed(42)

    # Annual procurement cost data
    procurement_costs = {
        "dairy_spend": 1_850_000,
        "produce_spend": 1_650_000,
        "meat_spend": 1_200_000,
        "grocery_spend": 950_000,
        "frozen_spend": 850_000,
        "packaging_spend": 650_000,
        "transportation_spend": 450_000,
        "equipment_spend": 350_000,
        "other_spend": 250_000,
    }

    procurement_costs["total_spend"] = sum(procurement_costs.values())
    procurement_costs["savings_achieved"] = 1_210_000
    procurement_costs["savings_rate"] = 14.8
    procurement_costs["cost_avoidance"] = 1_180_000

    return procurement_costs


def generate_procurement_kpi_data(data_gen):
    """Generate procurement KPI data"""

    np.random.seed(42)
    months = 12

    kpi_data = []

    for month in range(months):
        kpi = {
            "month": (
                datetime.now()
                - timedelta(days=30 * (months - month - 1))
            ).strftime("%Y-%m"),
            "total_spend": np.random.uniform(600_000, 800_000),
            "savings_achieved": np.random.uniform(80_000, 120_000),
            "cost_avoidance": np.random.uniform(90_000, 110_000),
            "maverick_spend": np.random.uniform(15_000, 35_000),
        }

        kpi["savings_rate"] = (kpi["savings_achieved"] / kpi["total_spend"]) * 100
        kpi_data.append(kpi)

    return pd.DataFrame(kpi_data)


def generate_detailed_spend_data(data_gen):
    """Generate detailed spend analysis data"""

    np.random.seed(42)
    n_transactions = 1000

    spend_data = []
    categories = [
        "Dairy",
        "Produce",
        "Meat",
        "Grocery",
        "Frozen",
        "Packaging",
        "Transportation",
        "Equipment",
    ]
    suppliers = [
        "Fresh Farms Co.",
        "Dairy Partners Ltd.",
        "Quality Meats Inc.",
        "Global Grocers",
        "Frozen Foods Intl.",
    ]

    for i in range(n_transactions):
        transaction = {
            "transaction_id": f"TRX{10000 + i}",
            "date": datetime.now() - timedelta(days=np.random.randint(1, 365)),
            "category": np.random.choice(categories),
            "supplier": np.random.choice(suppliers),
            "amount": np.random.uniform(100, 50_000),
            "contract_compliant": np.random.choice(
                [True, False], p=[0.85, 0.15]
            ),
            "payment_terms": np.random.choice([30, 45, 60, 90]),
            "maverick_spend": np.random.choice(
                [True, False], p=[0.08, 0.92]
            ),
        }

        spend_data.append(transaction)

    return pd.DataFrame(spend_data)


def generate_price_intelligence_data(data_gen):
    """Generate price intelligence data"""

    np.random.seed(42)
    months = 12

    price_data = []
    categories = ["Dairy", "Produce", "Meat", "Packaging", "Transportation"]

    for category in categories:
        base_price = np.random.uniform(50, 200)
        for month in range(months):
            price_data.append(
                {
                    "category": category,
                    "month": (
                        datetime.now()
                        - timedelta(days=30 * (months - month - 1))
                    ).strftime("%Y-%m"),
                    "your_price": base_price * (1 + np.random.uniform(-0.1, 0.1)),
                    "market_price": base_price * (1 + np.random.uniform(-0.05, 0.15)),
                    "should_cost": base_price * (1 + np.random.uniform(-0.15, 0.05)),
                }
            )

    return pd.DataFrame(price_data)


def generate_market_data(data_gen):
    """Generate market intelligence data"""

    np.random.seed(42)

    return {
        "commodities": ["Dairy", "Produce", "Meat", "Packaging", "Fuel"],
        "price_trends": [2.5, -1.2, 3.8, 1.5, 8.2],  # % change
        "supply_risk": [25, 35, 20, 15, 45],  # risk score
        "market_volatility": [15, 25, 30, 10, 40],  # volatility index
    }


def generate_optimization_data(data_gen):
    """Generate optimization data"""

    np.random.seed(42)

    return {
        "total_savings_potential": 1_450_000,
        "quick_wins": 320_000,
        "strategic_initiatives": 1_130_000,
        "implementation_rate": 68.5,
        "initiatives": [
            {
                "name": "Supplier Consolidation Program",
                "status": "In Progress",
                "objective": "Reduce supplier base by 25% while maintaining quality",
                "scope": "All procurement categories",
                "target_savings": "$450,000",
                "timeline": "12 months",
                "budget": "$75,000",
                "resources": "Sourcing team, category managers",
                "risks": "Supplier relationship impact",
                "progress": 55,
            },
            {
                "name": "Strategic Sourcing Initiative",
                "status": "Planning",
                "objective": "Implement category management for high-spend areas",
                "scope": "Dairy, Produce, Meat categories",
                "target_savings": "$680,000",
                "timeline": "18 months",
                "budget": "$120,000",
                "resources": "Strategic sourcing team, analysts",
                "risks": "Market volatility, implementation complexity",
                "progress": 20,
            },
        ],
    }


# ---------- ANALYTICAL FUNCTIONS ----------


def calculate_total_spend(cost_data):
    """Calculate total procurement spend"""
    return cost_data["total_spend"]


def calculate_savings_achieved(cost_data):
    """Calculate savings achieved"""
    return cost_data["savings_achieved"]


def calculate_savings_rate(cost_data):
    """Calculate savings rate"""
    return cost_data["savings_rate"]


def calculate_cost_avoidance(cost_data):
    """Calculate cost avoidance"""
    return cost_data["cost_avoidance"]


def calculate_maverick_spend(spend_data):
    """Calculate maverick spend"""
    maverick = spend_data[spend_data["maverick_spend"] == True]
    return round(maverick["amount"].sum())


def calculate_contract_compliance(spend_data):
    """Calculate contract compliance rate"""
    compliant = len(spend_data[spend_data["contract_compliant"] == True])
    return round((compliant / len(spend_data)) * 100, 1)


def calculate_tail_spend(spend_data):
    """Calculate tail spend (simplified as 20% of total spend)"""
    return round(spend_data["amount"].sum() * 0.2)


def calculate_avg_payment_terms(spend_data):
    """Calculate average payment terms"""
    return round(spend_data["payment_terms"].mean())


def calculate_price_variance(price_data):
    """Calculate price variance"""
    latest_prices = price_data.groupby("category").last().reset_index()
    variance = (
        (latest_prices["your_price"] - latest_prices["market_price"])
        / latest_prices["market_price"]
        * 100
    )
    return round(variance.mean(), 1)


def calculate_market_intelligence_score(market_data):
    """Calculate market intelligence score (simulated)"""
    return round(np.random.uniform(7.5, 9.5), 1)


def calculate_should_cost_savings(price_data):
    """Calculate should-cost savings potential"""
    latest_prices = price_data.groupby("category").last().reset_index()
    savings_potential = (latest_prices["your_price"] - latest_prices["should_cost"]).sum()
    # scale up for realism
    return round(savings_potential * 1000)


def calculate_benchmark_gap(price_data):
    """Calculate benchmark gap"""
    latest_prices = price_data.groupby("category").last().reset_index()
    gap = (
        (latest_prices["your_price"] - latest_prices["market_price"])
        / latest_prices["market_price"]
        * 100
    )
    return round(gap.mean(), 1)


def calculate_total_savings_potential(optimization_data):
    """Calculate total savings potential"""
    return optimization_data["total_savings_potential"]


def calculate_quick_wins(optimization_data):
    """Calculate quick wins savings"""
    return optimization_data["quick_wins"]


def calculate_strategic_initiatives(optimization_data):
    """Calculate strategic initiatives savings"""
    return optimization_data["strategic_initiatives"]


def calculate_implementation_rate(optimization_data):
    """Calculate implementation rate"""
    return optimization_data["implementation_rate"]


# ---------- VISUALIZATION FUNCTIONS ----------


def create_spend_distribution_chart(cost_data):
    """Create spend distribution chart"""
    categories = [
        "Dairy",
        "Produce",
        "Meat",
        "Grocery",
        "Frozen",
        "Packaging",
        "Transportation",
        "Equipment",
        "Other",
    ]
    values = [
        cost_data["dairy_spend"],
        cost_data["produce_spend"],
        cost_data["meat_spend"],
        cost_data["grocery_spend"],
        cost_data["frozen_spend"],
        cost_data["packaging_spend"],
        cost_data["transportation_spend"],
        cost_data["equipment_spend"],
        cost_data["other_spend"],
    ]

    fig = px.pie(
        values=values,
        names=categories,
        title="Procurement Spend Distribution by Category",
        color_discrete_sequence=px.colors.sequential.RdBu,
    )

    return fig


def create_category_savings_chart(cost_data):
    """Create category savings analysis chart"""
    categories = ["Dairy", "Produce", "Meat", "Grocery", "Frozen"]
    savings_rates = [18.5, 15.2, 12.8, 10.5, 14.3]  # Example savings rates

    fig = px.bar(
        x=categories,
        y=savings_rates,
        title="Savings Rate by Category",
        color=savings_rates,
        color_continuous_scale="RdYlGn",
    )

    return fig


def create_supplier_cost_analysis(cost_data):
    """Create supplier cost analysis chart (simulated)"""
    suppliers = ["Supplier A", "Supplier B", "Supplier C", "Supplier D", "Supplier E"]
    spend = [450_000, 380_000, 320_000, 280_000, 220_000]
    savings = [85_000, 65_000, 48_000, 35_000, 28_000]

    fig = px.scatter(
        x=suppliers,
        y=spend,
        size=savings,
        title="Supplier Spend vs Savings Analysis",
        color=savings,
        color_continuous_scale="RdYlGn",
        labels={"x": "Supplier", "y": "Annual Spend"},
    )

    return fig


def create_spend_pattern_analysis(spend_data):
    """Create spend pattern analysis chart"""
    monthly_spend = (
        spend_data.groupby(spend_data["date"].dt.to_period("M"))["amount"]
        .sum()
        .reset_index()
    )
    monthly_spend["date"] = monthly_spend["date"].astype(str)

    fig = px.line(
        monthly_spend,
        x="date",
        y="amount",
        title="Monthly Spend Pattern Analysis",
        markers=True,
        labels={"date": "Month", "amount": "Spend"},
    )

    return fig


def create_compliance_analysis(spend_data):
    """Create compliance analysis chart"""
    compliance_by_category = (
        spend_data.groupby("category")["contract_compliant"].mean().reset_index()
    )

    fig = px.bar(
        compliance_by_category,
        x="category",
        y="contract_compliant",
        title="Contract Compliance by Category",
        color="contract_compliant",
        color_continuous_scale="RdYlGn",
        labels={"contract_compliant": "Compliance Rate"},
    )

    return fig


def create_price_trend_analysis(price_data):
    """Create price trend analysis chart"""
    fig = px.line(
        price_data,
        x="month",
        y="your_price",
        color="category",
        title="Price Trends by Category",
        markers=True,
        labels={"month": "Month", "your_price": "Your Price"},
    )
    return fig


def create_benchmark_analysis(price_data):
    """Create benchmark analysis chart"""
    latest_prices = price_data.groupby("category").last().reset_index()

    fig = go.Figure()
    fig.add_trace(
        go.Bar(name="Your Price", x=latest_prices["category"], y=latest_prices["your_price"])
    )
    fig.add_trace(
        go.Bar(
            name="Market Price",
            x=latest_prices["category"],
            y=latest_prices["market_price"],
        )
    )
    fig.add_trace(
        go.Bar(
            name="Should Cost", x=latest_prices["category"], y=latest_prices["should_cost"]
        )
    )

    fig.update_layout(
        title="Price Benchmarking Analysis",
        barmode="group",
        xaxis_title="Category",
        yaxis_title="Price",
    )

    return fig


def create_commodity_analysis(market_data):
    """Create commodity analysis chart"""
    fig = px.bar(
        x=market_data["commodities"],
        y=market_data["price_trends"],
        title="Commodity Price Trends",
        color=market_data["price_trends"],
        color_continuous_scale="RdYlGn",
        labels={"x": "Commodity", "y": "Price Trend (%)"},
    )

    return fig


def create_supplier_market_analysis(market_data):
    """Create supplier market analysis chart"""
    fig = px.scatter(
        x=market_data["commodities"],
        y=market_data["supply_risk"],
        size=market_data["market_volatility"],
        title="Supply Risk vs Market Volatility",
        color=market_data["supply_risk"],
        color_continuous_scale="RdYlGn_r",
        labels={"x": "Commodity", "y": "Supply Risk"},
    )

    return fig


# ---------- DISPLAY FUNCTIONS ----------


def display_procurement_alerts(cost_data):
    """Display procurement cost alerts (robust – no missing keys)"""
    high_spend_categories = []

    if cost_data.get("dairy_spend", 0) > 2_000_000:
        high_spend_categories.append("Dairy")
    if cost_data.get("produce_spend", 0) > 1_800_000:
        high_spend_categories.append("Produce")

    if not high_spend_categories:
        st.success("✅ No critical procurement alerts")
        return

    for category in high_spend_categories:
        st.error(f"🚨 {category} Alert")
        st.caption("Above target threshold – review optimization opportunities")


def display_realtime_spend_monitor(cost_data):
    """Display real-time spend monitoring"""
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("MTD Spend", "$685,200")
    with col2:
        st.metric("vs Budget", "-3.2%")
    with col3:
        st.metric("YTD Savings", "$245,800")
    with col4:
        st.metric("Savings Rate", "15.8%")


def display_maverick_analysis(maverick_analysis):
    """Display maverick spend analysis"""
    st.dataframe(maverick_analysis, use_container_width=True)


def display_should_cost_results(analysis_results):
    """Display should-cost analysis results"""
    st.success("🔍 Should-Cost Analysis Complete!")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Target Price", f"${analysis_results['target_price']:,.2f}")
        st.metric(
            "Savings Potential",
            f"{analysis_results['savings_potential']} %",
        )

    with col2:
        st.metric(
            "Negotiation Range",
            f"${analysis_results['negotiation_range_low']:,.2f} - "
            f"${analysis_results['negotiation_range_high']:,.2f}",
        )
        st.metric("Confidence Level", f"{analysis_results['confidence']} %")

    with col3:
        st.metric("Implementation Priority", analysis_results["priority"])
        st.metric("Risk Assessment", analysis_results["risk_level"])


def display_savings_tracking(savings_tracking):
    """Display savings tracking"""
    st.dataframe(savings_tracking, use_container_width=True)


def display_optimization_simulation_results(simulation_results):
    """Display optimization simulation results"""
    st.success("🎯 Optimization Simulation Complete!")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Annual Savings", f"${simulation_results['annual_savings']:,.0f}")
        st.metric("ROI Period", f"{simulation_results['roi_period']} months")

    with col2:
        st.metric("Total Impact", f"{simulation_results['total_impact']} %")
        st.metric("Payback Period", f"{simulation_results['payback_period']} months")

    with col3:
        st.metric("Implementation Risk", simulation_results["implementation_risk"])
        st.metric("Confidence Level", f"{simulation_results['confidence']} %")


# ---------- ANALYSIS / LOGIC HELPERS ----------


def analyze_maverick_spend(spend_data):
    """Analyze maverick spend"""
    maverick_data = spend_data[spend_data["maverick_spend"] == True]
    analysis = (
        maverick_data.groupby("category")
        .agg({"amount": "sum", "transaction_id": "count"})
        .reset_index()
    )

    analysis.rename(
        columns={
            "amount": "total_maverick_spend",
            "transaction_id": "transaction_count",
        },
        inplace=True,
    )
    analysis["total_maverick_spend"] = analysis["total_maverick_spend"].apply(
        lambda x: f"${x:,.0f}"
    )

    return analysis


def identify_spend_optimizations(spend_data):
    """Identify spend optimization opportunities (static examples)"""
    return [
        {
            "category": "Dairy",
            "savings_potential": "$185,000",
            "current_spend": f"${1_850_000:,.0f}",
            "target_spend": f"${1_665_000:,.0f}",
            "drivers": "Supplier consolidation, contract renegotiation",
            "strategy": "Strategic sourcing with 2 primary suppliers",
            "timeline": "6-9 months",
            "confidence": "85%",
        },
        {
            "category": "Packaging",
            "savings_potential": "$85,000",
            "current_spend": f"${650_000:,.0f}",
            "target_spend": f"${565_000:,.0f}",
            "drivers": "Material standardization, volume leverage",
            "strategy": "Consolidate packaging specifications",
            "timeline": "4-6 months",
            "confidence": "90%",
        },
    ]


def run_should_cost_analysis(
    product_category,
    volume_tier,
    complexity_level,
    current_price,
    market_conditions,
    negotiation_leverage,
):
    """Run should-cost analysis (simplified heuristic)"""
    base_reduction = 0.15  # 15% base reduction

    volume_multiplier = {
        "Low (<$50K)": 1.0,
        "Medium ($50K-$500K)": 1.2,
        "High (>$500K)": 1.4,
    }[volume_tier]

    complexity_multiplier = {"Simple": 1.3, "Moderate": 1.0, "Complex": 0.8}[
        complexity_level
    ]

    market_multiplier = {
        "Stable": 1.0,
        "Volatile": 0.9,
        "Favorable": 1.2,
        "Challenging": 0.8,
    }[market_conditions]

    reduction_factor = (
        base_reduction
        * volume_multiplier
        * complexity_multiplier
        * market_multiplier
        * (negotiation_leverage / 10)
    )

    # Cap reduction_factor at 0.6 to avoid negative target price
    reduction_factor = min(reduction_factor, 0.6)

    target_price = current_price * (1 - reduction_factor)

    return {
        "target_price": round(target_price, 2),
        "savings_potential": round((1 - target_price / current_price) * 100, 1)
        if current_price > 0
        else 0.0,
        "negotiation_range_low": round(target_price * 0.95, 2),
        "negotiation_range_high": round(target_price * 1.05, 2),
        "confidence": min(95, 70 + negotiation_leverage * 2),
        "priority": "High" if (1 - target_price / current_price) > 0.1 else "Medium"
        if current_price > 0
        else "Low",
        "risk_level": "Low",
    }


def get_active_initiatives(optimization_data):
    """Get active optimization initiatives"""
    return optimization_data["initiatives"]


def generate_savings_tracking(optimization_data):
    """Generate savings tracking data"""
    tracking_data = [
        {
            "Initiative": "Supplier Consolidation",
            "Planned Savings": "$450,000",
            "Actual Savings": "$245,000",
            "Variance": "-45.6%",
            "Status": "On Track",
        },
        {
            "Initiative": "Contract Renegotiation",
            "Planned Savings": "$280,000",
            "Actual Savings": "$315,000",
            "Variance": "+12.5%",
            "Status": "Ahead",
        },
        {
            "Initiative": "Process Automation",
            "Planned Savings": "$150,000",
            "Actual Savings": "$120,000",
            "Variance": "-20.0%",
            "Status": "Behind",
        },
        {
            "Initiative": "Category Management",
            "Planned Savings": "$680,000",
            "Actual Savings": "$0",
            "Variance": "-100%",
            "Status": "Not Started",
        },
    ]

    return pd.DataFrame(tracking_data)


def run_optimization_simulation(
    cost_data,
    supplier_consolidation,
    contract_negotiation,
    process_efficiency,
    technology_investment,
    implementation_period,
    risk_tolerance,
):
    """Run optimization simulation (simplified heuristic)"""
    annual_savings = (
        cost_data["total_spend"] * (supplier_consolidation / 100) * 0.6
        + cost_data["total_spend"] * (contract_negotiation / 100) * 0.8
        + cost_data["total_spend"] * (process_efficiency / 100) * 0.4
    )

    annual_savings = max(1, annual_savings)  # avoid division by zero

    roi_months = max(6, round(technology_investment / (annual_savings / 12)))
    payback_period = round(technology_investment / (annual_savings / 12))

    total_impact = round((annual_savings / cost_data["total_spend"]) * 100, 1)

    if risk_tolerance == "High":
        implementation_risk = "Low"
        confidence = 65
    elif risk_tolerance == "Medium":
        implementation_risk = "Medium"
        confidence = 75
    else:
        implementation_risk = "High"
        confidence = 85

    return {
        "annual_savings": round(annual_savings),
        "roi_period": roi_months,
        "total_impact": total_impact,
        "payback_period": payback_period,
        "implementation_risk": implementation_risk,
        "confidence": confidence,
    }


if __name__ == "__main__":
    render()
