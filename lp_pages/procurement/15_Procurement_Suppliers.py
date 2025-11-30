# logistics_pro/pages/procurement/15_Procurement_Suppliers.py
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import numpy as np


def render():
    """🏭 SUPPLIER INTELLIGENCE - Strategic Partner Management & Optimization"""

    st.title("🏭 Supplier Intelligence")

    # 🌈 Gradient hero header – aligned with 01_Dashboard style
    st.markdown(
        f"""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                padding: 20px; border-radius: 20px; color: white; margin-bottom: 12px;">
        <h3 style="margin: 0; color: white;">🎯 Strategic Partner Management & Performance Optimization</h3>
        <p style="margin: 5px 0 0 0; opacity: 0.9; color: white;">
            <strong>📍</strong> Procurement Intelligence &gt; Supplier Management |
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
            🤝 <strong>Supplier Excellence:</strong> 87.2% Overall Partner Performance • 
            📦 <strong>On-Time Delivery:</strong> 93.5% OTIF | Service Stability High • 
            💰 <strong>Cost Optimization:</strong> $245K Annual Savings from Strategic Sourcing • 
            ⚠️ <strong>Risk Watch:</strong> 6 High/Critical-Risk Suppliers Under Close Monitoring • 
            🏅 <strong>Strategic Partners:</strong> Tier-1 Suppliers Driving Quality & Innovation • 
            🌱 <strong>Sustainability:</strong> 58% of Spend with Certified / ESG-Aligned Suppliers • 
            📊 <strong>AI Insights:</strong> Consolidation & Rebalancing Opportunities Flagged in Real-Time
        </marquee>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Data initialization and checks (keep consistent with 01_Dashboard guardrails)
    if "analytics" not in st.session_state:
        st.error("❌ Please go to the main dashboard first to initialize data")
        return

    analytics = st.session_state.analytics
    data_gen = st.session_state.get("data_gen", None)

    # Main Tab Structure (4 tabs standard)
    tab1, tab2, tab3, tab4 = st.tabs(
        [
            "📊 Supplier Dashboard",
            "📋 Supplier Scorecards",
            "🔍 Performance Analytics",
            "🤝 Relationship Management",
        ]
    )

    with tab1:
        render_supplier_dashboard(analytics, data_gen)
    with tab2:
        render_supplier_scorecards(analytics, data_gen)
    with tab3:
        render_performance_analytics(analytics, data_gen)
    with tab4:
        render_relationship_management(analytics, data_gen)


def render_supplier_dashboard(analytics, data_gen):
    """Tab 1: Comprehensive supplier performance dashboard"""

    # AI Insights expander
    with st.expander("🧠 AI SUPPLIER OPTIMIZATION INSIGHTS", expanded=True):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "🏅 Supplier Performance",
                "87.2%",
                "3.5%",
                help="Overall supplier performance score across all metrics",
            )
        with col2:
            st.metric(
                "💰 Cost Savings",
                "$245K",
                "+$45K",
                help="Annual savings through supplier optimization",
            )
        with col3:
            st.metric(
                "⚙️ Process Efficiency",
                "78.5%",
                "5.2%",
                help="Supplier process and relationship efficiency",
            )

        st.info(
            "💡 **AI Recommendation**: Consolidate 3 underperforming dairy suppliers into "
            "1 strategic partner. This could improve quality by 15% and reduce costs by "
            "$85,000 annually while maintaining supply chain resilience."
        )

    # Generate supplier data
    supplier_data = generate_supplier_data(data_gen)
    performance_data = generate_performance_data(data_gen)

    # Top KPIs
    st.subheader("🎯 Supplier Performance Dashboard")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        total_suppliers = len(supplier_data)
        st.metric("Total Suppliers", f"{total_suppliers}")

    with col2:
        strategic_suppliers = len(
            supplier_data[supplier_data["performance_tier"] == "Strategic"]
        )
        st.metric("Strategic Partners", f"{strategic_suppliers}")

    with col3:
        avg_performance = calculate_avg_performance(supplier_data)
        st.metric("Avg Performance Score", f"{avg_performance}%")

    with col4:
        risk_exposure = calculate_risk_exposure(supplier_data)
        st.metric("Risk Exposure", f"{risk_exposure}%", delta_color="inverse")

    # Supplier Performance Overview
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("📈 Supplier Performance Trends")
        performance_fig = create_performance_trend_chart(performance_data)
        st.plotly_chart(performance_fig, use_container_width=True)

    with col2:
        st.subheader("🚨 Supplier Alerts")
        display_supplier_alerts(supplier_data)

    # Supplier Category Analysis
    st.subheader("📊 Supplier Performance by Category")

    col1, col2 = st.columns(2)

    with col1:
        category_fig = create_category_performance_chart(supplier_data)
        st.plotly_chart(category_fig, use_container_width=True)

    with col2:
        risk_fig = create_risk_analysis_chart(supplier_data)
        st.plotly_chart(risk_fig, use_container_width=True)

    # Real-time Supplier Status
    st.subheader("🔍 Real-time Supplier Status")
    display_realtime_supplier_status(supplier_data)


def render_supplier_scorecards(analytics, data_gen):
    """Tab 2: Detailed supplier scorecards and evaluation"""

    st.subheader("📋 Comprehensive Supplier Scorecards")

    supplier_data = generate_supplier_data(data_gen)

    # Scorecard Filters
    col1, col2, col3 = st.columns(3)

    with col1:
        performance_filter = st.selectbox(
            "Performance Tier",
            ["All Tiers", "Strategic", "Preferred", "Standard", "Developing"],
        )

    with col2:
        category_filter = st.multiselect(
            "Supplier Categories",
            ["Dairy", "Produce", "Meat", "Grocery", "Frozen", "Bakery", "Beverages"],
            default=["Dairy", "Produce", "Meat"],
        )

    with col3:
        risk_filter = st.selectbox(
            "Risk Level", ["All Levels", "Low", "Medium", "High", "Critical"]
        )

    # Filter suppliers
    filtered_suppliers = filter_suppliers(
        supplier_data, performance_filter, category_filter, risk_filter
    )

    # Display Supplier Scorecards
    for _, supplier in filtered_suppliers.iterrows():
        with st.expander(
            f"🏅 {supplier['supplier_name']} - {supplier['performance_tier']} Partner"
        ):
            # Header with key metrics
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Overall Score", f"{supplier['overall_score']:.1f}/100")

            with col2:
                st.metric("Risk Level", supplier["risk_level"])

            with col3:
                st.metric(
                    "Relationship Score",
                    f"{supplier['relationship_score']:.1f}/10",
                )

            with col4:
                st.metric("Spend (Annual)", f"${supplier['annual_spend']:,.0f}")

            # Detailed Scorecard
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("📊 Performance Metrics")

                # Quality Performance
                st.write(f"**Quality Score**: {supplier['quality_score']:.1f}/10")
                st.progress(supplier["quality_score"] / 10)

                # Delivery Performance
                st.write(
                    f"**On-Time Delivery**: {supplier['on_time_delivery']:.1f}%"
                )
                st.progress(supplier["on_time_delivery"] / 100)

                # Cost Performance
                st.write(f"**Cost Competitiveness**: {supplier['cost_score']:.1f}/10")
                st.progress(supplier["cost_score"] / 10)

                # Innovation Score
                st.write(
                    f"**Innovation Score**: {supplier['innovation_score']:.1f}/10"
                )
                st.progress(supplier["innovation_score"] / 10)

            with col2:
                st.subheader("🔍 Business Details")

                st.write(f"**Primary Categories**: {supplier['primary_categories']}")
                st.write(f"**Contract Status**: {supplier['contract_status']}")
                st.write(
                    f"**Relationship Duration**: {supplier['relationship_years']} years"
                )
                st.write(f"**Payment Terms**: {supplier['payment_terms']} days")
                st.write(
                    f"**Geographic Coverage**: {supplier['geographic_coverage']}"
                )
                st.write(f"**Certifications**: {supplier['certifications']}")

            # Action Buttons
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                if st.button(
                    f"📞 Contact", key=f"contact_{supplier['supplier_id']}"
                ):
                    st.success(f"Contacting {supplier['supplier_name']}")

            with col2:
                if st.button(
                    f"📊 Performance Review", key=f"review_{supplier['supplier_id']}"
                ):
                    st.info(
                        f"Scheduling performance review with {supplier['supplier_name']}"
                    )

            with col3:
                if st.button(
                    f"📝 Update Scorecard", key=f"update_{supplier['supplier_id']}"
                ):
                    st.info(f"Updating scorecard for {supplier['supplier_name']}")

            with col4:
                if st.button(
                    f"🔁 Renew Contract", key=f"renew_{supplier['supplier_id']}"
                ):
                    st.warning(
                        f"Initiating contract renewal for {supplier['supplier_name']}"
                    )


def render_performance_analytics(analytics, data_gen):
    """Tab 3: Advanced supplier performance analytics"""

    st.subheader("🔍 Advanced Supplier Performance Analytics")

    supplier_data = generate_supplier_data(data_gen)
    analytics_data = generate_analytics_data(data_gen)

    # Performance Metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        avg_quality = calculate_avg_quality(supplier_data)
        st.metric("Average Quality Score", f"{avg_quality:.1f}/10")

    with col2:
        avg_delivery = calculate_avg_delivery(supplier_data)
        st.metric("Average On-Time Delivery", f"{avg_delivery:.1f}%")

    with col3:
        cost_savings = calculate_cost_savings(supplier_data)
        st.metric("Annual Cost Savings", f"${cost_savings:,.0f}")

    with col4:
        risk_reduction = calculate_risk_reduction(supplier_data)
        st.metric("Risk Reduction", f"{risk_reduction:.1f}%")

    # Performance Analysis
    st.subheader("📈 Performance Trend Analysis")

    col1, col2 = st.columns(2)

    with col1:
        trend_fig = create_performance_trend_analysis(analytics_data)
        st.plotly_chart(trend_fig, use_container_width=True)

    with col2:
        benchmark_fig = create_benchmark_analysis(supplier_data)
        st.plotly_chart(benchmark_fig, use_container_width=True)

    # Supplier Segmentation
    st.subheader("🎯 Supplier Segmentation Analysis")

    segmentation_fig = create_segmentation_analysis(supplier_data)
    st.plotly_chart(segmentation_fig, use_container_width=True)

    # Performance Improvement Opportunities
    st.subheader("🚀 Performance Improvement Opportunities")

    improvement_opportunities = identify_improvement_opportunities(supplier_data)

    for opportunity in improvement_opportunities:
        with st.expander(
            f"💡 {opportunity['supplier_name']} - {opportunity['improvement_area']}"
        ):
            col1, col2 = st.columns(2)

            with col1:
                st.write(f"**Current Performance**: {opportunity['current_performance']}")
                st.write(f"**Target Performance**: {opportunity['target_performance']}")
                st.write(
                    f"**Improvement Potential**: {opportunity['improvement_potential']}"
                )

            with col2:
                st.write(f"**Key Actions**: {opportunity['key_actions']}")
                st.write(f"**Timeline**: {opportunity['timeline']}")
                st.write(f"**Expected Impact**: {opportunity['expected_impact']}")

            if st.button("Implement Improvements", key=opportunity["supplier_name"]):
                st.success(
                    f"Improvement plan initiated for {opportunity['supplier_name']}"
                )


def render_relationship_management(analytics, data_gen):
    """Tab 4: Supplier relationship management and collaboration"""

    st.subheader("🤝 Supplier Relationship Management")

    supplier_data = generate_supplier_data(data_gen)
    relationship_data = generate_relationship_data(data_gen)

    # Relationship Metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        avg_relationship_score = calculate_avg_relationship_score(supplier_data)
        st.metric("Avg Relationship Score", f"{avg_relationship_score:.1f}/10")

    with col2:
        collaboration_projects = count_collaboration_projects(relationship_data)
        st.metric("Active Collaborations", f"{collaboration_projects}")

    with col3:
        joint_initiatives = count_joint_initiatives(relationship_data)
        st.metric("Joint Initiatives", f"{joint_initiatives}")

    with col4:
        innovation_projects = count_innovation_projects(relationship_data)
        st.metric("Innovation Projects", f"{innovation_projects}")

    # Relationship Health Analysis
    st.subheader("❤️ Relationship Health Analysis")

    col1, col2 = st.columns(2)

    with col1:
        health_fig = create_relationship_health_chart(supplier_data)
        st.plotly_chart(health_fig, use_container_width=True)

    with col2:
        collaboration_fig = create_collaboration_analysis(relationship_data)
        st.plotly_chart(collaboration_fig, use_container_width=True)

    # Strategic Partnership Opportunities
    st.subheader("🌟 Strategic Partnership Opportunities")

    partnership_opportunities = identify_partnership_opportunities(supplier_data)

    for opportunity in partnership_opportunities:
        with st.expander(
            f"🤝 {opportunity['supplier_name']} - {opportunity['partnership_type']}"
        ):
            col1, col2 = st.columns(2)

            with col1:
                st.write(
                    f"**Opportunity Description**: {opportunity['description']}"
                )
                st.write(f"**Strategic Fit**: {opportunity['strategic_fit']}")
                st.write(f"**Expected Benefits**: {opportunity['benefits']}")

            with col2:
                st.write(f"**Investment Required**: {opportunity['investment']}")
                st.write(f"**Implementation Timeline**: {opportunity['timeline']}")
                st.write(f"**Risk Assessment**: {opportunity['risk_level']}")

            # Action buttons
            col1b, col2b, col3b = st.columns(3)
            with col1b:
                if st.button(
                    "Explore Partnership",
                    key=f"explore_{opportunity['supplier_name']}",
                ):
                    st.success(
                        f"Partnership exploration started with {opportunity['supplier_name']}"
                    )
            with col2b:
                if st.button(
                    "Create Business Case",
                    key=f"case_{opportunity['supplier_name']}",
                ):
                    st.info(
                        f"Business case created for {opportunity['supplier_name']}"
                    )
            with col3b:
                if st.button(
                    "Schedule Meeting",
                    key=f"meeting_{opportunity['supplier_name']}",
                ):
                    st.info(
                        f"Meeting scheduled with {opportunity['supplier_name']}"
                    )

    # Supplier Development Programs
    st.subheader("📚 Supplier Development Programs")

    development_programs = get_development_programs(relationship_data)

    for program in development_programs:
        with st.expander(
            f"🎓 {program['program_name']} - {program['status']}"
        ):
            col1, col2 = st.columns(2)

            with col1:
                st.write(f"**Program Objective**: {program['objective']}")
                st.write(f"**Target Suppliers**: {program['target_suppliers']}")
                st.write(f"**Program Duration**: {program['duration']}")

            with col2:
                st.write(f"**Budget**: {program['budget']}")
                st.write(f"**Success Metrics**: {program['success_metrics']}")
                st.write(f"**Progress**: {program['progress']}%")

            st.progress(program["progress"] / 100)

            if st.button("Enroll Suppliers", key=program["program_name"]):
                st.success(
                    f"Supplier enrollment initiated for {program['program_name']}"
                )


# Data Generation Functions
def generate_supplier_data(data_gen):
    """Generate comprehensive supplier data"""

    np.random.seed(42)

    suppliers = [
        "Fresh Farms Co.",
        "Dairy Partners Ltd.",
        "Quality Meats Inc.",
        "Global Grocers",
        "Frozen Foods Intl.",
        "Produce Masters",
        "Bakery Suppliers Co.",
        "Beverage Distributors",
        "Premium Seafood Ltd.",
        "Organic Produce Co.",
        "Spice Traders Inc.",
        "Canned Goods Corp.",
    ]

    supplier_data = []

    for i, supplier in enumerate(suppliers):
        supplier_info = {
            "supplier_id": f"SUP{1000 + i}",
            "supplier_name": supplier,
            "performance_tier": np.random.choice(
                ["Strategic", "Preferred", "Standard", "Developing"],
                p=[0.2, 0.3, 0.4, 0.1],
            ),
            "risk_level": np.random.choice(
                ["Low", "Medium", "High", "Critical"],
                p=[0.6, 0.25, 0.1, 0.05],
            ),
            "overall_score": np.random.uniform(75, 95),
            "quality_score": np.random.uniform(7, 10),
            "on_time_delivery": np.random.uniform(85, 98),
            "cost_score": np.random.uniform(6, 9),
            "innovation_score": np.random.uniform(5, 9),
            "relationship_score": np.random.uniform(7, 10),
            "annual_spend": np.random.uniform(50_000, 500_000),
            "primary_categories": np.random.choice(
                [
                    "Dairy,Produce",
                    "Meat",
                    "Grocery",
                    "Frozen",
                    "Bakery",
                    "Beverages",
                    "Seafood",
                ],
                1,
            )[0],
            "contract_status": np.random.choice(
                ["Active", "Pending Renewal", "Under Review", "New"],
                p=[0.7, 0.15, 0.1, 0.05],
            ),
            "relationship_years": np.random.randint(1, 15),
            "payment_terms": np.random.choice([30, 45, 60, 90]),
            "geographic_coverage": np.random.choice(
                ["Local", "Regional", "National", "International"],
                p=[0.3, 0.4, 0.2, 0.1],
            ),
            "certifications": np.random.choice(
                ["ISO 9001, HACCP", "Organic, ISO 22000", "FSSC 22000", "BRC", "None"],
                p=[0.3, 0.25, 0.2, 0.15, 0.1],
            ),
        }

        supplier_data.append(supplier_info)

    return pd.DataFrame(supplier_data)


def generate_performance_data(data_gen):
    """Generate supplier performance trend data"""

    np.random.seed(42)
    months = 12

    performance_data = []

    for month in range(months):
        month_date = datetime.now() - timedelta(days=30 * (months - month - 1))
        performance_data.append(
            {
                "month": month_date.strftime("%Y-%m"),
                "avg_performance_score": np.random.uniform(80, 90),
                "avg_quality_score": np.random.uniform(8.0, 9.5),
                "avg_delivery_rate": np.random.uniform(88, 96),
                "supplier_count": np.random.randint(8, 12),
            }
        )

    return pd.DataFrame(performance_data)


def generate_analytics_data(data_gen):
    """Generate advanced analytics data"""

    np.random.seed(42)

    analytics_data = {
        "categories": [
            "Dairy",
            "Produce",
            "Meat",
            "Grocery",
            "Frozen",
            "Bakery",
            "Beverages",
        ],
        "performance_scores": [88.5, 85.2, 91.8, 82.3, 89.7, 86.4, 84.1],
        "cost_savings": [45_000, 32_000, 28_000, 15_000, 22_000, 18_000, 12_000],
        "risk_scores": [25, 35, 20, 45, 30, 40, 38],
    }

    return analytics_data


def generate_relationship_data(data_gen):
    """Generate supplier relationship data"""

    np.random.seed(42)

    return {
        "collaboration_projects": 8,
        "joint_initiatives": 12,
        "innovation_projects": 5,
        "development_programs": [
            {
                "program_name": "Quality Excellence Program",
                "status": "Active",
                "objective": "Improve quality standards across supplier base",
                "target_suppliers": "All strategic partners",
                "duration": "12 months",
                "budget": "$75,000",
                "success_metrics": "15% quality improvement",
                "progress": 65,
            },
            {
                "program_name": "Sustainability Initiative",
                "status": "Planning",
                "objective": "Develop sustainable sourcing practices",
                "target_suppliers": "Produce and dairy suppliers",
                "duration": "18 months",
                "budget": "$120,000",
                "success_metrics": "25% carbon footprint reduction",
                "progress": 25,
            },
        ],
    }


# Analytical Functions
def calculate_avg_performance(supplier_data):
    """Calculate average supplier performance"""
    return round(supplier_data["overall_score"].mean(), 1)


def calculate_risk_exposure(supplier_data):
    """Calculate risk exposure percentage"""
    high_risk = len(
        supplier_data[supplier_data["risk_level"].isin(["High", "Critical"])]
    )
    return round((high_risk / len(supplier_data)) * 100, 1)


def calculate_avg_quality(supplier_data):
    """Calculate average quality score"""
    return round(supplier_data["quality_score"].mean(), 1)


def calculate_avg_delivery(supplier_data):
    """Calculate average on-time delivery"""
    return round(supplier_data["on_time_delivery"].mean(), 1)


def calculate_cost_savings(supplier_data):
    """Calculate annual cost savings (simple 8% savings estimate)"""
    return round(supplier_data["annual_spend"].sum() * 0.08)


def calculate_risk_reduction(supplier_data):
    """Calculate risk reduction percentage (simulated)"""
    return round(np.random.uniform(15, 30), 1)


def calculate_avg_relationship_score(supplier_data):
    """Calculate average relationship score"""
    return round(supplier_data["relationship_score"].mean(), 1)


def count_collaboration_projects(relationship_data):
    """Count collaboration projects"""
    return relationship_data["collaboration_projects"]


def count_joint_initiatives(relationship_data):
    """Count joint initiatives"""
    return relationship_data["joint_initiatives"]


def count_innovation_projects(relationship_data):
    """Count innovation projects"""
    return relationship_data["innovation_projects"]


# Visualization Functions
def create_performance_trend_chart(performance_data):
    """Create performance trend chart"""
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=performance_data["month"],
            y=performance_data["avg_performance_score"],
            name="Performance Score",
            line=dict(color="#1f77b4", width=3),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=performance_data["month"],
            y=performance_data["avg_delivery_rate"],
            name="Delivery Rate",
            line=dict(color="#2ca02c", width=2),
        )
    )

    fig.update_layout(
        title="Supplier Performance Trends (12 Months)",
        xaxis_title="Month",
        yaxis_title="Performance (%)",
        hovermode="x unified",
    )

    return fig


def create_category_performance_chart(supplier_data):
    """Create category performance chart"""
    category_performance = []
    categories = ["Dairy", "Produce", "Meat", "Grocery", "Frozen", "Bakery", "Beverages"]

    for category in categories:
        category_suppliers = supplier_data[
            supplier_data["primary_categories"].str.contains(category)
        ]
        if len(category_suppliers) > 0:
            category_performance.append(
                {
                    "category": category,
                    "avg_score": category_suppliers["overall_score"].mean(),
                    "supplier_count": len(category_suppliers),
                }
            )

    perf_df = pd.DataFrame(category_performance)

    fig = px.bar(
        perf_df,
        x="category",
        y="avg_score",
        title="Average Performance by Category",
        color="avg_score",
        color_continuous_scale="RdYlGn",
    )

    return fig


def create_risk_analysis_chart(supplier_data):
    """Create risk analysis chart"""
    risk_counts = supplier_data["risk_level"].value_counts()

    fig = px.pie(
        values=risk_counts.values,
        names=risk_counts.index,
        title="Supplier Risk Distribution",
        color=risk_counts.index,
        color_discrete_map={
            "Low": "#2ca02c",
            "Medium": "#ff7f0e",
            "High": "#ff6b6b",
            "Critical": "#d62728",
        },
    )

    return fig


def create_performance_trend_analysis(analytics_data):
    """Create performance trend analysis chart"""
    fig = px.line(
        x=analytics_data["categories"],
        y=analytics_data["performance_scores"],
        title="Performance Scores by Category",
        markers=True,
    )

    fig.update_layout(xaxis_title="Category", yaxis_title="Performance Score")

    return fig


def create_benchmark_analysis(supplier_data):
    """Create benchmark analysis chart (simulated data)"""
    categories = ["Dairy", "Produce", "Meat", "Grocery", "Frozen", "Bakery", "Beverages"]
    your_scores = [88.5, 85.2, 91.8, 82.3, 89.7, 86.4, 84.1]
    industry_avg = [85.2, 82.8, 88.5, 79.6, 86.3, 83.1, 81.5]

    fig = go.Figure()

    fig.add_trace(go.Bar(name="Your Score", x=categories, y=your_scores))
    fig.add_trace(go.Bar(name="Industry Average", x=categories, y=industry_avg))

    fig.update_layout(
        title="Performance vs Industry Benchmark",
        barmode="group",
        xaxis_title="Category",
        yaxis_title="Score",
    )

    return fig


def create_segmentation_analysis(supplier_data):
    """Create supplier segmentation analysis"""
    fig = px.scatter(
        supplier_data,
        x="annual_spend",
        y="overall_score",
        size="relationship_score",
        color="performance_tier",
        title="Supplier Segmentation Analysis",
        hover_name="supplier_name",
        size_max=20,
    )

    return fig


def create_relationship_health_chart(supplier_data):
    """Create relationship health chart"""
    health_by_tier = (
        supplier_data.groupby("performance_tier")["relationship_score"]
        .mean()
        .reset_index()
    )

    fig = px.bar(
        health_by_tier,
        x="performance_tier",
        y="relationship_score",
        title="Relationship Health by Performance Tier",
        color="relationship_score",
        color_continuous_scale="RdYlGn",
    )

    return fig


def create_collaboration_analysis(relationship_data):
    """Create collaboration analysis chart (simple distribution)"""
    collaboration_types = [
        "Quality Projects",
        "Cost Initiatives",
        "Innovation",
        "Sustainability",
    ]
    project_counts = [5, 3, 2, 2]  # From relationship_data (simplified)

    fig = px.pie(
        values=project_counts,
        names=collaboration_types,
        title="Collaboration Project Distribution",
    )

    return fig


# Display Functions
def display_supplier_alerts(supplier_data):
    """Display supplier performance alerts"""
    critical_suppliers = supplier_data[supplier_data["risk_level"] == "Critical"]
    low_performers = supplier_data[supplier_data["overall_score"] < 75]

    if len(critical_suppliers) == 0 and len(low_performers) == 0:
        st.success("✅ No critical supplier alerts")
        return

    for _, supplier in critical_suppliers.head(3).iterrows():
        st.error(f"🔴 {supplier['supplier_name']} - Critical Risk")
        st.caption(f"Score: {supplier['overall_score']:.1f} | Review required")

    for _, supplier in low_performers.head(3).iterrows():
        st.warning(f"⚠️ {supplier['supplier_name']} - Low Performance")
        st.caption(f"Score: {supplier['overall_score']:.1f} | Improvement needed")


def display_realtime_supplier_status(supplier_data):
    """Display real-time supplier status"""
    status_counts = supplier_data["contract_status"].value_counts()

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        active = status_counts.get("Active", 0)
        st.metric("Active", active)

    with col2:
        renewal = status_counts.get("Pending Renewal", 0)
        st.metric("Pending Renewal", renewal, delta_color="inverse")

    with col3:
        review = status_counts.get("Under Review", 0)
        st.metric("Under Review", review)

    with col4:
        new = status_counts.get("New", 0)
        st.metric("New Suppliers", new)


# Filter and Analysis Functions
def filter_suppliers(supplier_data, performance_filter, category_filter, risk_filter):
    """Filter suppliers based on criteria"""
    filtered = supplier_data.copy()

    if performance_filter != "All Tiers":
        filtered = filtered[filtered["performance_tier"] == performance_filter]

    if category_filter:
        mask = filtered["primary_categories"].apply(
            lambda x: any(cat in x for cat in category_filter)
        )
        filtered = filtered[mask]

    if risk_filter != "All Levels":
        filtered = filtered[filtered["risk_level"] == risk_filter]

    return filtered


def identify_improvement_opportunities(supplier_data):
    """Identify improvement opportunities (sample data)"""
    return [
        {
            "supplier_name": "Dairy Partners Ltd.",
            "improvement_area": "Quality Consistency",
            "current_performance": "8.2/10",
            "target_performance": "9.0/10",
            "improvement_potential": "9.8%",
            "key_actions": "Implement quality monitoring, training programs",
            "timeline": "6 months",
            "expected_impact": "15% reduction in returns",
        },
        {
            "supplier_name": "Global Grocers",
            "improvement_area": "Delivery Performance",
            "current_performance": "86% on-time",
            "target_performance": "94% on-time",
            "improvement_potential": "8.2%",
            "key_actions": "Route optimization, better communication",
            "timeline": "4 months",
            "expected_impact": "12% improvement in service level",
        },
    ]


def identify_partnership_opportunities(supplier_data):
    """Identify partnership opportunities (sample data)"""
    return [
        {
            "supplier_name": "Fresh Farms Co.",
            "partnership_type": "Strategic Alliance",
            "description": "Co-develop sustainable packaging solutions",
            "strategic_fit": "High",
            "benefits": "Cost savings, sustainability leadership",
            "investment": "$50,000",
            "timeline": "12 months",
            "risk_level": "Low",
        },
        {
            "supplier_name": "Premium Seafood Ltd.",
            "partnership_type": "Joint Innovation",
            "description": "Develop value-added seafood products",
            "strategic_fit": "Medium",
            "benefits": "Market differentiation, margin improvement",
            "investment": "$75,000",
            "timeline": "18 months",
            "risk_level": "Medium",
        },
    ]


def get_development_programs(relationship_data):
    """Get supplier development programs"""
    return relationship_data["development_programs"]


if __name__ == "__main__":
    render()
