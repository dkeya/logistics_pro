# logistics_pro/pages/procurement/17_Procurement_Recommendations.py
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import numpy as np


def render():
    """AI PROCUREMENT INTELLIGENCE - Strategic Sourcing & Optimization"""

    st.title("🤖 AI Procurement Intelligence")

    # 🌈 Gradient hero header (aligned with 01_Dashboard.py style)
    st.markdown(
        f"""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                padding: 20px; border-radius: 12px; color: white; margin-bottom: 12px;">
        <h3 style="margin: 0; color: white;">🎯 Strategic Sourcing & AI-Powered Optimization</h3>
        <p style="margin: 5px 0 0 0; opacity: 0.9; color: white;">
        <strong>📍</strong> Procurement Intelligence &gt; AI Recommendations |
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
            🤖 <strong>AI Procurement Intelligence:</strong> Strategic sourcing • 
            💰 <strong>Negotiation Savings:</strong> Multi-year contracts | Portfolio optimization • 
            📦 <strong>Supplier Consolidation:</strong> Reduced fragmentation | Deeper partnerships • 
            ⚖️ <strong>Risk & Compliance:</strong> Supplier diversification | Geo-risk mitigation • 
            ⚙️ <strong>Process Automation:</strong> P2P digitization | Cycle time reduction • 
            📈 <strong>Value Delivery:</strong> Working capital release | Sustainable savings pipeline
        </marquee>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Data initialization and checks
    if "analytics" not in st.session_state or "data_gen" not in st.session_state:
        st.error("❌ Please go to the main dashboard first to initialize procurement data.")
        return

    analytics = st.session_state.analytics
    data_gen = st.session_state.data_gen

    # Main Tab Structure (4 tabs standard)
    tab1, tab2, tab3, tab4 = st.tabs(
        [
            "🎯 AI Recommendations",
            "📊 Strategic Sourcing",
            "🛠️ Process Optimization",
            "🚀 Implementation Hub",
        ]
    )

    with tab1:
        render_ai_recommendations(analytics, data_gen)
    with tab2:
        render_strategic_sourcing(analytics, data_gen)
    with tab3:
        render_process_optimization(analytics, data_gen)
    with tab4:
        render_implementation_hub(analytics, data_gen)


def render_ai_recommendations(analytics, data_gen):
    """Tab 1: AI-powered procurement recommendations"""

    st.subheader("🤖 AI-Powered Procurement Recommendations")

    # AI Recommendation Engine
    with st.form("ai_recommendation_engine"):
        col1, col2 = st.columns(2)

        with col1:
            focus_area = st.selectbox(
                "Focus Area",
                [
                    "All Categories",
                    "Cost Reduction",
                    "Supplier Optimization",
                    "Risk Mitigation",
                    "Process Efficiency",
                ],
            )
            time_horizon = st.selectbox(
                "Time Horizon",
                [
                    "Immediate (0-3 months)",
                    "Short-term (3-6 months)",
                    "Medium-term (6-12 months)",
                    "Long-term (12+ months)",
                ],
            )

        with col2:
            impact_level = st.selectbox(
                "Impact Level",
                ["All Levels", "High Impact", "Medium Impact", "Low Impact"],
            )
            implementation_complexity = st.selectbox(
                "Implementation Complexity",
                ["Any Complexity", "Low", "Medium", "High"],
            )

        generate_recommendations = st.form_submit_button("🚀 Generate AI Recommendations")

    if generate_recommendations:
        with st.spinner(
            "🤖 AI analyzing procurement data and generating recommendations..."
        ):
            recommendations = generate_ai_recommendations(
                focus_area,
                time_horizon,
                impact_level,
                implementation_complexity,
            )
            display_ai_recommendations(recommendations)

    # Quick Action Recommendations
    st.subheader("⚡ Quick Action Recommendations")

    quick_actions = generate_quick_actions()

    for action in quick_actions:
        with st.expander(f"🎯 {action['title']} - {action['impact']}"):
            col1, col2 = st.columns(2)

            with col1:
                st.write(f"**Opportunity**: {action['opportunity']}")
                st.write(f"**Recommended Action**: {action['action']}")
                st.write(f"**Expected Savings**: {action['savings']}")

            with col2:
                st.write(f"**Implementation Time**: {action['implementation_time']}")
                st.write(f"**Confidence Level**: {action['confidence']}")
                st.write(f"**Priority**: {action['priority']}")

            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button("Implement", key=f"imp_{action['id']}"):
                    st.success(f"Action implemented: {action['title']}")
            with col2:
                if st.button("Schedule", key=f"sch_{action['id']}"):
                    st.info(f"Action scheduled: {action['title']}")
            with col3:
                if st.button("More Info", key=f"info_{action['id']}"):
                    st.info(f"Detailed analysis for {action['title']}")

    # AI Insights Dashboard
    st.subheader("📊 AI Insights Dashboard")

    insights_data = generate_insights_data()

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        total_opportunities = len(insights_data["opportunities"])
        st.metric("Total Opportunities", f"{total_opportunities}")

    with col2:
        total_savings = sum(
            [opp["savings_potential"] for opp in insights_data["opportunities"]]
        )
        st.metric("Total Savings Potential", f"${total_savings:,.0f}")

    with col3:
        avg_confidence = np.mean(
            [opp["confidence"] for opp in insights_data["opportunities"]]
        )
        st.metric("Avg Confidence", f"{avg_confidence:.1f}%")

    with col4:
        high_impact = len(
            [opp for opp in insights_data["opportunities"] if opp["impact"] == "High"]
        )
        st.metric("High Impact Opportunities", f"{high_impact}")

    # Opportunity Heatmap
    st.subheader("🌍 Opportunity Heatmap")

    heatmap_fig = create_opportunity_heatmap(insights_data)
    st.plotly_chart(heatmap_fig, use_container_width=True)


def render_strategic_sourcing(analytics, data_gen):
    """Tab 2: Strategic sourcing recommendations"""

    st.subheader("📊 Strategic Sourcing Recommendations")

    sourcing_data = generate_sourcing_data(data_gen)

    # Sourcing Strategy Metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        consolidation_potential = calculate_consolidation_potential(sourcing_data)
        st.metric("Supplier Consolidation Potential", f"{consolidation_potential}%")

    with col2:
        strategic_sourcing_savings = calculate_strategic_sourcing_savings(sourcing_data)
        st.metric(
            "Strategic Sourcing Savings",
            f"${strategic_sourcing_savings:,.0f}",
        )

    with col3:
        category_management_impact = calculate_category_management_impact(sourcing_data)
        st.metric("Category Management Impact", f"{category_management_impact}%")

    with col4:
        risk_mitigation_opportunity = calculate_risk_mitigation_opportunity(
            sourcing_data
        )
        st.metric(
            "Risk Mitigation Opportunity",
            f"{risk_mitigation_opportunity}%",
        )

    # Category Strategy Recommendations
    st.subheader("🎯 Category Strategy Recommendations")

    category_strategies = generate_category_strategies(sourcing_data)

    for strategy in category_strategies:
        with st.expander(f"📦 {strategy['category']} - {strategy['strategy_type']}"):
            col1, col2 = st.columns(2)

            with col1:
                st.write(f"**Current State**: {strategy['current_state']}")
                st.write(
                    f"**Recommended Strategy**: {strategy['recommended_strategy']}"
                )
                st.write(f"**Key Initiatives**: {strategy['key_initiatives']}")

            with col2:
                st.write(f"**Expected Benefits**: {strategy['expected_benefits']}")
                st.write(f"**Implementation Complexity**: {strategy['complexity']}")
                st.write(f"**Timeline**: {strategy['timeline']}")

            # Strategy scoring
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(
                    "Savings Potential",
                    f"${strategy['savings_potential']:,.0f}",
                )
            with col2:
                st.metric("Risk Reduction", f"{strategy['risk_reduction']}%")
            with col3:
                st.metric("Strategic Fit", f"{strategy['strategic_fit']}/10")

            if st.button(
                f"Develop {strategy['category']} Strategy",
                key=strategy["category"],
            ):
                st.success(
                    f"Strategy development initiated for {strategy['category']}"
                )

    # Supplier Portfolio Optimization
    st.subheader("🏗️ Supplier Portfolio Optimization")

    portfolio_recommendations = generate_portfolio_recommendations(sourcing_data)

    for recommendation in portfolio_recommendations:
        with st.expander(
            f"🛠️ {recommendation['action']} - {recommendation['impact']}"
        ):
            st.write(f"**Rationale**: {recommendation['rationale']}")
            st.write(
                f"**Affected Suppliers**: {recommendation['affected_suppliers']}"
            )
            st.write(
                f"**Implementation Steps**: {recommendation['implementation_steps']}"
            )
            st.write(f"**Risk Assessment**: {recommendation['risk_assessment']}")

            col1, col2 = st.columns(2)
            with col1:
                if st.button(
                    f"Execute {recommendation['action']}",
                    key=f"exec_{recommendation['id']}",
                ):
                    st.success(
                        f"Portfolio optimization initiated: {recommendation['action']}"
                    )
            with col2:
                if st.button(
                    "Create Business Case",
                    key=f"case_{recommendation['id']}",
                ):
                    st.info(
                        f"Business case created for {recommendation['action']}"
                    )

    # Sourcing Scenario Analysis
    st.subheader("📈 Sourcing Scenario Analysis")

    with st.form("scenario_analysis"):
        col1, col2 = st.columns(2)

        with col1:
            scenario_type = st.selectbox(
                "Scenario Type",
                [
                    "Supplier Consolidation",
                    "Geographic Diversification",
                    "Cost Optimization",
                    "Risk Mitigation",
                ],
            )
            time_frame = st.slider("Time Frame (months)", 6, 36, 12)
            risk_tolerance = st.select_slider(
                "Risk Tolerance", options=["Low", "Medium", "High"]
            )

        with col2:
            investment_budget = st.number_input(
                "Investment Budget ($)", 0, 1_000_000, 100_000
            )
            implementation_speed = st.select_slider(
                "Implementation Speed",
                options=["Slow", "Moderate", "Fast"],
            )
            focus_categories = st.multiselect(
                "Focus Categories",
                ["Dairy", "Produce", "Meat", "Packaging", "Transportation"],
            )

        analyze_scenario = st.form_submit_button("🔍 Analyze Scenario")

        if analyze_scenario:
            scenario_results = run_sourcing_scenario(
                scenario_type,
                time_frame,
                risk_tolerance,
                investment_budget,
                implementation_speed,
                focus_categories,
            )
            display_scenario_results(scenario_results)


def render_process_optimization(analytics, data_gen):
    """Tab 3: Procurement process optimization"""

    st.subheader("🛠️ Procurement Process Optimization")

    process_data = generate_process_data(data_gen)

    # Process Optimization Metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        process_efficiency = calculate_process_efficiency(process_data)
        st.metric("Process Efficiency", f"{process_efficiency}%")

    with col2:
        automation_potential = calculate_automation_potential(process_data)
        st.metric("Automation Potential", f"{automation_potential}%")

    with col3:
        cycle_time_reduction = calculate_cycle_time_reduction(process_data)
        st.metric("Cycle Time Reduction", f"{cycle_time_reduction}%")

    with col4:
        digital_transformation_impact = calculate_digital_transformation_impact(
            process_data
        )
        st.metric(
            "Digital Transformation Impact",
            f"{digital_transformation_impact}%",
        )

    # Process Improvement Recommendations
    st.subheader("⚙️ Process Improvement Recommendations")

    process_recommendations = generate_process_recommendations(process_data)

    for recommendation in process_recommendations:
        with st.expander(
            f"🛠️ {recommendation['process_area']} - {recommendation['improvement_type']}"
        ):
            col1, col2 = st.columns(2)

            with col1:
                st.write(f"**Current Process**: {recommendation['current_process']}")
                st.write(
                    f"**Recommended Improvement**: {recommendation['recommended_improvement']}"
                )
                st.write(f"**Key Benefits**: {recommendation['key_benefits']}")

            with col2:
                st.write(
                    f"**Implementation Steps**: {recommendation['implementation_steps']}"
                )
                st.write(
                    f"**Technology Requirements**: {recommendation['technology_requirements']}"
                )
                st.write(
                    f"**Change Management**: {recommendation['change_management']}"
                )

            # Impact metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Efficiency Gain", f"{recommendation['efficiency_gain']}%")
            with col2:
                st.metric(
                    "Cost Reduction",
                    f"${recommendation['cost_reduction']:,.0f}",
                )
            with col3:
                st.metric(
                    "Cycle Time Improvement",
                    f"{recommendation['cycle_time_improvement']}%",
                )
            with col4:
                st.metric("ROI Period", f"{recommendation['roi_period']} months")

            if st.button(
                f"Implement {recommendation['process_area']} Improvement",
                key=recommendation["process_area"],
            ):
                st.success(
                    f"Process improvement initiated for {recommendation['process_area']}"
                )

    # Digital Transformation Roadmap
    st.subheader("🚀 Digital Transformation Roadmap")

    digital_roadmap = generate_digital_roadmap(process_data)

    for phase in digital_roadmap:
        with st.expander(f"📅 {phase['phase_name']} - {phase['timeline']}"):
            col1, col2 = st.columns(2)

            with col1:
                st.write(f"**Objectives**: {phase['objectives']}")
                st.write(f"**Key Initiatives**: {phase['key_initiatives']}")
                st.write(f"**Success Metrics**: {phase['success_metrics']}")

            with col2:
                st.write(f"**Budget**: {phase['budget']}")
                st.write(f"**Resources**: {phase['resources']}")
                st.write(f"**Dependencies**: {phase['dependencies']}")

            st.write(f"**Progress**: {phase['progress']}%")
            st.progress(phase["progress"] / 100)

    # Process Benchmarking
    st.subheader("🏆 Process Benchmarking")

    benchmarking_data = generate_benchmarking_data()

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Your Efficiency", f"{process_efficiency}%")
    with col2:
        st.metric("Industry Average", f"{benchmarking_data['industry_avg']}%")
    with col3:
        st.metric("Best in Class", f"{benchmarking_data['best_in_class']}%")

    st.info(
        f"📊 **Performance Gap**: You are "
        f"{process_efficiency - benchmarking_data['industry_avg']:.1f}% above industry average "
        f"and {benchmarking_data['best_in_class'] - process_efficiency:.1f}% below best in class."
    )


def render_implementation_hub(analytics, data_gen):
    """Tab 4: Recommendation implementation and tracking"""

    st.subheader("🚀 Recommendation Implementation Hub")

    implementation_data = generate_implementation_data(data_gen)

    # Implementation Metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        active_implementations = len(
            [imp for imp in implementation_data if imp["status"] == "In Progress"]
        )
        st.metric("Active Implementations", f"{active_implementations}")

    with col2:
        completed_implementations = len(
            [imp for imp in implementation_data if imp["status"] == "Completed"]
        )
        st.metric("Completed", f"{completed_implementations}")

    with col3:
        total_savings_realized = sum(
            [
                imp["savings_realized"]
                for imp in implementation_data
                if imp["status"] == "Completed"
            ]
        )
        st.metric("Savings Realized", f"${total_savings_realized:,.0f}")

    with col4:
        overall_success_rate = calculate_success_rate(implementation_data)
        st.metric("Success Rate", f"{overall_success_rate}%")

    # Implementation Portfolio
    st.subheader("📋 Implementation Portfolio")

    for implementation in implementation_data:
        status_color = {
            "Not Started": "gray",
            "Planning": "blue",
            "In Progress": "orange",
            "Completed": "green",
            "On Hold": "red",
        }.get(implementation["status"], "gray")

        with st.expander(f"{implementation['title']} - {implementation['status']}"):
            col1, col2 = st.columns(2)

            with col1:
                st.write(f"**Category**: {implementation['category']}")
                st.write(f"**Priority**: {implementation['priority']}")
                st.write(f"**Start Date**: {implementation['start_date']}")
                st.write(f"**Target Completion**: {implementation['target_completion']}")
                st.write(f"**Project Lead**: {implementation['project_lead']}")

            with col2:
                st.write(f"**Budget**: {implementation['budget']}")
                st.write(f"**Resources**: {implementation['resources']}")
                st.write(f"**Risks**: {implementation['risks']}")
                st.write(f"**Dependencies**: {implementation['dependencies']}")

            # Progress tracking
            st.write(f"**Progress**: {implementation['progress']}%")
            st.progress(implementation["progress"] / 100)

            # Savings tracking
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Target Savings", implementation["target_savings"])
            with col2:
                st.metric(
                    "Savings Realized",
                    f"${implementation['savings_realized']:,.0f}",
                )
            with col3:
                st.metric("Variance", f"{implementation['variance']}%")

            # Action buttons
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                if st.button(
                    "Update Progress",
                    key=f"progress_{implementation['id']}",
                ):
                    st.success(
                        f"Progress updated for {implementation['title']}"
                    )
            with col2:
                if st.button(
                    "Request Resources",
                    key=f"resources_{implementation['id']}",
                ):
                    st.info(
                        f"Resource request submitted for {implementation['title']}"
                    )
            with col3:
                if st.button(
                    "Report Issue",
                    key=f"issue_{implementation['id']}",
                ):
                    st.warning(
                        f"Issue reported for {implementation['title']}"
                    )
            with col4:
                if st.button(
                    "Complete",
                    key=f"complete_{implementation['id']}",
                ):
                    st.success(
                        f"Implementation completed for {implementation['title']}"
                    )

    # Implementation Dashboard
    st.subheader("📊 Implementation Dashboard")

    col1, col2 = st.columns(2)

    with col1:
        status_fig = create_implementation_status_chart(implementation_data)
        st.plotly_chart(status_fig, use_container_width=True)

    with col2:
        savings_fig = create_savings_tracking_chart(implementation_data)
        st.plotly_chart(savings_fig, use_container_width=True)

    # Implementation Best Practices
    st.subheader("💡 Implementation Best Practices")

    best_practices = generate_best_practices()

    for practice in best_practices:
        with st.expander(f"⭐ {practice['area']}"):
            st.write(f"**Practice**: {practice['practice']}")
            st.write(f"**Benefits**: {practice['benefits']}")
            st.write(
                f"**Implementation Tips**: {practice['implementation_tips']}"
            )
            st.write(
                f"**Success Stories**: {practice['success_stories']}"
            )


# Data Generation Functions
def generate_ai_recommendations(
    focus_area, time_horizon, impact_level, implementation_complexity
):
    """Generate AI-powered procurement recommendations"""

    recommendations = [
        {
            "id": 1,
            "title": "Supplier Consolidation - Dairy Category",
            "description": "Consolidate 5 dairy suppliers into 2 strategic partners to achieve economies of scale",
            "category": "Supplier Optimization",
            "impact": "High",
            "savings_potential": 185000,
            "implementation_time": "3-6 months",
            "confidence": 92,
            "complexity": "Medium",
            "key_metrics": [
                "25% supplier reduction",
                "15% cost savings",
                "Improved quality control",
            ],
            "risks": [
                "Supplier relationship management",
                "Transition period disruption",
            ],
            "next_steps": [
                "Supplier performance analysis",
                "Contract review",
                "Stakeholder alignment",
            ],
        },
        {
            "id": 2,
            "title": "Digital Procurement Platform Implementation",
            "description": "Implement AI-powered procurement platform to automate processes and improve decision-making",
            "category": "Process Efficiency",
            "impact": "High",
            "savings_potential": 320000,
            "implementation_time": "6-12 months",
            "confidence": 88,
            "complexity": "High",
            "key_metrics": [
                "40% process automation",
                "30% cycle time reduction",
                "Improved compliance",
            ],
            "risks": [
                "Implementation complexity",
                "User adoption",
                "Integration challenges",
            ],
            "next_steps": [
                "Vendor evaluation",
                "Business case development",
                "Change management planning",
            ],
        },
    ]

    # Filter recommendations based on user input
    filtered_recommendations = []
    for rec in recommendations:
        if focus_area != "All Categories" and rec["category"] != focus_area:
            continue
        if impact_level != "All Levels" and rec["impact"] != impact_level:
            continue
        if (
            implementation_complexity != "Any Complexity"
            and rec["complexity"] != implementation_complexity
        ):
            continue
        filtered_recommendations.append(rec)

    return filtered_recommendations


def generate_quick_actions():
    """Generate quick action recommendations"""
    return [
        {
            "id": 1,
            "title": "Renegotiate Packaging Contracts",
            "opportunity": "Current contracts expire in 60 days",
            "action": "Initiate competitive bidding process",
            "savings": "$85,000",
            "implementation_time": "4-6 weeks",
            "confidence": "95%",
            "priority": "High",
            "impact": "Medium",
        },
        {
            "id": 2,
            "title": "Implement E-Procurement for Tail Spend",
            "opportunity": "45% of transactions are manual",
            "action": "Deploy automated procurement system",
            "savings": "$45,000",
            "implementation_time": "8-12 weeks",
            "confidence": "88%",
            "priority": "Medium",
            "impact": "High",
        },
    ]


def generate_insights_data():
    """Generate insights data for dashboard"""
    return {
        "opportunities": [
            {
                "category": "Supplier Consolidation",
                "savings_potential": 185000,
                "confidence": 92,
                "impact": "High",
            },
            {
                "category": "Process Automation",
                "savings_potential": 320000,
                "confidence": 88,
                "impact": "High",
            },
            {
                "category": "Contract Optimization",
                "savings_potential": 125000,
                "confidence": 85,
                "impact": "Medium",
            },
            {
                "category": "Risk Mitigation",
                "savings_potential": 75000,
                "confidence": 78,
                "impact": "Medium",
            },
        ]
    }


def generate_sourcing_data(data_gen):
    """Generate strategic sourcing data"""
    return {
        "consolidation_potential": 35,
        "strategic_sourcing_savings": 850000,
        "category_management_impact": 28,
        "risk_mitigation_opportunity": 42,
    }


def generate_process_data(data_gen):
    """Generate process optimization data"""
    return {
        "process_efficiency": 68,
        "automation_potential": 45,
        "cycle_time_reduction": 32,
        "digital_transformation_impact": 55,
    }


def generate_implementation_data(data_gen):
    """Generate implementation tracking data"""
    return [
        {
            "id": 1,
            "title": "Supplier Consolidation Program",
            "category": "Supplier Optimization",
            "status": "In Progress",
            "priority": "High",
            "start_date": "2024-01-15",
            "target_completion": "2024-06-30",
            "project_lead": "John Smith",
            "budget": "$75,000",
            "resources": "Sourcing Team, Category Managers",
            "risks": "Supplier transition, Quality consistency",
            "dependencies": "Contract negotiations, Stakeholder approval",
            "progress": 65,
            "target_savings": "$450,000",
            "savings_realized": 245000,
            "variance": "-45.6%",
        },
        {
            "id": 2,
            "title": "Digital Procurement Implementation",
            "category": "Process Efficiency",
            "status": "Planning",
            "priority": "High",
            "start_date": "2024-03-01",
            "target_completion": "2024-12-31",
            "project_lead": "Sarah Johnson",
            "budget": "$250,000",
            "resources": "IT Team, Procurement, Change Management",
            "risks": "User adoption, Integration complexity",
            "dependencies": "Vendor selection, Budget approval",
            "progress": 20,
            "target_savings": "$680,000",
            "savings_realized": 0,
            "variance": "-100%",
        },
    ]


# Analytical Functions
def calculate_consolidation_potential(sourcing_data):
    """Calculate supplier consolidation potential"""
    return sourcing_data["consolidation_potential"]


def calculate_strategic_sourcing_savings(sourcing_data):
    """Calculate strategic sourcing savings"""
    return sourcing_data["strategic_sourcing_savings"]


def calculate_category_management_impact(sourcing_data):
    """Calculate category management impact"""
    return sourcing_data["category_management_impact"]


def calculate_risk_mitigation_opportunity(sourcing_data):
    """Calculate risk mitigation opportunity"""
    return sourcing_data["risk_mitigation_opportunity"]


def calculate_process_efficiency(process_data):
    """Calculate process efficiency"""
    return process_data["process_efficiency"]


def calculate_automation_potential(process_data):
    """Calculate automation potential"""
    return process_data["automation_potential"]


def calculate_cycle_time_reduction(process_data):
    """Calculate cycle time reduction potential"""
    return process_data["cycle_time_reduction"]


def calculate_digital_transformation_impact(process_data):
    """Calculate digital transformation impact"""
    return process_data["digital_transformation_impact"]


def calculate_success_rate(implementation_data):
    """Calculate implementation success rate"""
    completed = len([imp for imp in implementation_data if imp["status"] == "Completed"])
    total = len(implementation_data)
    return round((completed / total) * 100, 1) if total > 0 else 0


# Visualization Functions
def create_opportunity_heatmap(insights_data):
    """Create opportunity heatmap"""
    categories = [opp["category"] for opp in insights_data["opportunities"]]
    savings = [opp["savings_potential"] for opp in insights_data["opportunities"]]
    confidence = [opp["confidence"] for opp in insights_data["opportunities"]]

    fig = px.scatter(
        x=categories,
        y=savings,
        size=confidence,
        title="Opportunity Heatmap - Savings vs Confidence",
        color=confidence,
        color_continuous_scale="RdYlGn",
    )

    return fig


def create_implementation_status_chart(implementation_data):
    """Create implementation status chart"""
    status_counts = {}
    for imp in implementation_data:
        status = imp["status"]
        status_counts[status] = status_counts.get(status, 0) + 1

    fig = px.pie(
        values=list(status_counts.values()),
        names=list(status_counts.keys()),
        title="Implementation Status Distribution",
    )

    return fig


def create_savings_tracking_chart(implementation_data):
    """Create savings tracking chart"""
    projects = [imp["title"] for imp in implementation_data]
    target_savings = [
        int(imp["target_savings"].replace("$", "").replace(",", ""))
        for imp in implementation_data
    ]
    realized_savings = [imp["savings_realized"] for imp in implementation_data]

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            name="Target Savings",
            x=projects,
            y=target_savings,
        )
    )
    fig.add_trace(
        go.Bar(
            name="Realized Savings",
            x=projects,
            y=realized_savings,
        )
    )

    fig.update_layout(
        title="Savings Tracking - Target vs Realized",
        barmode="group",
    )

    return fig


# Display Functions
def display_ai_recommendations(recommendations):
    """Display AI recommendations"""
    if not recommendations:
        st.warning(
            "No recommendations match your criteria. Try broadening your search."
        )
        return

    for rec in recommendations:
        with st.expander(f"🎯 {rec['title']} - {rec['impact']} Impact"):
            col1, col2 = st.columns(2)

            with col1:
                st.write(f"**Description**: {rec['description']}")
                st.write(f"**Category**: {rec['category']}")
                st.write(
                    f"**Savings Potential**: ${rec['savings_potential']:,.0f}"
                )
                st.write(f"**Implementation Time**: {rec['implementation_time']}")

            with col2:
                st.write(f"**Confidence**: {rec['confidence']}%")
                st.write(f"**Complexity**: {rec['complexity']}")
                st.write("**Key Metrics:**")
                for metric in rec["key_metrics"]:
                    st.write(f"• {metric}")

            # Risks and next steps
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Risks:**")
                for risk in rec["risks"]:
                    st.write(f"• {risk}")

            with col2:
                st.write("**Next Steps:**")
                for step in rec["next_steps"]:
                    st.write(f"• {step}")

            # Action buttons
            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button(
                    "Implement Recommendation",
                    key=f"imp_{rec['id']}",
                ):
                    st.success(f"Implementation started: {rec['title']}")
            with col2:
                if st.button(
                    "Create Business Case",
                    key=f"case_{rec['id']}",
                ):
                    st.info(f"Business case created: {rec['title']}")
            with col3:
                if st.button(
                    "Schedule Review",
                    key=f"review_{rec['id']}",
                ):
                    st.info(f"Review scheduled: {rec['title']}")


def generate_category_strategies(sourcing_data):
    """Generate category strategy recommendations"""
    return [
        {
            "category": "Dairy",
            "strategy_type": "Strategic Sourcing",
            "current_state": "Fragmented supplier base with 8 active suppliers",
            "recommended_strategy": "Consolidate to 2-3 strategic partners with long-term contracts",
            "key_initiatives": [
                "Supplier performance evaluation",
                "Contract renegotiation",
                "Quality standardization",
            ],
            "expected_benefits": "25% cost reduction, improved quality consistency",
            "complexity": "Medium",
            "timeline": "6-9 months",
            "savings_potential": 185000,
            "risk_reduction": 35,
            "strategic_fit": 9,
        }
    ]


def generate_portfolio_recommendations(sourcing_data):
    """Generate supplier portfolio recommendations"""
    return [
        {
            "id": 1,
            "action": "Consolidate Dairy Suppliers",
            "impact": "High",
            "rationale": "Current supplier base is fragmented, leading to inefficiencies and higher costs",
            "affected_suppliers": "Reduce from 8 to 3 strategic partners",
            "implementation_steps": [
                "Performance evaluation",
                "Contract negotiation",
                "Transition planning",
            ],
            "risk_assessment": "Medium - managed through phased transition",
        }
    ]


def run_sourcing_scenario(
    scenario_type,
    time_frame,
    risk_tolerance,
    investment_budget,
    implementation_speed,
    focus_categories,
):
    """Run sourcing scenario analysis"""
    return {
        "scenario_name": f"{scenario_type} Scenario",
        "estimated_savings": investment_budget * 3.5,
        "implementation_timeline": f"{time_frame} months",
        "risk_level": risk_tolerance,
        "key_benefits": [
            "Cost reduction",
            "Process efficiency",
            "Risk mitigation",
        ],
        "success_probability": 85,
        "recommendation": "Proceed with scenario implementation",
    }


def display_scenario_results(scenario_results):
    """Display scenario analysis results"""
    st.success("✅ Scenario Analysis Complete!")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Estimated Savings",
            f"${scenario_results['estimated_savings']:,.0f}",
        )
        st.metric(
            "Success Probability",
            f"{scenario_results['success_probability']}%",
        )

    with col2:
        st.metric(
            "Implementation Timeline",
            scenario_results["implementation_timeline"],
        )
        st.metric("Risk Level", scenario_results["risk_level"])

    with col3:
        st.metric("Recommendation", scenario_results["recommendation"])

    st.write("**Key Benefits:**")
    for benefit in scenario_results["key_benefits"]:
        st.write(f"• {benefit}")


def generate_process_recommendations(process_data):
    """Generate process improvement recommendations"""
    return [
        {
            "process_area": "Purchase-to-Pay",
            "improvement_type": "Automation",
            "current_process": "Manual PO creation and approval workflows",
            "recommended_improvement": "Implement AI-powered automated procurement system",
            "key_benefits": "Faster processing, reduced errors, better compliance",
            "implementation_steps": [
                "System selection",
                "Configuration",
                "Training",
                "Go-live",
            ],
            "technology_requirements": "Procurement software, integration tools",
            "change_management": "User training, process documentation, support structure",
            "efficiency_gain": 45,
            "cost_reduction": 125000,
            "cycle_time_improvement": 60,
            "roi_period": 18,
        }
    ]


def generate_digital_roadmap(process_data):
    """Generate digital transformation roadmap"""
    return [
        {
            "phase_name": "Foundation Phase",
            "timeline": "Months 1-6",
            "objectives": "Establish digital infrastructure and basic automation",
            "key_initiatives": [
                "System selection",
                "Data migration",
                "Basic automation",
            ],
            "success_metrics": "30% process automation, 25% cycle time reduction",
            "budget": "$150,000",
            "resources": "Core team, IT support",
            "dependencies": "Budget approval, vendor selection",
            "progress": 75,
        }
    ]


def generate_benchmarking_data():
    """Generate benchmarking data"""
    return {
        "industry_avg": 58,
        "best_in_class": 82,
        "top_quartile": 71,
        "bottom_quartile": 45,
    }


def generate_best_practices():
    """Generate implementation best practices"""
    return [
        {
            "area": "Change Management",
            "practice": "Structured change management program",
            "benefits": "Higher user adoption, smoother transitions",
            "implementation_tips": "Early stakeholder engagement, comprehensive training",
            "success_stories": "45% faster implementation at Company XYZ",
        }
    ]


if __name__ == "__main__":
    render()
