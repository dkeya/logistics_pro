# logistics_pro/pages/18_Admin_Tenants.py
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import numpy as np


def render():
    """Admin Tenant Management - ENTERPRISE VERSION"""

    st.title("🏢 Admin Tenant Management")
    st.markdown(
        f"""
    <div style="background: #e0f2fe; padding: 15px; border-radius: 8px; border-left: 4px solid #1f77b4; margin-bottom: 20px;">
        <strong>📍 Location:</strong> System Administration &gt; Tenant Management | 
        <strong>Tenant:</strong> {st.session_state.get('current_tenant', 'ELORA Holding')}
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Data initialization and checks
    if "analytics" not in st.session_state:
        st.error("❌ Please go to the main dashboard first to initialize data")
        return

    analytics = st.session_state.analytics
    data_gen = st.session_state.data_gen

    # Main Tab Structure (4 tabs standard)
    tab1, tab2, tab3, tab4 = st.tabs(
        [
            "📊 Tenant Dashboard",
            "👥 Tenant Management",
            "⚙️ Configuration",
            "📈 Performance Analytics",
        ]
    )

    with tab1:
        render_tenant_dashboard(analytics, data_gen)
    with tab2:
        render_tenant_management(analytics, data_gen)
    with tab3:
        render_tenant_configuration(analytics, data_gen)
    with tab4:
        render_performance_analytics(analytics, data_gen)


def render_tenant_dashboard(analytics, data_gen):
    """Tab 1: Comprehensive tenant overview dashboard"""

    # System Insights expander
    with st.expander("🧠 SYSTEM HEALTH & PERFORMANCE INSIGHTS", expanded=True):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "🏢 Active Tenants",
                "8",
                "+2",
                help="Number of active tenants in the system",
            )
        with col2:
            st.metric(
                "📊 System Utilization",
                "78.5%",
                "5.2%",
                help="Overall system utilization across all tenants",
            )
        with col3:
            st.metric(
                "💰 Monthly Revenue",
                "$45.8K",
                "+$8.2K",
                help="Monthly recurring revenue from all tenants",
            )

        st.info(
            "💡 **System Recommendation**: Tenant **QuickMart Kenya** shows 95% utilization "
            "- consider upgrading their plan to Enterprise tier for better performance and "
            "additional **$2,500** monthly revenue."
        )

    # Generate tenant data
    tenant_data = generate_tenant_data(data_gen)
    system_data = generate_system_data(data_gen)

    # Top KPIs
    st.subheader("🎯 Multi-Tenant System Dashboard")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        total_tenants = len(tenant_data)
        st.metric("Total Tenants", f"{total_tenants}")

    with col2:
        active_tenants = len(tenant_data[tenant_data["status"] == "Active"])
        st.metric("Active Tenants", f"{active_tenants}")

    with col3:
        avg_utilization = calculate_avg_utilization(tenant_data)
        st.metric("Avg Utilization", f"{avg_utilization}%")

    with col4:
        system_health = calculate_system_health(system_data)
        st.metric("System Health", f"{system_health}%")

    # Tenant Distribution Overview
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("📊 Tenant Distribution & Performance")
        distribution_fig = create_tenant_distribution_chart(tenant_data)
        st.plotly_chart(distribution_fig, use_container_width=True)

    with col2:
        st.subheader("🚨 System Alerts")
        display_system_alerts(tenant_data)

    # Tenant Performance Analysis
    st.subheader("📈 Tenant Performance Analysis")

    col1, col2 = st.columns(2)

    with col1:
        performance_fig = create_tenant_performance_chart(tenant_data)
        st.plotly_chart(performance_fig, use_container_width=True)

    with col2:
        revenue_fig = create_revenue_analysis_chart(tenant_data)
        st.plotly_chart(revenue_fig, use_container_width=True)

    # Real-time System Monitoring
    st.subheader("🔍 Real-time System Monitoring")
    display_realtime_system_monitor(system_data)


def render_tenant_management(analytics, data_gen):
    """Tab 2: Tenant management and operations"""

    st.subheader("👥 Tenant Management & Operations")

    tenant_data = generate_tenant_data(data_gen)

    # Tenant Management Actions
    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("➕ Create New Tenant", use_container_width=True):
            st.session_state.show_tenant_creation = True

    with col2:
        if st.button("🔄 Refresh Tenant Data", use_container_width=True):
            st.success("Tenant data refreshed successfully!")

    with col3:
        if st.button("📊 Generate Tenant Report", use_container_width=True):
            st.info("Tenant report generation started...")

    # Tenant Creation Form
    if st.session_state.get("show_tenant_creation", False):
        with st.form("create_tenant"):
            st.subheader("🏢 Create New Tenant")

            col1, col2 = st.columns(2)

            with col1:
                tenant_name = st.text_input("Tenant Name*")
                tenant_domain = st.text_input("Tenant Domain*")
                plan_tier = st.selectbox(
                    "Plan Tier*",
                    ["Starter", "Professional", "Enterprise", "Custom"],
                )
                industry = st.selectbox(
                    "Industry*",
                    ["Retail", "Manufacturing", "Logistics", "Healthcare", "Other"],
                )

            with col2:
                admin_email = st.text_input("Admin Email*")
                max_users = st.number_input(
                    "Maximum Users*", min_value=1, max_value=1000, value=10
                )
                contract_value = st.number_input(
                    "Contract Value ($)*", min_value=0, value=5000
                )
                contract_duration = st.selectbox(
                    "Contract Duration*",
                    ["Monthly", "Quarterly", "Annual", "Multi-year"],
                )

            # Additional configuration
            st.subheader("⚙️ Tenant Configuration")

            col1, col2 = st.columns(2)

            with col1:
                modules_enabled = st.multiselect(
                    "Enabled Modules",
                    [
                        "Sales Intelligence",
                        "Inventory Management",
                        "Logistics",
                        "Procurement",
                        "Analytics",
                    ],
                    default=["Sales Intelligence", "Inventory Management"],
                )
                data_retention = st.slider("Data Retention (months)", 1, 36, 12)
                api_access = st.checkbox("Enable API Access", value=True)

            with col2:
                custom_branding = st.checkbox("Enable Custom Branding")
                sso_enabled = st.checkbox("Enable Single Sign-On")
                backup_frequency = st.selectbox(
                    "Backup Frequency", ["Daily", "Weekly", "Monthly"]
                )

            col1, col2 = st.columns(2)
            with col1:
                create_tenant = st.form_submit_button(
                    "🚀 Create Tenant", use_container_width=True
                )
            with col2:
                cancel_creation = st.form_submit_button(
                    "❌ Cancel", use_container_width=True
                )

            if create_tenant:
                if tenant_name and tenant_domain and admin_email:
                    _ = create_new_tenant(
                        tenant_name,
                        tenant_domain,
                        plan_tier,
                        industry,
                        admin_email,
                        max_users,
                        contract_value,
                        contract_duration,
                        modules_enabled,
                        data_retention,
                        api_access,
                        custom_branding,
                        sso_enabled,
                        backup_frequency,
                    )
                    st.success(f"✅ Tenant '{tenant_name}' created successfully!")
                    st.session_state.show_tenant_creation = False
                else:
                    st.error("❌ Please fill in all required fields (marked with *)")

            if cancel_creation:
                st.session_state.show_tenant_creation = False
                st.info("Tenant creation cancelled")

    # Tenant List with Management Options
    st.subheader("📋 Tenant Directory")

    # Filters
    col1, col2, col3 = st.columns(3)

    with col1:
        status_filter = st.selectbox(
            "Filter by Status",
            ["All Status", "Active", "Trial", "Suspended", "Inactive"],
        )

    with col2:
        plan_filter = st.selectbox(
            "Filter by Plan",
            ["All Plans", "Starter", "Professional", "Enterprise", "Custom"],
        )

    with col3:
        search_term = st.text_input("Search Tenants")

    # Filter tenants
    filtered_tenants = filter_tenants(tenant_data, status_filter, plan_filter, search_term)

    # Display tenant cards
    for _, tenant in filtered_tenants.iterrows():
        with st.expander(f"🏢 {tenant['tenant_name']} - {tenant['plan_tier']}"):
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Status", tenant["status"])
                st.metric(
                    "Users",
                    f"{tenant['active_users']}/{tenant['max_users']}",
                )
                st.metric("Utilization", f"{tenant['utilization']:.1f}%")

            with col2:
                st.metric("MRR", f"${tenant['mrr']:,.0f}")
                st.metric("Contract Value", f"${tenant['contract_value']:,.0f}")
                st.metric("Health Score", f"{tenant['health_score']:.1f}/100")

            with col3:
                st.write(f"**Domain**: {tenant['domain']}")
                st.write(f"**Industry**: {tenant['industry']}")
                st.write(f"**Created**: {tenant['created_date']}")
                st.write(f"**Renewal Date**: {tenant['renewal_date']}")

            # Action buttons
            col1, col2, col3, col4, col5 = st.columns(5)

            with col1:
                if st.button("👁️ View", key=f"view_{tenant['tenant_id']}"):
                    st.session_state.current_tenant = tenant["tenant_name"]
                    st.success(f"Switched to tenant: {tenant['tenant_name']}")

            with col2:
                if st.button("⚙️ Configure", key=f"config_{tenant['tenant_id']}"):
                    st.info(f"Opening configuration for {tenant['tenant_name']}")

            with col3:
                if st.button("📊 Analytics", key=f"analytics_{tenant['tenant_id']}"):
                    st.info(f"Showing analytics for {tenant['tenant_name']}")

            with col4:
                if st.button("🔧 Update", key=f"update_{tenant['tenant_id']}"):
                    st.info(f"Updating {tenant['tenant_name']}")

            with col5:
                if tenant["status"] == "Active":
                    if st.button("⏸️ Suspend", key=f"suspend_{tenant['tenant_id']}"):
                        st.warning(f"Suspending tenant: {tenant['tenant_name']}")
                else:
                    if st.button("▶️ Activate", key=f"activate_{tenant['tenant_id']}"):
                        st.success(f"Activating tenant: {tenant['tenant_name']}")


def render_tenant_configuration(analytics, data_gen):
    """Tab 3: Tenant configuration and settings"""

    st.subheader("⚙️ Tenant Configuration & Settings")

    # Select tenant to configure
    tenant_data = generate_tenant_data(data_gen)
    tenant_options = tenant_data["tenant_name"].tolist()

    selected_tenant = st.selectbox("Select Tenant to Configure", tenant_options)

    if selected_tenant:
        tenant_config = get_tenant_configuration(selected_tenant)

        # Configuration Tabs
        config_tab1, config_tab2, config_tab3, config_tab4 = st.tabs(
            [
                "🛠 General Settings",
                "📊 Business Rules",
                "🔐 Security",
                "📈 Performance",
            ]
        )

        with config_tab1:
            render_general_settings(tenant_config, selected_tenant)

        with config_tab2:
            render_business_rules(tenant_config, selected_tenant)

        with config_tab3:
            render_security_settings(tenant_config, selected_tenant)

        with config_tab4:
            render_performance_settings(tenant_config, selected_tenant)


def render_performance_analytics(analytics, data_gen):
    """Tab 4: Tenant performance analytics"""

    st.subheader("📈 Tenant Performance Analytics")

    tenant_data = generate_tenant_data(data_gen)
    analytics_data = generate_tenant_analytics_data(data_gen)

    # Performance Metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        avg_health_score = calculate_avg_health_score(tenant_data)
        st.metric("Avg Health Score", f"{avg_health_score:.1f}/100")

    with col2:
        total_mrr = calculate_total_mrr(tenant_data)
        st.metric("Total MRR", f"${total_mrr:,.0f}")

    with col3:
        churn_rate = calculate_churn_rate(analytics_data)
        st.metric("Monthly Churn Rate", f"{churn_rate}%", delta_color="inverse")

    with col4:
        growth_rate = calculate_growth_rate(analytics_data)
        st.metric("Growth Rate", f"{growth_rate}%")


    # Tenant Performance Analysis
    st.subheader("📊 Tenant Performance Analysis")

    col1, col2 = st.columns(2)

    with col1:
        health_fig = create_health_score_chart(tenant_data)
        st.plotly_chart(health_fig, use_container_width=True)

    with col2:
        revenue_fig = create_revenue_trend_chart(analytics_data)
        st.plotly_chart(revenue_fig, use_container_width=True)

    # Usage Analytics
    st.subheader("🔍 Usage Analytics")

    col1, col2 = st.columns(2)

    with col1:
        usage_fig = create_usage_analysis_chart(analytics_data)
        st.plotly_chart(usage_fig, use_container_width=True)

    with col2:
        module_usage_fig = create_module_usage_chart(analytics_data)
        st.plotly_chart(module_usage_fig, use_container_width=True)

    # Tenant Segmentation
    st.subheader("🎯 Tenant Segmentation")

    segmentation_data = generate_segmentation_data(tenant_data)
    display_tenant_segmentation(segmentation_data)

    # Performance Recommendations
    st.subheader("💡 Performance Recommendations")

    recommendations = generate_performance_recommendations(tenant_data)

    for rec in recommendations:
        with st.expander(f"🎯 {rec['tenant_name']} - {rec['recommendation_type']}"):
            col1, col2 = st.columns(2)

            with col1:
                st.write(f"**Issue**: {rec['issue']}")
                st.write(f"**Recommended Action**: {rec['recommended_action']}")
                st.write(f"**Expected Impact**: {rec['expected_impact']}")

            with col2:
                st.write(f"**Implementation Complexity**: {rec['complexity']}")
                st.write(f"**Timeline**: {rec['timeline']}")
                st.write(f"**Confidence**: {rec['confidence']}%")

            if st.button(
                f"Implement for {rec['tenant_name']}", key=rec["tenant_name"]
            ):
                st.success(f"Implementation started for {rec['tenant_name']}")


# Data Generation Functions
def generate_tenant_data(data_gen):
    """Generate comprehensive tenant data"""

    np.random.seed(42)

    tenants = [
        "ELORA Holding",
        "Naivas Supermarkets",
        "QuickMart Kenya",
        "Chandarana FoodPlus",
        "Eastmatt Superstores",
        "Uchumi Supermarkets",
        "Tuskys",
        "Carrefour Kenya",
    ]

    tenant_data = []

    for i, tenant in enumerate(tenants):
        tenant_info = {
            "tenant_id": f"T{1000 + i}",
            "tenant_name": tenant,
            "domain": f"{tenant.lower().replace(' ', '')}.com",
            "status": np.random.choice(
                ["Active", "Trial", "Suspended", "Inactive"],
                p=[0.7, 0.1, 0.1, 0.1],
            ),
            "plan_tier": np.random.choice(
                ["Starter", "Professional", "Enterprise", "Custom"],
                p=[0.3, 0.4, 0.2, 0.1],
            ),
            "industry": np.random.choice(
                ["Retail", "Manufacturing", "Logistics", "Healthcare"]
            ),
            "active_users": np.random.randint(5, 50),
            "max_users": np.random.randint(10, 100),
            "utilization": np.random.uniform(45, 95),
            "mrr": np.random.uniform(1000, 10000),
            "contract_value": np.random.uniform(5000, 50000),
            "health_score": np.random.uniform(65, 95),
            "created_date": (
                datetime.now() - timedelta(days=np.random.randint(30, 730))
            ).strftime("%Y-%m-%d"),
            "renewal_date": (
                datetime.now() + timedelta(days=np.random.randint(30, 365))
            ).strftime("%Y-%m-%d"),
            "data_usage_gb": np.random.uniform(10, 500),
            "api_calls": np.random.randint(1000, 50000),
        }

        tenant_data.append(tenant_info)

    return pd.DataFrame(tenant_data)


def generate_system_data(data_gen):
    """Generate system performance data"""

    np.random.seed(42)

    return {
        "system_health": 92.5,
        "active_sessions": 245,
        "avg_response_time": 1.2,
        "error_rate": 0.8,
        "database_size_gb": 45.8,
        "backup_status": "Healthy",
    }


def generate_tenant_analytics_data(data_gen):
    """Generate tenant analytics data"""

    np.random.seed(42)
    months = 12

    analytics_data = []

    for month in range(months):
        month_date = datetime.now() - timedelta(days=30 * (months - month - 1))
        analytics_data.append(
            {
                "month": month_date.strftime("%Y-%m"),
                "total_mrr": np.random.uniform(35000, 50000),
                "active_tenants": np.random.randint(6, 9),
                "new_tenants": np.random.randint(0, 3),
                "churned_tenants": np.random.randint(0, 2),
                "avg_utilization": np.random.uniform(70, 85),
                "total_api_calls": np.random.randint(200000, 400000),
            }
        )

    return pd.DataFrame(analytics_data)


# Analytical Functions
def calculate_avg_utilization(tenant_data):
    """Calculate average tenant utilization"""
    return round(tenant_data["utilization"].mean(), 1)


def calculate_system_health(system_data):
    """Calculate system health score"""
    return system_data["system_health"]


def calculate_avg_health_score(tenant_data):
    """Calculate average tenant health score"""
    return round(tenant_data["health_score"].mean(), 1)


def calculate_total_mrr(tenant_data):
    """Calculate total monthly recurring revenue"""
    return round(tenant_data["mrr"].sum())


def calculate_churn_rate(analytics_data):
    """Calculate monthly churn rate (simulated)"""
    return round(np.random.uniform(1.5, 4.5), 1)


def calculate_growth_rate(analytics_data):
    """Calculate growth rate (simulated)"""
    return round(np.random.uniform(8.5, 15.2), 1)


# Visualization Functions
def create_tenant_distribution_chart(tenant_data):
    """Create tenant distribution chart"""
    plan_distribution = tenant_data["plan_tier"].value_counts()

    fig = px.pie(
        values=plan_distribution.values,
        names=plan_distribution.index,
        title="Tenant Distribution by Plan Tier",
    )

    return fig


def create_tenant_performance_chart(tenant_data):
    """Create tenant performance chart"""
    fig = px.scatter(
        tenant_data,
        x="utilization",
        y="health_score",
        size="mrr",
        color="plan_tier",
        title="Tenant Performance Analysis",
        hover_name="tenant_name",
    )

    return fig


def create_revenue_analysis_chart(tenant_data):
    """Create revenue analysis chart"""
    revenue_by_tier = (
        tenant_data.groupby("plan_tier")["mrr"].sum().reset_index()
    )

    fig = px.bar(
        revenue_by_tier,
        x="plan_tier",
        y="mrr",
        title="Monthly Revenue by Plan Tier",
        color="mrr",
        color_continuous_scale="RdYlGn",
    )

    return fig


def create_health_score_chart(tenant_data):
    """Create health score chart"""
    fig = px.box(
        tenant_data,
        x="plan_tier",
        y="health_score",
        title="Health Score Distribution by Plan Tier",
    )

    return fig


def create_revenue_trend_chart(analytics_data):
    """Create revenue trend chart"""
    fig = px.line(
        analytics_data,
        x="month",
        y="total_mrr",
        title="Monthly Recurring Revenue Trend",
        markers=True,
    )

    return fig


def create_usage_analysis_chart(analytics_data):
    """Create usage analysis chart"""
    fig = px.area(
        analytics_data,
        x="month",
        y="total_api_calls",
        title="API Usage Trend",
    )

    return fig


def create_module_usage_chart(analytics_data):
    """Create module usage chart (simulated)"""
    modules = ["Sales", "Inventory", "Logistics", "Procurement", "Analytics"]
    usage = [85, 78, 65, 45, 35]  # Percentage of tenants using each module

    fig = px.bar(
        x=modules,
        y=usage,
        title="Module Adoption Rate",
        color=usage,
        color_continuous_scale="RdYlGn",
        labels={"x": "Module", "y": "Adoption (%)"},
    )

    return fig


# Display Functions
def display_system_alerts(tenant_data):
    """Display system alerts"""
    low_health = tenant_data[tenant_data["health_score"] < 70]
    high_utilization = tenant_data[tenant_data["utilization"] > 90]

    if len(low_health) == 0 and len(high_utilization) == 0:
        st.success("✅ No critical system alerts")
        return

    for _, tenant in low_health.head(2).iterrows():
        st.error(f"🔴 {tenant['tenant_name']} - Low Health Score")
        st.caption(f"Health: {tenant['health_score']:.1f}/100 | Review needed")

    for _, tenant in high_utilization.head(2).iterrows():
        st.warning(f"🟠 {tenant['tenant_name']} - High Utilization")
        st.caption(
            f"Utilization: {tenant['utilization']:.1f}% | Consider plan upgrade"
        )


def display_realtime_system_monitor(system_data):
    """Display real-time system monitoring"""
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Active Sessions", system_data["active_sessions"])
    with col2:
        st.metric(
            "Avg Response Time",
            f"{system_data['avg_response_time']}s",
        )
    with col3:
        st.metric(
            "Error Rate",
            f"{system_data['error_rate']}%",
            delta_color="inverse",
        )
    with col4:
        st.metric(
            "Database Size",
            f"{system_data['database_size_gb']} GB",
        )


def filter_tenants(tenant_data, status_filter, plan_filter, search_term):
    """Filter tenants based on criteria"""
    filtered = tenant_data.copy()

    if status_filter != "All Status":
        filtered = filtered[filtered["status"] == status_filter]

    if plan_filter != "All Plans":
        filtered = filtered[filtered["plan_tier"] == plan_filter]

    if search_term:
        filtered = filtered[
            filtered["tenant_name"].str.contains(search_term, case=False)
        ]

    return filtered


def create_new_tenant(
    tenant_name,
    tenant_domain,
    plan_tier,
    industry,
    admin_email,
    max_users,
    contract_value,
    contract_duration,
    modules_enabled,
    data_retention,
    api_access,
    custom_branding,
    sso_enabled,
    backup_frequency,
):
    """Create new tenant (stub)"""
    # In a real implementation, this would create the tenant in the database
    return {
        "tenant_name": tenant_name,
        "domain": tenant_domain,
        "plan_tier": plan_tier,
        "status": "Active",
    }


def get_tenant_configuration(tenant_name):
    """Get tenant configuration (simulated)"""
    return {
        "general": {
            "tenant_name": tenant_name,
            "domain": f"{tenant_name.lower().replace(' ', '')}.com",
            "timezone": "Africa/Nairobi",
            "currency": "KES",
            "language": "English",
        },
        "business_rules": {
            "auto_approve_orders": True,
            "min_stock_level": 50,
            "max_credit_limit": 100000,
            "payment_terms": "Net 30",
        },
        "security": {
            "sso_enabled": False,
            "mfa_required": True,
            "session_timeout": 60,
            "ip_restrictions": [],
        },
        "performance": {
            "data_retention": 12,
            "backup_frequency": "Daily",
            "api_rate_limit": 1000,
            "cache_enabled": True,
        },
    }


def render_general_settings(tenant_config, tenant_name):
    """Render general settings configuration"""
    st.subheader("🛠 General Settings")

    with st.form("general_settings"):
        col1, col2 = st.columns(2)

        with col1:
            tenant_name_val = st.text_input(
                "Tenant Name", value=tenant_config["general"]["tenant_name"]
            )
            domain = st.text_input(
                "Domain", value=tenant_config["general"]["domain"]
            )
            timezone = st.selectbox(
                "Timezone",
                ["Africa/Nairobi", "UTC", "EST", "PST"],
                index=0
                if tenant_config["general"]["timezone"] == "Africa/Nairobi"
                else 1,
            )

        with col2:
            currency = st.selectbox(
                "Currency",
                ["KES", "USD", "EUR", "GBP"],
                index=0
                if tenant_config["general"]["currency"] == "KES"
                else 1,
            )
            language = st.selectbox(
                "Language",
                ["English", "Swahili", "French", "Spanish"],
                index=0
                if tenant_config["general"]["language"] == "English"
                else 1,
            )

        save_general = st.form_submit_button("💾 Save General Settings")

        if save_general:
            # In real implementation, persist settings
            st.success("General settings saved successfully!")


def render_business_rules(tenant_config, tenant_name):
    """Render business rules configuration"""
    st.subheader("📊 Business Rules")

    with st.form("business_rules"):
        col1, col2 = st.columns(2)

        with col1:
            auto_approve = st.checkbox(
                "Auto-approve Orders",
                value=tenant_config["business_rules"]["auto_approve_orders"],
            )
            min_stock = st.number_input(
                "Minimum Stock Level",
                value=tenant_config["business_rules"]["min_stock_level"],
            )
            max_credit = st.number_input(
                "Maximum Credit Limit",
                value=tenant_config["business_rules"]["max_credit_limit"],
            )

        with col2:
            payment_terms = st.selectbox(
                "Payment Terms",
                ["Net 15", "Net 30", "Net 45", "Net 60"],
                index=1
                if tenant_config["business_rules"]["payment_terms"]
                == "Net 30"
                else 0,
            )
            approval_threshold = st.number_input(
                "Approval Threshold ($)", value=5000
            )
            discount_policy = st.selectbox(
                "Discount Policy", ["Standard", "Flexible", "Restricted"]
            )

        save_rules = st.form_submit_button("💾 Save Business Rules")

        if save_rules:
            # In real implementation, persist settings
            st.success("Business rules saved successfully!")


def render_security_settings(tenant_config, tenant_name):
    """Render security settings configuration"""
    st.subheader("🔐 Security Settings")

    with st.form("security_settings"):
        col1, col2 = st.columns(2)

        with col1:
            sso_enabled = st.checkbox(
                "Enable Single Sign-On",
                value=tenant_config["security"]["sso_enabled"],
            )
            mfa_required = st.checkbox(
                "Require MFA",
                value=tenant_config["security"]["mfa_required"],
            )
            session_timeout = st.slider(
                "Session Timeout (minutes)",
                15,
                240,
                tenant_config["security"]["session_timeout"],
            )

        with col2:
            ip_restrictions = st.text_area(
                "IP Restrictions (one per line)",
                value="\n".join(
                    tenant_config["security"]["ip_restrictions"]
                ),
            )
            audit_logging = st.checkbox("Enable Audit Logging", value=True)
            data_encryption = st.checkbox(
                "Enable Data Encryption", value=True
            )

        save_security = st.form_submit_button("💾 Save Security Settings")

        if save_security:
            # In real implementation, persist settings
            st.success("Security settings saved successfully!")


def render_performance_settings(tenant_config, tenant_name):
    """Render performance settings configuration"""
    st.subheader("📈 Performance Settings")

    with st.form("performance_settings"):
        col1, col2 = st.columns(2)

        with col1:
            data_retention = st.slider(
                "Data Retention (months)",
                1,
                36,
                tenant_config["performance"]["data_retention"],
            )
            backup_frequency = st.selectbox(
                "Backup Frequency",
                ["Daily", "Weekly", "Monthly"],
                index=0
                if tenant_config["performance"]["backup_frequency"]
                == "Daily"
                else 1,
            )
            api_rate_limit = st.number_input(
                "API Rate Limit (requests/hour)",
                value=tenant_config["performance"]["api_rate_limit"],
            )

        with col2:
            cache_enabled = st.checkbox(
                "Enable Caching",
                value=tenant_config["performance"]["cache_enabled"],
            )
            compression_enabled = st.checkbox(
                "Enable Data Compression", value=True
            )
            query_timeout = st.number_input(
                "Query Timeout (seconds)", value=30
            )

        save_performance = st.form_submit_button(
            "💾 Save Performance Settings"
        )

        if save_performance:
            # In real implementation, persist settings
            st.success("Performance settings saved successfully!")


def generate_segmentation_data(tenant_data):
    """Generate tenant segmentation data"""
    segments = [
        {
            "segment": "High Value",
            "criteria": "MRR > $5,000 & Health > 80",
            "tenants": 3,
            "avg_mrr": 7850,
        },
        {
            "segment": "Growth",
            "criteria": "Utilization > 75% & Recent signup",
            "tenants": 2,
            "avg_mrr": 3200,
        },
        {
            "segment": "At Risk",
            "criteria": "Health < 70 or Utilization < 50%",
            "tenants": 1,
            "avg_mrr": 1500,
        },
        {
            "segment": "Stable",
            "criteria": "All other tenants",
            "tenants": 2,
            "avg_mrr": 4200,
        },
    ]

    return pd.DataFrame(segments)


def display_tenant_segmentation(segmentation_data):
    """Display tenant segmentation"""
    for _, segment in segmentation_data.iterrows():
        with st.expander(
            f"🎯 {segment['segment']} - {segment['tenants']} tenants"
        ):
            col1, col2, col3 = st.columns(3)

            with col1:
                st.write(f"**Criteria**: {segment['criteria']}")
            with col2:
                st.write(f"**Tenant Count**: {segment['tenants']}")
            with col3:
                st.write(f"**Avg MRR**: ${segment['avg_mrr']:,.0f}")


def generate_performance_recommendations(tenant_data):
    """Generate performance recommendations (simulated)"""
    return [
        {
            "tenant_name": "QuickMart Kenya",
            "recommendation_type": "Plan Upgrade",
            "issue": "95% utilization, approaching capacity limits",
            "recommended_action": "Upgrade to Enterprise plan",
            "expected_impact": "$2,500 additional MRR, better performance",
            "complexity": "Low",
            "timeline": "2-4 weeks",
            "confidence": 92,
        },
        {
            "tenant_name": "Uchumi Supermarkets",
            "recommendation_type": "Health Improvement",
            "issue": "Low health score (68/100), declining usage",
            "recommended_action": "Engagement campaign and feature training",
            "expected_impact": (
                "15% health score improvement, reduced churn risk"
            ),
            "complexity": "Medium",
            "timeline": "4-8 weeks",
            "confidence": 78,
        },
    ]


if __name__ == "__main__":
    render()
