# logistics_pro/pages/19_Admin_Users.py
import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime, timedelta
import numpy as np


def render():
    """Admin User Management - ENTERPRISE VERSION"""

    st.title("👥 Admin User Management")
    st.markdown(
        f"""
    <div style="background: #e0f2fe; padding: 15px; border-radius: 8px; border-left: 4px solid #1f77b4; margin-bottom: 20px;">
        <strong>📍 Location:</strong> System Administration &gt; User Management | 
        <strong>Tenant:</strong> {st.session_state.get('current_tenant', 'ELORA Holding')}
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Data initialization and checks
    if "analytics" not in st.session_state:
        st.error("❌ Please go to the main dashboard first to initialize data.")
        return

    analytics = st.session_state.analytics
    data_gen = st.session_state.get("data_gen", None)

    # Main Tab Structure (4 tabs standard)
    tab1, tab2, tab3, tab4 = st.tabs(
        [
            "📊 User Dashboard",
            "👤 User Management",
            "🔐 Security & Access",
            "📈 Activity Analytics",
        ]
    )

    with tab1:
        render_user_dashboard(analytics, data_gen)
    with tab2:
        render_user_management(analytics, data_gen)
    with tab3:
        render_security_access(analytics, data_gen)
    with tab4:
        render_activity_analytics(analytics, data_gen)


def render_user_dashboard(analytics, data_gen):
    """Tab 1: Comprehensive user overview dashboard"""

    # System Insights expander
    with st.expander("🧠 USER MANAGEMENT INSIGHTS", expanded=True):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "👥 Total Users",
                "142",
                "+8",
                help="Total number of users across all tenants",
            )
        with col2:
            st.metric(
                "🔐 Security Score",
                "92.5%",
                "3.2%",
                help="Overall system security score based on user practices",
            )
        with col3:
            st.metric(
                "📊 Active Sessions",
                "89",
                "+12",
                help="Current active user sessions across the platform",
            )

        st.info(
            "💡 **Security Recommendation**: 15 users haven't enabled MFA. "
            "Consider enforcing MFA policy to improve security score by ~8% and reduce risk by ~45%."
        )

    # Generate user and security data
    user_data = generate_user_data(data_gen)
    security_data = generate_security_data(data_gen)

    # Top KPIs
    st.subheader("🎯 User Management Dashboard")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        total_users = len(user_data)
        st.metric("Total Users", f"{total_users}")

    with col2:
        active_users = len(user_data[user_data["status"] == "Active"])
        st.metric("Active Users", f"{active_users}")

    with col3:
        mfa_adoption = calculate_mfa_adoption(user_data)
        st.metric("MFA Adoption", f"{mfa_adoption}%")

    with col4:
        avg_login_frequency = calculate_avg_login_frequency(user_data)
        st.metric("Avg Logins/Week", f"{avg_login_frequency}")

    # User Distribution Overview
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("📊 User Distribution & Roles")
        distribution_fig = create_user_distribution_chart(user_data)
        st.plotly_chart(distribution_fig, use_container_width=True)

    with col2:
        st.subheader("🚨 Security Alerts")
        display_security_alerts(user_data)

    # User Activity Analysis
    st.subheader("📈 User Activity Analysis")

    col1, col2 = st.columns(2)

    with col1:
        activity_fig = create_user_activity_chart(user_data)
        st.plotly_chart(activity_fig, use_container_width=True)

    with col2:
        role_fig = create_role_analysis_chart(user_data)
        st.plotly_chart(role_fig, use_container_width=True)

    # Real-time User Monitoring
    st.subheader("🔍 Real-time User Monitoring")
    display_realtime_user_monitor(user_data)


def render_user_management(analytics, data_gen):
    """Tab 2: User management and operations"""

    st.subheader("👤 User Management & Operations")

    user_data = generate_user_data(data_gen)

    # User Management Actions
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        if st.button("➕ Create User", use_container_width=True):
            st.session_state.show_user_creation = True

    with col2:
        if st.button("📥 Bulk Import", use_container_width=True):
            st.info("Bulk user import initiated...")

    with col3:
        if st.button("📧 Send Invites", use_container_width=True):
            st.info("User invitation process started...")

    with col4:
        if st.button("📊 Export Users", use_container_width=True):
            st.success("User data exported successfully!")

    # User Creation Form
    if st.session_state.get("show_user_creation", False):
        with st.form("create_user"):
            st.subheader("👤 Create New User")

            col1, col2 = st.columns(2)

            with col1:
                first_name = st.text_input("First Name*")
                last_name = st.text_input("Last Name*")
                email = st.text_input("Email Address*")
                phone = st.text_input("Phone Number")

            with col2:
                username = st.text_input("Username*")
                tenant = st.selectbox(
                    "Assign to Tenant*",
                    [
                        "ELORA Holding",
                        "Naivas Supermarkets",
                        "QuickMart Kenya",
                        "Chandarana FoodPlus",
                    ],
                )
                department = st.selectbox(
                    "Department*",
                    [
                        "Sales",
                        "Operations",
                        "Logistics",
                        "Procurement",
                        "Finance",
                        "Admin",
                    ],
                )
                job_title = st.text_input("Job Title*")

            # Role and Permissions
            st.subheader("🎯 Role & Permissions")

            col1, col2 = st.columns(2)

            with col1:
                user_role = st.selectbox(
                    "User Role*",
                    [
                        "System Admin",
                        "Tenant Admin",
                        "Manager",
                        "Analyst",
                        "Viewer",
                        "Custom",
                    ],
                )

                if user_role == "Custom":
                    st.multiselect(
                        "Custom Permissions",
                        [
                            "read_sales",
                            "write_sales",
                            "read_inventory",
                            "write_inventory",
                            "read_logistics",
                            "write_logistics",
                            "read_procurement",
                            "write_procurement",
                            "user_management",
                            "system_config",
                        ],
                    )

            with col2:
                modules_access = st.multiselect(
                    "Module Access*",
                    [
                        "Sales Intelligence",
                        "Inventory Management",
                        "Logistics",
                        "Procurement",
                        "Analytics",
                    ],
                    default=["Sales Intelligence", "Inventory Management"],
                )

                data_access = st.selectbox(
                    "Data Access Level*",
                    ["All Data", "Tenant Data", "Department Data", "Personal Data"],
                )

            # Security Settings
            st.subheader("🔐 Security Settings")

            col1, col2 = st.columns(2)

            with col1:
                require_mfa = st.checkbox("Require MFA", value=True)
                force_password_change = st.checkbox(
                    "Force Password Change on First Login", value=True
                )
                session_timeout = st.slider(
                    "Session Timeout (minutes)", 15, 240, 60
                )

            with col2:
                login_hours = st.select_slider(
                    "Allowed Login Hours",
                    options=["24/7", "Business Hours", "Custom"],
                    value="24/7",
                )

                if login_hours == "Custom":
                    st.time_input(
                        "Start Time",
                        value=datetime.strptime("08:00", "%H:%M"),
                    )
                    st.time_input(
                        "End Time",
                        value=datetime.strptime("18:00", "%H:%M"),
                    )

            col1, col2 = st.columns(2)
            with col1:
                create_user_btn = st.form_submit_button(
                    "🚀 Create User", use_container_width=True
                )
            with col2:
                cancel_creation = st.form_submit_button(
                    "❌ Cancel", use_container_width=True
                )

            if create_user_btn:
                if all([first_name, last_name, email, username, job_title]):
                    _ = create_new_user(
                        first_name,
                        last_name,
                        email,
                        phone,
                        username,
                        tenant,
                        department,
                        job_title,
                        user_role,
                        modules_access,
                        data_access,
                        require_mfa,
                        force_password_change,
                        session_timeout,
                        login_hours,
                    )
                    st.success(
                        f"✅ User '{first_name} {last_name}' created successfully!"
                    )
                    st.session_state.show_user_creation = False
                else:
                    st.error(
                        "❌ Please fill in all required fields (marked with *)."
                    )

            if cancel_creation:
                st.session_state.show_user_creation = False
                st.info("User creation cancelled")

    # User List with Management Options
    st.subheader("📋 User Directory")

    # Filters
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        status_filter = st.selectbox(
            "Filter by Status",
            ["All Status", "Active", "Inactive", "Suspended", "Pending"],
        )

    with col2:
        role_filter = st.selectbox(
            "Filter by Role",
            [
                "All Roles",
                "System Admin",
                "Tenant Admin",
                "Manager",
                "Analyst",
                "Viewer",
            ],
        )

    with col3:
        tenant_filter = st.selectbox(
            "Filter by Tenant",
            [
                "All Tenants",
                "ELORA Holding",
                "Naivas Supermarkets",
                "QuickMart Kenya",
                "Chandarana FoodPlus",
            ],
        )

    with col4:
        search_term = st.text_input("Search Users")

    # Filter users
    filtered_users = filter_users(
        user_data, status_filter, role_filter, tenant_filter, search_term
    )

    # Display user cards
    for _, user in filtered_users.iterrows():
        with st.expander(f"👤 {user['full_name']} - {user['role']}"):
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Status", user["status"])
                st.metric("Last Login", user["last_login"])
                st.metric("Login Count", user["login_count"])

            with col2:
                st.metric("MFA Enabled", "Yes" if user["mfa_enabled"] else "No")
                st.metric("Failed Logins", user["failed_logins"])
                st.metric("Tenant", user["tenant"])

            with col3:
                st.write(f"**Department**: {user['department']}")
                st.write(f"**Job Title**: {user['job_title']}")
                st.write(f"**Created**: {user['created_date']}")
                st.write(f"**Modules**: {user['modules']}")

            # Action buttons
            col1, col2, col3, col4, col5, col6 = st.columns(6)

            with col1:
                if st.button("✏️ Edit", key=f"edit_{user['user_id']}"):
                    st.info(f"Editing user: {user['full_name']}")

            with col2:
                if st.button("🔄 Reset PW", key=f"reset_{user['user_id']}"):
                    st.warning(f"Password reset for: {user['full_name']}")

            with col3:
                if st.button("🔐 MFA", key=f"mfa_{user['user_id']}"):
                    if user["mfa_enabled"]:
                        st.warning(
                            f"Disabling MFA for: {user['full_name']}"
                        )
                    else:
                        st.info(f"Enabling MFA for: {user['full_name']}")

            with col4:
                if user["status"] == "Active":
                    if st.button("⏸️ Suspend", key=f"suspend_{user['user_id']}"):
                        st.warning(f"Suspending user: {user['full_name']}")
                else:
                    if st.button("▶️ Activate", key=f"activate_{user['user_id']}"):
                        st.success(f"Activating user: {user['full_name']}")

            with col5:
                if st.button("📊 Activity", key=f"activity_{user['user_id']}"):
                    st.info(f"Showing activity for: {user['full_name']}")

            with col6:
                if st.button("🗑️ Delete", key=f"delete_{user['user_id']}"):
                    st.error(f"Deleting user: {user['full_name']}")


def render_security_access(analytics, data_gen):
    """Tab 3: Security and access control management"""

    st.subheader("🔐 Security & Access Control")

    user_data = generate_user_data(data_gen)
    security_data = generate_security_data(data_gen)

    # Security Overview
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        security_score = calculate_security_score(security_data)
        st.metric("Security Score", f"{security_score}%")

    with col2:
        mfa_rate = calculate_mfa_adoption(user_data)
        st.metric("MFA Adoption", f"{mfa_rate}%")

    with col3:
        password_strength = calculate_password_strength(security_data)
        st.metric("Password Strength", f"{password_strength}%")

    with col4:
        suspicious_activity = detect_suspicious_activity(security_data)
        st.metric(
            "Suspicious Activities",
            f"{suspicious_activity}",
            delta_color="inverse",
        )

    # Security Policies
    st.subheader("📋 Security Policies")

    with st.form("security_policies"):
        col1, col2 = st.columns(2)

        with col1:
            mfa_policy = st.selectbox(
                "MFA Policy",
                [
                    "Optional",
                    "Recommended",
                    "Required for Admins",
                    "Required for All",
                ],
            )
            password_policy = st.selectbox(
                "Password Policy",
                ["Basic", "Standard", "Strong", "Very Strong"],
            )
            session_timeout = st.slider(
                "Global Session Timeout (minutes)", 15, 240, 60
            )

        with col2:
            max_login_attempts = st.number_input(
                "Max Login Attempts", min_value=3, max_value=10, value=5
            )
            account_lockout_duration = st.number_input(
                "Account Lockout (minutes)",
                min_value=15,
                max_value=1440,
                value=30,
            )
            password_expiry = st.selectbox(
                "Password Expiry",
                ["30 days", "60 days", "90 days", "180 days", "Never"],
            )

        save_policies = st.form_submit_button("💾 Save Security Policies")

        if save_policies:
            # In a real app, persist these in configuration storage
            st.success("Security policies updated successfully!")

    # Access Control Lists
    st.subheader("🎯 Access Control Lists")

    # Role-based Access Control
    st.write("**Role Permissions Matrix**")

    roles_data = generate_roles_data()
    display_roles_matrix(roles_data)

    # API Access Management
    st.subheader("🔓 API Access Management")

    api_keys = generate_api_keys_data()

    for api_key in api_keys:
        with st.expander(f"🔑 {api_key['name']} - {api_key['status']}"):
            col1, col2 = st.columns(2)

            with col1:
                st.write(f"**Key ID**: {api_key['key_id']}")
                st.write(f"**Permissions**: {api_key['permissions']}")
                st.write(f"**Created**: {api_key['created_date']}")

            with col2:
                st.write(f"**Last Used**: {api_key['last_used']}")
                st.write(f"**Usage Count**: {api_key['usage_count']}")
                st.write(f"**Rate Limit**: {api_key['rate_limit']}/hour")

            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button("Regenerate", key=f"regen_{api_key['key_id']}"):
                    st.warning(f"Regenerating API key: {api_key['name']}")
            with col2:
                if st.button("Revoke", key=f"revoke_{api_key['key_id']}"):
                    st.error(f"Revoking API key: {api_key['name']}")
            with col3:
                if st.button("Copy", key=f"copy_{api_key['key_id']}"):
                    st.info(
                        f"API key copied to clipboard (placeholder): {api_key['name']}"
                    )

    # Security Audit Log
    st.subheader("🧾 Security Audit Log")

    audit_log = generate_audit_log()
    display_audit_log(audit_log)


def render_activity_analytics(analytics, data_gen):
    """Tab 4: User activity analytics and monitoring"""

    st.subheader("📈 User Activity Analytics")

    user_data = generate_user_data(data_gen)
    activity_data = generate_activity_data(data_gen)

    # Activity Metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        daily_active_users = calculate_daily_active_users(activity_data)
        st.metric("Daily Active Users", f"{daily_active_users}")

    with col2:
        weekly_active_users = calculate_weekly_active_users(activity_data)
        st.metric("Weekly Active Users", f"{weekly_active_users}")

    with col3:
        avg_session_duration = calculate_avg_session_duration(activity_data)
        st.metric("Avg Session Duration", f"{avg_session_duration}m")

    with col4:
        feature_adoption = calculate_feature_adoption(activity_data)
        st.metric("Feature Adoption", f"{feature_adoption}%")

    # User Activity Trends
    st.subheader("📊 User Activity Trends")

    col1, col2 = st.columns(2)

    with col1:
        activity_trend_fig = create_activity_trend_chart(activity_data)
        st.plotly_chart(activity_trend_fig, use_container_width=True)

    with col2:
        login_pattern_fig = create_login_pattern_chart(activity_data)
        st.plotly_chart(login_pattern_fig, use_container_width=True)

    # Module Usage Analysis
    st.subheader("🔍 Module Usage Analysis")

    module_usage_fig = create_module_usage_chart(activity_data)
    st.plotly_chart(module_usage_fig, use_container_width=True)

    # User Engagement Scoring
    st.subheader("🎯 User Engagement Scoring")

    engagement_data = generate_engagement_data(user_data, activity_data)
    display_engagement_scores(engagement_data)

    # Anomaly Detection
    st.subheader("⚠️ Anomaly Detection")

    anomalies = detect_anomalies(activity_data)
    display_anomalies(anomalies)


# -------------------------
# Data Generation Functions
# -------------------------
def generate_user_data(data_gen):
    """Generate comprehensive user data (synthetic for demo)"""

    np.random.seed(42)

    first_names = [
        "John",
        "Sarah",
        "Michael",
        "Emily",
        "David",
        "Lisa",
        "Robert",
        "Maria",
        "James",
        "Susan",
    ]
    last_names = [
        "Smith",
        "Johnson",
        "Williams",
        "Brown",
        "Jones",
        "Garcia",
        "Miller",
        "Davis",
        "Rodriguez",
        "Martinez",
    ]
    departments = ["Sales", "Operations", "Logistics", "Procurement", "Finance", "Admin"]
    tenants = [
        "ELORA Holding",
        "Naivas Supermarkets",
        "QuickMart Kenya",
        "Chandarana FoodPlus",
    ]

    user_data = []

    for i in range(50):  # Generate 50 users
        first_name = np.random.choice(first_names)
        last_name = np.random.choice(last_names)

        user_info = {
            "user_id": f"U{1000 + i}",
            "username": f"{first_name.lower()}.{last_name.lower()}",
            "first_name": first_name,
            "last_name": last_name,
            "full_name": f"{first_name} {last_name}",
            "email": f"{first_name.lower()}.{last_name.lower()}@company.com",
            "phone": f"+2547{np.random.randint(10000000, 99999999)}",
            "tenant": np.random.choice(tenants),
            "department": np.random.choice(departments),
            "job_title": np.random.choice(
                ["Manager", "Analyst", "Specialist", "Coordinator", "Director"]
            ),
            "role": np.random.choice(
                ["System Admin", "Tenant Admin", "Manager", "Analyst", "Viewer"],
                p=[0.05, 0.1, 0.2, 0.3, 0.35],
            ),
            "status": np.random.choice(
                ["Active", "Inactive", "Suspended", "Pending"],
                p=[0.8, 0.1, 0.05, 0.05],
            ),
            "mfa_enabled": np.random.choice([True, False], p=[0.7, 0.3]),
            "last_login": (
                datetime.now() - timedelta(days=np.random.randint(0, 30))
            ).strftime("%Y-%m-%d %H:%M"),
            "login_count": np.random.randint(1, 500),
            "failed_logins": np.random.randint(0, 5),
            "created_date": (
                datetime.now() - timedelta(days=np.random.randint(1, 365))
            ).strftime("%Y-%m-%d"),
            "modules": np.random.choice(
                ["Sales,Inventory", "Logistics", "Procurement,Analytics", "All"],
                p=[0.3, 0.2, 0.3, 0.2],
            ),
        }

        user_data.append(user_info)

    return pd.DataFrame(user_data)


def generate_security_data(data_gen):
    """Generate security data (synthetic)"""

    np.random.seed(42)

    return {
        "security_score": 92.5,
        "password_strength": 88.2,
        "suspicious_activities": 3,
        "failed_logins_24h": 12,
        "mfa_adoption": 78.5,
    }


def generate_activity_data(data_gen):
    """Generate user activity data (synthetic)"""

    np.random.seed(42)
    days = 30

    activity_data = []

    for day in range(days):
        date = datetime.now() - timedelta(days=days - day - 1)
        activity_data.append(
            {
                "date": date.strftime("%Y-%m-%d"),
                "active_users": np.random.randint(80, 120),
                "total_logins": np.random.randint(200, 400),
                "failed_logins": np.random.randint(5, 20),
                "avg_session_minutes": np.random.uniform(25, 45),
                "api_calls": np.random.randint(5000, 15000),
                "sales_module_usage": np.random.uniform(60, 85),
                "inventory_module_usage": np.random.uniform(55, 80),
                "logistics_module_usage": np.random.uniform(40, 70),
                "procurement_module_usage": np.random.uniform(35, 65),
            }
        )

    return pd.DataFrame(activity_data)


# ------------------------
# Analytical Functions
# ------------------------
def calculate_mfa_adoption(user_data: pd.DataFrame) -> float:
    """Calculate MFA adoption rate (fixed to avoid KeyError / 'False' bug)."""
    if user_data.empty or "mfa_enabled" not in user_data.columns:
        return 0.0
    # Booleans sum directly to counts
    mfa_enabled = int(user_data["mfa_enabled"].sum())
    total = len(user_data)
    if total == 0:
        return 0.0
    return round((mfa_enabled / total) * 100, 1)


def calculate_avg_login_frequency(user_data):
    """Calculate average login frequency (approx weekly)"""
    if user_data.empty:
        return 0.0
    return round(user_data["login_count"].mean() / 4.33, 1)  # Approx weekly average


def calculate_security_score(security_data):
    """Return security score"""
    return float(security_data.get("security_score", 0.0))


def calculate_password_strength(security_data):
    """Return password strength"""
    return float(security_data.get("password_strength", 0.0))


def detect_suspicious_activity(security_data):
    """Return count of suspicious activities"""
    return int(security_data.get("suspicious_activities", 0))


def calculate_daily_active_users(activity_data):
    """Calculate daily active users (latest day)"""
    if activity_data.empty:
        return 0
    return int(activity_data["active_users"].iloc[-1])


def calculate_weekly_active_users(activity_data):
    """Calculate weekly active users (sum of last 7 days)"""
    if activity_data.empty:
        return 0
    return int(activity_data["active_users"].tail(7).sum())


def calculate_avg_session_duration(activity_data):
    """Calculate average session duration"""
    if activity_data.empty:
        return 0.0
    return round(activity_data["avg_session_minutes"].mean(), 1)


def calculate_feature_adoption(activity_data):
    """Calculate feature adoption rate (synthetic)"""
    return round(float(np.random.uniform(65, 85)), 1)


# ------------------------
# Visualization Functions
# ------------------------
def create_user_distribution_chart(user_data):
    """Create user distribution chart"""
    if user_data.empty:
        return px.pie(
            names=["No Data"],
            values=[1],
            title="User Distribution by Role",
        )
    role_distribution = user_data["role"].value_counts()
    fig = px.pie(
        values=role_distribution.values,
        names=role_distribution.index,
        title="User Distribution by Role",
    )
    return fig


def create_user_activity_chart(user_data):
    """Create user activity chart"""
    if user_data.empty:
        return px.scatter(title="User Activity Analysis (No Data)")
    fig = px.scatter(
        user_data,
        x="login_count",
        y="failed_logins",
        size="login_count",
        color="status",
        title="User Activity Analysis",
        hover_name="full_name",
    )
    return fig


def create_role_analysis_chart(user_data):
    """Create role analysis chart"""
    if user_data.empty:
        return px.bar(title="Average Login Activity by Role (No Data)")
    role_analysis = (
        user_data.groupby("role")
        .agg({"login_count": "mean", "failed_logins": "mean", "user_id": "count"})
        .reset_index()
    )

    fig = px.bar(
        role_analysis,
        x="role",
        y="login_count",
        title="Average Login Activity by Role",
        color="failed_logins",
        color_continuous_scale="RdYlGn_r",
    )
    return fig


def create_activity_trend_chart(activity_data):
    """Create activity trend chart"""
    if activity_data.empty:
        return px.line(title="Daily Active Users Trend (No Data)")
    fig = px.line(
        activity_data,
        x="date",
        y="active_users",
        title="Daily Active Users Trend",
        markers=True,
    )
    return fig


def create_login_pattern_chart(activity_data):
    """Create login pattern chart"""
    if activity_data.empty:
        return px.area(title="Login Patterns - Success vs Failed (No Data)")
    fig = px.area(
        activity_data,
        x="date",
        y=["total_logins", "failed_logins"],
        title="Login Patterns - Success vs Failed",
    )
    return fig


def create_module_usage_chart(activity_data):
    """Create module usage chart"""
    if activity_data.empty:
        return px.bar(title="Average Module Usage Rates (No Data)")

    modules = ["Sales", "Inventory", "Logistics", "Procurement"]
    usage = [
        float(activity_data["sales_module_usage"].mean()),
        float(activity_data["inventory_module_usage"].mean()),
        float(activity_data["logistics_module_usage"].mean()),
        float(activity_data["procurement_module_usage"].mean()),
    ]

    fig = px.bar(
        x=modules,
        y=usage,
        title="Average Module Usage Rates",
        color=usage,
        color_continuous_scale="RdYlGn",
    )
    fig.update_layout(xaxis_title="Module", yaxis_title="Usage (%)")
    return fig


# ------------------------
# Display Functions
# ------------------------
def display_security_alerts(user_data):
    """Display security alerts"""
    if user_data.empty:
        st.success("✅ No critical security alerts")
        return

    no_mfa = user_data[
        (user_data["mfa_enabled"] == False) & (user_data["status"] == "Active")
    ]
    high_failed_logins = user_data[user_data["failed_logins"] >= 3]
    # Simplified "inactive" check; in real system we'd use last_login timestamps
    inactive_30_days = user_data[user_data["login_count"] == 0]

    if len(no_mfa) == 0 and len(high_failed_logins) == 0 and len(inactive_30_days) == 0:
        st.success("✅ No critical security alerts")
        return

    for _, user in no_mfa.head(2).iterrows():
        st.error(f"🔴 {user['full_name']} - MFA Not Enabled")
        st.caption(f"Role: {user['role']} | Security risk")

    for _, user in high_failed_logins.head(2).iterrows():
        st.warning(f"🟡 {user['full_name']} - High Failed Logins")
        st.caption(
            f"Failed attempts: {user['failed_logins']} | Possible credential stuffing"
        )


def display_realtime_user_monitor(user_data):
    """Display real-time user monitoring"""
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        current_sessions = np.random.randint(80, 100)
        st.metric("Current Sessions", current_sessions)
    with col2:
        st.metric("New Today", "12")
    with col3:
        st.metric("Password Resets", "3")
    with col4:
        st.metric("Locked Accounts", "2", delta_color="inverse")


def filter_users(user_data, status_filter, role_filter, tenant_filter, search_term):
    """Filter users based on criteria"""
    filtered = user_data.copy()

    if status_filter != "All Status":
        filtered = filtered[filtered["status"] == status_filter]

    if role_filter != "All Roles":
        filtered = filtered[filtered["role"] == role_filter]

    if tenant_filter != "All Tenants":
        filtered = filtered[filtered["tenant"] == tenant_filter]

    if search_term:
        mask = (
            filtered["full_name"].str.contains(search_term, case=False)
            | filtered["email"].str.contains(search_term, case=False)
            | filtered["username"].str.contains(search_term, case=False)
        )
        filtered = filtered[mask]

    return filtered


def create_new_user(
    first_name,
    last_name,
    email,
    phone,
    username,
    tenant,
    department,
    job_title,
    user_role,
    modules_access,
    data_access,
    require_mfa,
    force_password_change,
    session_timeout,
    login_hours,
):
    """Create new user (placeholder - in real implementation, persist to DB)"""
    return {
        "first_name": first_name,
        "last_name": last_name,
        "email": email,
        "username": username,
        "tenant": tenant,
        "role": user_role,
        "status": "Pending",
    }


def generate_roles_data():
    """Generate roles and permissions data"""
    return [
        {
            "role": "System Admin",
            "permissions": ["All"],
            "modules": ["All"],
            "data_access": "All Data",
            "user_count": 3,
        },
        {
            "role": "Tenant Admin",
            "permissions": ["user_management", "system_config"],
            "modules": ["All"],
            "data_access": "Tenant Data",
            "user_count": 8,
        },
        {
            "role": "Manager",
            "permissions": ["read_all", "write_department"],
            "modules": ["Sales", "Inventory", "Logistics"],
            "data_access": "Department Data",
            "user_count": 15,
        },
        {
            "role": "Analyst",
            "permissions": ["read_all"],
            "modules": ["All"],
            "data_access": "Department Data",
            "user_count": 22,
        },
        {
            "role": "Viewer",
            "permissions": ["read_limited"],
            "modules": ["Sales", "Inventory"],
            "data_access": "Personal Data",
            "user_count": 42,
        },
    ]


def display_roles_matrix(roles_data):
    """Display roles permissions matrix"""
    for role in roles_data:
        with st.expander(f"🎯 {role['role']} - {role['user_count']} users"):
            col1, col2, col3 = st.columns(3)

            with col1:
                st.write("**Permissions**:")
                for perm in role["permissions"]:
                    st.write(f"• {perm}")

            with col2:
                st.write("**Modules**:")
                if role["modules"] == ["All"]:
                    st.write("• All Modules")
                else:
                    for module in role["modules"]:
                        st.write(f"• {module}")

            with col3:
                st.write(f"**Data Access**: {role['data_access']}")

                if st.button(f"Edit {role['role']}", key=role["role"]):
                    st.info(f"Editing role: {role['role']}")


def generate_api_keys_data():
    """Generate API keys data"""
    return [
        {
            "name": "Sales Integration",
            "key_id": "sk_78901234567890",
            "status": "Active",
            "permissions": "read_sales, write_sales",
            "created_date": "2024-01-15",
            "last_used": "2024-02-20 14:30",
            "usage_count": 1245,
            "rate_limit": 1000,
        },
        {
            "name": "Inventory Sync",
            "key_id": "sk_12345678901234",
            "status": "Active",
            "permissions": "read_inventory, write_inventory",
            "created_date": "2024-02-01",
            "last_used": "2024-02-20 09:15",
            "usage_count": 892,
            "rate_limit": 500,
        },
    ]


def generate_audit_log():
    """Generate audit log data"""
    return [
        {
            "timestamp": "2024-02-20 14:25:32",
            "user": "admin@elora.com",
            "action": "User created",
            "details": "Created user: john.doe@company.com",
            "ip": "192.168.1.100",
        },
        {
            "timestamp": "2024-02-20 13:45:18",
            "user": "sarah.johnson@naivas.com",
            "action": "Password changed",
            "details": "User changed their password",
            "ip": "196.201.32.45",
        },
        {
            "timestamp": "2024-02-20 12:30:55",
            "user": "michael.brown@quickmart.com",
            "action": "Failed login",
            "details": "3 failed login attempts",
            "ip": "41.89.64.23",
        },
        {
            "timestamp": "2024-02-20 11:15:42",
            "user": "admin@elora.com",
            "action": "Role modified",
            "details": "Updated permissions for Manager role",
            "ip": "192.168.1.100",
        },
    ]


def display_audit_log(audit_log):
    """Display audit log"""
    log_df = pd.DataFrame(audit_log)
    st.dataframe(log_df, use_container_width=True)


def generate_engagement_data(user_data, activity_data):
    """Generate user engagement data"""
    engagement = []

    for _, user in user_data.iterrows():
        score = int(np.random.randint(20, 95))
        engagement.append(
            {
                "user_name": user["full_name"],
                "engagement_score": score,
                "login_frequency": user["login_count"],
                "last_activity": user["last_login"],
                "risk_level": "High"
                if score < 40
                else "Medium"
                if score < 70
                else "Low",
            }
        )

    return pd.DataFrame(engagement)


def display_engagement_scores(engagement_data):
    """Display engagement scores"""
    st.dataframe(engagement_data, use_container_width=True)


def detect_anomalies(activity_data):
    """Detect anomalies in user activity (synthetic examples)"""
    return [
        {
            "type": "Unusual Login Pattern",
            "user": "john.smith@company.com",
            "description": "Login from unusual location (USA)",
            "risk_level": "High",
            "timestamp": "2024-02-20 03:45:22",
        },
        {
            "type": "Multiple Failed Logins",
            "user": "sarah.johnson@naivas.com",
            "description": "5 failed login attempts within 10 minutes",
            "risk_level": "Medium",
            "timestamp": "2024-02-20 14:12:33",
        },
    ]


def display_anomalies(anomalies):
    """Display detected anomalies"""
    for anomaly in anomalies:
        st.error(f"⚠️ {anomaly['type']} - {anomaly['risk_level']} Risk")
        st.caption(
            f"User: {anomaly['user']} | {anomaly['description']} | {anomaly['timestamp']}"
        )


if __name__ == "__main__":
    render()
