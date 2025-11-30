# logistics_pro/pages/00_Welcome.py
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime


def render():
    """Welcome Page - ENTERPRISE PREMIUM VERSION"""

    # Apply premium styling
    st.markdown(
        """
    <style>
    .welcome-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 3rem 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        color: white;
        text-align: center;
    }
    .feature-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        box-shadow: 0 2px 10px rgba(0,0,0,0.08);
        margin-bottom: 1rem;
        height: 100%;
    }
    /* Target the launch button on this page */
    div.stButton > button:first-child {
        background: linear-gradient(135deg, #dc2626 0%, #b91c1b 100%);
        color: white;
        border: none;
        border-radius: 25px;
        font-size: 1.1rem;
        font-weight: bold;
        transition: all 0.3s ease;
        width: 100%;
    }
    div.stButton > button:first-child:hover {
        background: linear-gradient(135deg, #b91c1b 0%, #991b1b 100%);
        color: white;
        border: none;
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
    }
    .tenant-selector {
        background: rgba(255,255,255,0.1);
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        border: 1px solid rgba(255,255,255,0.2);
    }
    .lp-max-width {
        max-width: 1200px;
        margin: 0 auto;
    }
    </style>
    """,
        unsafe_allow_html=True,
    )

    # Page variables
    total_modules = 27  # Total number of modules in the platform
    platform_version = "v2.0 Enterprise"

    # Constrain content width for large screens
    st.markdown("<div class='lp-max-width'>", unsafe_allow_html=True)

    # Hero Section
    st.markdown(
        """
    <div class="welcome-header">
        <h1 style="color: white; margin: 0; font-size: 3.5rem;">🚚 LOGISTICS PRO</h1>
        <p style="color: white; font-size: 1.5rem; margin: 0.5rem 0 0 0;">
            Enterprise Intelligence Platform for FMCG Distribution
        </p>
        <p style="color: rgba(255,255,255,0.9); font-size: 1.1rem; margin: 1rem 0 0 0;">
            AI-Powered Analytics • Multi-Tenant Architecture • Real-Time Insights
        </p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Soft green marquee strip
    st.markdown(
        """
    <div style="
        background: #ecfdf3;
        border-radius: 10px;
        padding: 8px 14px;
        border: 1px solid #bbf7d0;
        margin-bottom: 20px;">
        <marquee behavior="scroll" direction="left" scrollamount="4"
                 style="font-size: 0.9rem; color: #166534; font-weight: 500;">
            🚚 Logistics Pro Demo • AI-powered enterprise intelligence for FMCG distribution • 
            6 intelligence domains • 27 analytics modules • Synthetic demo tenant: ELORA Holding
        </marquee>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Personalized Welcome & Tenant Selection
    col1, col2 = st.columns([2, 1])

    with col1:
        user_name = st.session_state.get("user_name", "Enterprise User")
        last_login = st.session_state.get("last_login", "First visit")

        st.markdown(
            f"""
        <div style="text-align: center; color: #64748b; margin-bottom: 0.5rem;">
            <h3 style="color: #334155; margin-bottom: 0.5rem;">
                👋 Welcome back, <strong>{user_name}</strong>
            </h3>
            <small>Last login: {last_login}</small>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            """
        <div class="tenant-selector">
            <div style="color: #e5e7eb; text-align: center; margin-bottom: 0.5rem;">
                <strong>🏢 Active Organization</strong>
            </div>
        """,
            unsafe_allow_html=True,
        )

        tenant_options = [
            "ELORA Holding",
            "Premium Foods Ltd",
            "Global Distributors Inc",
            "Urban Retail Group",
        ]
        current_tenant = st.session_state.get("current_tenant", "ELORA Holding")

        selected_tenant = st.selectbox(
            "Select Organization",
            options=tenant_options,
            index=tenant_options.index(current_tenant)
            if current_tenant in tenant_options
            else 0,
            label_visibility="collapsed",
        )

        if selected_tenant != current_tenant:
            st.session_state.current_tenant = selected_tenant
            st.rerun()

        st.markdown("</div>", unsafe_allow_html=True)

    # Top CTA button (single launch control)
    top_cta_col = st.columns([3, 2, 3])[1]
    with top_cta_col:
        if st.button(
            "LAUNCH ENTERPRISE DASHBOARD",
            type="primary",
            use_container_width=True,
            key="launch_enterprise_top",
        ):
            with st.spinner("🔄 Initializing enterprise analytics engine..."):
                st.session_state.dashboard_launched = True
                st.session_state.last_login = datetime.now().strftime(
                    "%Y-%m-%d %H:%M:%S"
                )
                st.session_state.user_name = user_name or "Enterprise User"
                st.session_state.current_tenant = selected_tenant
                st.rerun()

    # Platform Overview
    st.subheader("🎯 Platform Overview")

    st.markdown(
        f"""
    **Logistics Pro** is an enterprise intelligence platform purpose-built for 
    **FMCG distributors and logistics operators**. It delivers end-to-end visibility across your 
    supply chain with AI-powered analytics, executive dashboards, and modular strategic frameworks.
    
    *Currently viewing data for: **{selected_tenant}***  
    
    *This demonstration runs on **synthetic enterprise data** showcasing the full analytical capabilities 
    that would integrate with your ERP, WMS, TMS, and CRM systems in production.*
    """
    )

    # === NEW: Compact Enterprise Modules section (using tabs instead of grids) ===
    st.subheader("🏗️ Enterprise Modules")

    (
        tab_sales,
        tab_logistics,
        tab_inventory,
        tab_procurement,
        tab_digital,
        tab_system,
    ) = st.tabs(
        [
            "📈 Sales Intelligence",
            "🚛 Logistics Intelligence",
            "📦 Inventory Intelligence",
            "🤝 Procurement Intelligence",
            "🌐 Digital Intelligence",
            "⚙️ System Administration",
        ]
    )

    with tab_sales:
        st.markdown(
            """
        <div class="feature-card">
            <h4>📈 Sales Intelligence</h4>
            <p style="color:#64748b;font-size:0.9rem;margin-bottom:0.5rem;">
                Executive view of revenue growth, customer value, and product mix across regions and channels.
            </p>
            <ul style="padding-left: 1.2rem;">
                <li>Revenue analytics & forecasting by region, channel, and SKU</li>
                <li>Customer 360° view with RFM segmentation for CLV</li>
                <li>Product portfolio optimization (BCG-style matrices)</li>
                <li>Regional market share & whitespace identification</li>
            </ul>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with tab_logistics:
        st.markdown(
            """
        <div class="feature-card">
            <h4>🚛 Logistics Intelligence</h4>
            <p style="color:#64748b;font-size:0.9rem;margin-bottom:0.5rem;">
                Fleet and route performance, OTIF, and cost-to-serve for every drop and route.
            </p>
            <ul style="padding-left: 1.2rem;">
                <li>OTIF performance tracking down to route and customer</li>
                <li>Route optimization engine with distance & drop economics</li>
                <li>Fleet utilization, turnaround time & idle capacity</li>
                <li>Cost-per-drop and lane profitability analytics</li>
            </ul>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with tab_inventory:
        st.markdown(
            """
        <div class="feature-card">
            <h4>📦 Inventory Intelligence</h4>
            <p style="color:#64748b;font-size:0.9rem;margin-bottom:0.5rem;">
                Working-capital aware stock health, service levels, and expiry risk across DCs.
            </p>
            <ul style="padding-left: 1.2rem;">
                <li>Stock health dashboards (coverage, DOH, service levels)</li>
                <li>ABC–XYZ classification for inventory investment decisions</li>
                <li>Expiry, waste & write-off monitoring with alerts</li>
                <li>Smart replenishment signals into ERP / WMS</li>
            </ul>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with tab_procurement:
        st.markdown(
            """
        <div class="feature-card">
            <h4>🤝 Procurement Intelligence</h4>
            <p style="color:#64748b;font-size:0.9rem;margin-bottom:0.5rem;">
                Supplier performance, spend analytics, and savings pipeline tracking.
            </p>
            <ul style="padding-left: 1.2rem;">
                <li>Supplier scorecards (quality, OTIF, pricing, risk)</li>
                <li>Spend analytics by category, supplier, and business unit</li>
                <li>Cost optimization & savings pipeline tracking</li>
                <li>AI-led recommendations for sourcing and consolidation</li>
            </ul>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with tab_digital:
        st.markdown(
            """
        <div class="feature-card">
            <h4>🌐 Digital Intelligence</h4>
            <p style="color:#64748b;font-size:0.9rem;margin-bottom:0.5rem;">
                Ecommerce, web and social analytics stitched to offline sales performance.
            </p>
            <ul style="padding-left: 1.2rem;">
                <li>Ecommerce conversion & basket analysis</li>
                <li>Web journey funnels mapped to trade channels</li>
                <li>Social media intelligence for campaign performance</li>
                <li>Digital operations and service performance dashboards</li>
            </ul>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with tab_system:
        st.markdown(
            """
        <div class="feature-card">
            <h4>⚙️ System Administration</h4>
            <p style="color:#64748b;font-size:0.9rem;margin-bottom:0.5rem;">
                Governance, user access, performance monitoring, and auditability.
            </p>
            <ul style="padding-left: 1.2rem;">
                <li>Multi-tenant management & onboarding workflows</li>
                <li>User security, role-based access & permissions</li>
                <li>System performance and usage analytics</li>
                <li>Audit & compliance reporting</li>
            </ul>
        </div>
        """,
            unsafe_allow_html=True,
        )

    # Strategic Value Proposition
    st.subheader("💡 Strategic Value Proposition")

    value_col1, value_col2 = st.columns(2)

    with value_col1:
        st.markdown(
            """
        **🎯 Decision Intelligence**
        - **BCG Matrix Analysis** for product portfolio strategy  
        - **RFM Segmentation** for customer lifetime value optimization  
        - **ABC-XYZ Classification** for inventory investment  
        - **Predictive Forecasting** for demand planning
        
        **📊 Enterprise-Grade Analytics**
        - Multi-tenant data isolation  
        - Real-time KPI monitoring  
        - Automated reporting & alerts  
        - Mobile-responsive design
        """
        )

    with value_col2:
        st.markdown(
            """
        **🚀 Operational Excellence**
        - **Reduce stockouts** by 15–25%  
        - **Improve OTIF** performance by 20–30%  
        - **Optimize fleet utilization** by 18–22%  
        - **Lower procurement costs** by 12–18%  
        
        **🔗 Seamless Integration**
        - ERP system connectors  
        - Warehouse management sync  
        - Transportation management  
        - Ecommerce platform APIs
        """
        )

    # Demo Information
    with st.expander("🔍 Demo Information & Getting Started", expanded=True):
        st.info(
            f"""
        **🎭 About This Demonstration:**
        
        This prototype showcases the complete Logistics Pro platform using **synthetic enterprise data** 
        that mimics real-world FMCG distribution patterns. All analytics, AI insights, and strategic 
        frameworks are fully functional and demonstrate exactly how the platform would operate with 
        your actual business data.
        
        **Current Organization:** {selected_tenant}
        
        **🚀 Getting Started:**
        1. Use the **'LAUNCH ENTERPRISE DASHBOARD'** button above to enter the platform  
        2. Start with the **Executive Cockpit** for an overview  
        3. Explore modules using the **sidebar navigation**  
        4. Each module features **4-tab analysis** (Overview, Analytics, Deep Dive, Actions)  
        5. Look for **collapsible strategic insights** in each section
        
        **💡 Pro Tip:** Pay special attention to the **strategic frameworks** (BCG Matrix, RFM Analysis, etc.) 
        that transform raw data into actionable business intelligence.
        """
        )

    # Small note under the “invisible” stats area
    st.markdown("<br>", unsafe_allow_html=True)

    center_col = st.columns([1, 2, 1])[1]
    with center_col:
        st.markdown(
            f"""
        <div style='text-align: center; margin-top: 0.5rem; color: #64748b;'>
            <small>Access {total_modules} analytics modules across 6 intelligence domains for {selected_tenant}</small><br>
            <small style="color:#9ca3af;">Demo mode • Synthetic data only</small>
        </div>
        """,
            unsafe_allow_html=True,
        )

    # Enterprise Footer
    st.markdown("---")

    footer_col1, footer_col2, footer_col3 = st.columns(3)

    with footer_col1:
        st.markdown(
            """
        **🏢 Enterprise Platform**
        - Multi-tenant architecture
        - Role-based access control
        - Audit trail & compliance
        - SLA guarantees
        """
        )

    with footer_col2:
        st.markdown(
            """
        **🔒 Security & Privacy**
        - SOC 2 compliant
        - Data encryption at rest
        - Regular security audits
        - GDPR ready
        """
        )

    with footer_col3:
        st.markdown(
            """
        **🌍 Global Support**
        - 24/7 enterprise support
        - Dedicated success managers
        - Custom implementation
        - Training & certification
        """
        )

    st.markdown(
        f"""
    <div style='text-align: center; color: #64748b; margin-top: 2rem;'>
        <strong>© 2024 Logistics Pro | Enterprise Analytics Platform {platform_version}</strong><br>
        <small>Transforming FMCG Distribution Through AI-Powered Intelligence • Active Tenant: {selected_tenant}</small>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Close max-width wrapper
    st.markdown("</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    render()