# logistics_pro/app.py 
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
import warnings
import logging
from pathlib import Path
import importlib
import sys
import os

warnings.filterwarnings('ignore')

# Configure enterprise logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/application/system.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class LogisticsProEnterprise:
    """Main application class for Logistics Pro enterprise platform"""

    def __init__(self):
        self.setup_page_config()
        self.initialize_session_state()

    def setup_page_config(self):
        """Configure Streamlit page settings"""
        st.set_page_config(
            page_title="Logistics Pro - FMCG Intelligence Platform",
            page_icon="🚚",
            layout="wide",
            initial_sidebar_state="expanded",
            menu_items={
                'Get Help': 'https://logisticspro.com/docs',
                'Report a bug': 'https://logisticspro.com/support',
                'About': '# Logistics Pro v2.0 - Multi-Tenant FMCG Intelligence'
            }
        )
        self.apply_custom_styles()

    def apply_custom_styles(self):
        """Apply enterprise-grade custom styles"""
        st.markdown("""
        <style>
        /* Enhanced CSS to hide Streamlit's default elements - SAFE VERSION */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        .stDeployButton {visibility: hidden;}
        
        /* Hide Streamlit's default page navigation */
        [data-testid="stSidebarNav"] {
            display: none !important;
        }
        .css-1lcbmhc.e1fqkh3o0 {
            display: none !important;
        }
        
        /* Main theme colors */
        :root {
            --primary: #1f77b4;
            --secondary: #ff7f0e;
            --success: #2ca02c;
            --warning: #ffbb78;
            --error: #d62728;
            --background: #f8fafc;
            --surface: #ffffff;
            --text: #1e293b;
        }

        /* Button enhancements */
        .stButton>button {
            border-radius: 6px;
            border: 1px solid #e2e8f0;
            transition: all 0.2s ease;
            width: 100%;
            text-align: left;
            padding: 8px 12px;
            margin: 1px 0;
            background-color: white;
            font-size: 0.9rem;
        }

        .stButton>button:hover {
            background-color: #f1f5f9;
            border-color: #cbd5e1;
        }

        /* Active button styling */
        .active-module {
            background-color: #e0f2fe !important;
            border-left: 4px solid #1f77b4 !important;
            color: #1e40af !important;
            font-weight: 600;
        }

        /* Expander header styling */
        .streamlit-expanderHeader {
            background-color: #f8fafc !important;
            border-radius: 8px !important;
            padding: 10px 15px !important;
            margin: 5px 0 !important;
            border-left: 4px solid #64748b !important;
            font-weight: 600 !important;
        }

        .streamlit-expanderHeader:hover {
            background-color: #f1f5f9 !important;
        }

        .streamlit-expanderHeader[aria-expanded="true"] {
            border-left-color: #1f77b4 !important;
            background-color: #e0f2fe !important;
        }
        </style>
        """, unsafe_allow_html=True)

    def initialize_session_state(self):
        """Initialize session state with enterprise defaults"""
        if 'initialized' not in st.session_state:
            st.session_state.update({
                'initialized': True,
                'authenticated': True,
                'username': 'admin',
                'user_role': 'admin',
                'tenant_id': 'elora_holding',
                'current_tenant': 'ELORA Holding',
                'last_activity': datetime.now(),
                'current_page': '01_Dashboard',
                'current_module': '📊 Executive Cockpit',
                'current_submodule': '🏠 Main Dashboard',
                'dashboard_launched': False,
                'user_name': 'Enterprise User'  # Added for enhanced welcome page
            })
            self.initialize_tenant_data()

    def initialize_tenant_data(self):
        """Initialize data for the current tenant"""
        try:
            if 'data_gen' not in st.session_state:
                try:
                    from logistics_core.connectors.data_generator_enhanced import EnhancedDataGenerator
                    st.session_state.data_gen = EnhancedDataGenerator()
                except ImportError:
                    from logistics_core.connectors.data_generator import LogisticsProDataGenerator
                    st.session_state.data_gen = LogisticsProDataGenerator()
                
                st.session_state.analytics = AnalyticsEngine(st.session_state.data_gen)
            logger.info(f"Data initialized for tenant: {st.session_state.tenant_id}")
        except Exception as e:
            logger.error(f"Data initialization error: {e}")
            # Create fallback data
            st.session_state.data_gen = type('FallbackGenerator', (), {})()
            st.session_state.analytics = type('FallbackAnalytics', (), {
                'sales_data': pd.DataFrame(),
                'inventory_data': pd.DataFrame(),
                'logistics_data': pd.DataFrame()
            })()

    def get_available_tenants(self):
        """Get available tenants for multi-tenancy"""
        return {
            "ELORA Holding": "elora_holding",
            "Nakumatt Holdings": "nakumatt_holdings", 
            "QuickMart Kenya": "quickmart_kenya",
            "Tuskys Supermarkets": "tuskys_supermarkets"
        }

    def get_page_categories(self):
        """Define the page categories and their modules - UPDATED WITH WELCOME PAGE"""
        return {
            "🎯 Welcome & Onboarding": [
                {"name": "🏠 Welcome Center", "module": "00_Welcome", "path": "pages.00_Welcome"}
            ],
            "📊 Executive Cockpit": [
                {"name": "🏠 Main Dashboard", "module": "01_Dashboard", "path": "pages.01_Dashboard"}
            ],
            "📈 Sales Intelligence": [
                {"name": "🎯 Revenue Command Center", "module": "02_Revenue_Command_Center", "path": "pages.sales.02_Revenue_Command_Center"},
                {"name": "💰 Revenue Analytics", "module": "03_Revenue_Analytics", "path": "pages.sales.03_Revenue_Analytics"},
                {"name": "👥 Customer Segmentation", "module": "04_Customer_Segmentation", "path": "pages.sales.04_Customer_Segmentation"},
                {"name": "🌍 Regional Performance", "module": "05_Regional_Performance", "path": "pages.sales.05_Regional_Performance"},
                {"name": "📦 Product Performance", "module": "06_Product_Performance", "path": "pages.sales.06_Product_Performance"}
            ],
            "📦 Inventory Intelligence": [
                {"name": "🏥 Stock Health Dashboard", "module": "07_Inventory_Health", "path": "pages.inventory.07_Inventory_Health"},
                {"name": "🔍 ABC-XYZ Analysis", "module": "08_Inventory_ABC", "path": "pages.inventory.08_Inventory_ABC"},
                {"name": "⏰ Expiry Management", "module": "09_Inventory_Expiry", "path": "pages.inventory.09_Inventory_Expiry"},
                {"name": "🔄 Smart Replenishment", "module": "10_Inventory_Replenishment", "path": "pages.inventory.10_Inventory_Replenishment"}
            ],
            "🚛 Logistics Intelligence": [
                {"name": "🎯 OTIF Performance", "module": "11_Logistics_OTIF", "path": "pages.logistics.11_Logistics_OTIF"},
                {"name": "🔄 Route Optimization", "module": "12_Logistics_Routes", "path": "pages.logistics.12_Logistics_Routes"},
                {"name": "🚚 Fleet Utilization", "module": "13_Logistics_Fleet", "path": "pages.logistics.13_Logistics_Fleet"},
                {"name": "💰 Cost Analysis", "module": "14_Logistics_Costs", "path": "pages.logistics.14_Logistics_Costs"}
            ],
            "🤝 Procurement Intelligence": [
                {"name": "🏆 Supplier Scorecards", "module": "15_Procurement_Suppliers", "path": "pages.procurement.15_Procurement_Suppliers"},
                {"name": "💰 Cost Optimization", "module": "16_Procurement_Costs", "path": "pages.procurement.16_Procurement_Costs"},
                {"name": "📋 Smart Procurement", "module": "17_Procurement_Recommendations", "path": "pages.procurement.17_Procurement_Recommendations"}
            ],
            "🌐 Digital Intelligence": [
                {"name": "📊 Digital Overview", "module": "22_Digital_Overview", "path": "pages.digital_intelligence.22_Digital_Overview"},
                {"name": "🛒 Ecommerce Analytics", "module": "23_Ecommerce_Analytics", "path": "pages.digital_intelligence.23_Ecommerce_Analytics"},
                {"name": "🌐 Web Analytics", "module": "24_Web_Analytics", "path": "pages.digital_intelligence.24_Web_Analytics"},
                {"name": "📱 Social Intelligence", "module": "25_Social_Media_Intel", "path": "pages.digital_intelligence.25_Social_Media_Intel"},
                {"name": "⚡ Digital Operations", "module": "26_Digital_Operations", "path": "pages.digital_intelligence.26_Digital_Operations"}
            ],
            "⚙️ System Administration": [
                {"name": "🔧 Tenant Management", "module": "18_Admin_Tenants", "path": "pages.admin.18_Admin_Tenants"},
                {"name": "👥 User Management", "module": "19_Admin_Users", "path": "pages.admin.19_Admin_Users"},
                {"name": "📊 System Analytics", "module": "20_Admin_Analytics", "path": "pages.admin.20_Admin_Analytics"}
            ]
        }

    def get_total_modules_count(self):
        """Calculate total number of available modules"""
        page_categories = self.get_page_categories()
        total_count = 0
        for category, modules in page_categories.items():
            total_count += len(modules)
        return total_count

    def render_sidebar(self):
        """Render properly categorized sidebar"""
        st.sidebar.markdown(f"""
        <div style="text-align: center; padding: 20px 0;">
            <h2>🚚 Logistics Pro</h2>
            <p style="color: #64748b; font-size: 0.9rem;">
                {st.session_state.get('current_tenant', 'System')}
            </p>
        </div>
        """, unsafe_allow_html=True)

        # Tenant selector
        tenants = self.get_available_tenants()
        current_tenant = st.session_state.get('current_tenant', 'ELORA Holding')
        
        selected_tenant = st.sidebar.selectbox(
            "Switch Organization",
            list(tenants.keys()),
            index=list(tenants.keys()).index(current_tenant) if current_tenant in tenants else 0
        )
        
        if selected_tenant != current_tenant:
            st.session_state.current_tenant = selected_tenant
            st.session_state.tenant_id = tenants[selected_tenant]
            self.initialize_tenant_data()
            st.rerun()

        # User info card
        st.sidebar.markdown(f"""
        <div style="background: #f8fafc; padding: 15px; border-radius: 10px; margin-bottom: 20px; border-left: 4px solid #1f77b4;">
            <strong>{st.session_state.username}</strong><br>
            <small style="color: #64748b;">{st.session_state.user_role.replace('_', ' ').title()}</small>
        </div>
        """, unsafe_allow_html=True)

        # Get page categories
        page_categories = self.get_page_categories()

        # 📊 EXECUTIVE COCKPIT (Single item, always visible)
        dashboard_module = page_categories["📊 Executive Cockpit"][0]
        if st.sidebar.button(
            dashboard_module["name"], 
            key=f"nav_{dashboard_module['module']}",
            use_container_width=True
        ):
            st.session_state.current_page = dashboard_module["module"]
            st.session_state.current_module = "📊 Executive Cockpit"
            st.session_state.current_submodule = dashboard_module["name"]
            st.rerun()

        st.sidebar.markdown("---")

        # 📈 SALES INTELLIGENCE
        with st.sidebar.expander("📈 Sales Intelligence", expanded=False):
            for module in page_categories["📈 Sales Intelligence"]:
                if st.button(
                    module["name"], 
                    key=f"nav_{module['module']}",
                    use_container_width=True
                ):
                    st.session_state.current_page = module["module"]
                    st.session_state.current_module = "📈 Sales Intelligence"
                    st.session_state.current_submodule = module["name"]
                    st.rerun()

        # 📦 INVENTORY INTELLIGENCE
        with st.sidebar.expander("📦 Inventory Intelligence", expanded=False):
            for module in page_categories["📦 Inventory Intelligence"]:
                if st.button(
                    module["name"], 
                    key=f"nav_{module['module']}",
                    use_container_width=True
                ):
                    st.session_state.current_page = module["module"]
                    st.session_state.current_module = "📦 Inventory Intelligence"
                    st.session_state.current_submodule = module["name"]
                    st.rerun()

        # 🚛 LOGISTICS INTELLIGENCE
        with st.sidebar.expander("🚛 Logistics Intelligence", expanded=False):
            for module in page_categories["🚛 Logistics Intelligence"]:
                if st.button(
                    module["name"], 
                    key=f"nav_{module['module']}",
                    use_container_width=True
                ):
                    st.session_state.current_page = module["module"]
                    st.session_state.current_module = "🚛 Logistics Intelligence"
                    st.session_state.current_submodule = module["name"]
                    st.rerun()

        # 🤝 PROCUREMENT INTELLIGENCE
        with st.sidebar.expander("🤝 Procurement Intelligence", expanded=False):
            for module in page_categories["🤝 Procurement Intelligence"]:
                if st.button(
                    module["name"], 
                    key=f"nav_{module['module']}",
                    use_container_width=True
                ):
                    st.session_state.current_page = module["module"]
                    st.session_state.current_module = "🤝 Procurement Intelligence"
                    st.session_state.current_submodule = module["name"]
                    st.rerun()

        # 🌐 DIGITAL INTELLIGENCE
        with st.sidebar.expander("🌐 Digital Intelligence", expanded=False):
            for module in page_categories["🌐 Digital Intelligence"]:
                if st.button(
                    module["name"], 
                    key=f"nav_{module['module']}",
                    use_container_width=True
                ):
                    st.session_state.current_page = module["module"]
                    st.session_state.current_module = "🌐 Digital Intelligence"
                    st.session_state.current_submodule = module["name"]
                    st.rerun()

        # ⚙️ SYSTEM ADMINISTRATION (admin only)
        if st.session_state.user_role == 'admin':
            st.sidebar.markdown("---")
            with st.sidebar.expander("⚙️ System Administration", expanded=False):
                for module in page_categories["⚙️ System Administration"]:
                    if st.button(
                        module["name"], 
                        key=f"nav_{module['module']}",
                        use_container_width=True
                    ):
                        st.session_state.current_page = module["module"]
                        st.session_state.current_module = "⚙️ System Administration"
                        st.session_state.current_submodule = module["name"]
                        st.rerun()

        # Quick Actions Section
        st.sidebar.markdown("---")
        st.sidebar.markdown("### 🚀 Quick Actions")

        col1, col2 = st.sidebar.columns(2)
        with col1:
            if st.sidebar.button("🔄 Refresh", use_container_width=True):
                st.rerun()
        with col2:
            if st.sidebar.button("📈 Reports", use_container_width=True):
                st.session_state.current_page = "01_Dashboard"
                st.session_state.current_module = "📊 Executive Cockpit"
                st.session_state.current_submodule = "🏠 Main Dashboard"
                st.rerun()

        # LOGOUT BUTTON (moved from top)
        if st.sidebar.button(
            "🚪 Logout", 
            key="nav_logout",
            use_container_width=True,
            type="secondary"
        ):
            # Reset session state to log out
            st.session_state.dashboard_launched = False
            st.session_state.current_page = "00_Welcome"
            st.session_state.current_module = "🎯 Welcome & Onboarding"
            st.session_state.current_submodule = "🏠 Welcome Center"
            st.rerun()

        # System status
        st.sidebar.markdown("""
        <div style="text-align: center; margin-top: 20px; padding: 10px; background: #10b98110; border-radius: 8px;">
            <small style="color: #10b981;">🟢 System Online</small><br>
            <small style="color: #64748b;">v2.0 Enterprise</small>
        </div>
        """, unsafe_allow_html=True)

    def render_welcome_page(self):
        """Render the enhanced welcome page by loading 00_Welcome.py"""
        welcome_module = self.import_page_module("00_Welcome")
        if welcome_module:
            welcome_module.render()
        else:
            # Silent fallback to basic version
            self.render_basic_welcome()

    def render_basic_welcome(self):
        """Fallback basic welcome page (your original content)"""
        st.title("🚚 Logistics Pro")
        st.subheader("Data, Analytics & AI Operating System for FMCG Distribution")

        st.markdown("""
        Welcome to **Logistics Pro** – a modular data, analytics & AI platform designed for 
        FMCG distributors and logistics operators.

        This prototype runs entirely on **inline synthetic data** (no external database yet) 
        so we can:

        - Demonstrate the full analytics and AI journey end-to-end.
        - Stress-test UX, KPIs, and workflows.
        - Easily extend the engine for real-world clients.

        Use the navigation in the sidebar to explore:

        1. **Sales & Customer Intelligence** – Revenue Command Center, Customer 360, SKU–Customer–Region views.
        2. **Inventory & Warehouse Intelligence** – Stock health, ABC–XYZ, shelf-life risk.
        3. **Logistics & Fleet Intelligence** – OTIF, cost per drop, route productivity.
        4. **Procurement & Supplier Excellence** – Supplier scorecards, landed cost, rebates.
        5. **Digital Intelligence** – Ecommerce, Web Analytics, Social Media insights.
        6. **Executive Cockpit & Sustainability** – Top-level KPIs, expansion, sustainability.

        In production, the same structure will connect to ERP, WMS, TMS and CRM systems.
        """)

        # Center the button
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🚀 Launch Dashboard", type="primary", use_container_width=True):
                with st.spinner('🔄 Loading analytics engine and generating demo data... This may take a few moments.'):
                    st.session_state.dashboard_launched = True
                    if 'data_gen' not in st.session_state:
                        self.initialize_tenant_data()
                    st.rerun()

        # Footer
        st.markdown("---")
        st.markdown(
            "<div style='text-align: center; color: #64748b;'>"
            "© 2024 Logistics Pro | Enterprise Analytics Platform"
            "</div>",
            unsafe_allow_html=True
        )

    def get_module_path(self, page_name: str):
        """Get the correct module path based on page name"""
        page_categories = self.get_page_categories()
        
        # Search through all categories to find the module
        for category, modules in page_categories.items():
            for module in modules:
                if module["module"] == page_name:
                    return module["path"]
        
        # Fallback logic for direct file access
        if page_name == "00_Welcome":
            return "pages.00_Welcome"
        elif page_name.startswith("01_"):
            return f"pages.{page_name}"
        elif page_name.startswith(("02_", "03_", "04_", "05_", "06_")):
            return f"pages.sales.{page_name}"
        elif page_name.startswith(("07_", "08_", "09_", "10_")):
            return f"pages.inventory.{page_name}"
        elif page_name.startswith(("11_", "12_", "13_", "14_")):
            return f"pages.logistics.{page_name}"
        elif page_name.startswith(("15_", "16_", "17_")):
            return f"pages.procurement.{page_name}"
        elif page_name.startswith(("22_", "23_", "24_", "25_", "26_")):
            return f"pages.digital_intelligence.{page_name}"
        elif page_name.startswith(("18_", "19_", "20_")):
            return f"pages.admin.{page_name}"
        else:
            return f"pages.{page_name}"

    def import_page_module(self, page_name: str):
        """Dynamically import page module"""
        try:
            # Get the correct module path
            module_path = self.get_module_path(page_name).replace('.py', '')
        
            # Import the module
            module = importlib.import_module(module_path)
        
            if hasattr(module, 'render'):
                return module
            else:
                logger.error(f"Page {page_name} missing 'render' function")
                return None
            
        except ImportError as e:
            logger.error(f"Cannot load {page_name}: {str(e)}")
            return None
        except Exception as e:
            logger.error(f"Error loading {page_name}: {str(e)}")
            return None

    def render_enterprise_ticker(self):
        """Render the enterprise ticker with strategic messaging (unused by default)"""
        st.markdown(f"""
        <div style="background: white; padding: 12px; border-radius: 12px; color: maroon; margin-bottom: 20px; border: 2px solid #F0F2F6;">
            <!-- Enterprise Intelligence Ticker - MARQUEE -->
            <div style="background: white; border-radius: 6px; padding: 10px; overflow: hidden;">
                <marquee behavior="scroll" direction="left" scrollamount="4" style="font-size: 0.95rem; font-weight: 500; color: maroon;">
                    🌟 <strong>Strategic Intelligence:</strong> Driving FMCG Excellence • 
                    📈 <strong>Revenue Performance:</strong> $2.8M MTD | +18.3% YoY Growth • 
                    🎯 <strong>Customer Excellence:</strong> 94.2% OTIF Delivery | Elite Service Levels • 
                    📦 <strong>Inventory Optimization:</strong> 6.8x Turnover | 98.5% Stock Availability • 
                    🚛 <strong>Logistics Mastery:</strong> 78.5% Fleet Utilization | Route Efficiency +22% • 
                    🤝 <strong>Supplier Intelligence:</strong> 156 Active Partners | 95.8% Compliance • 
                    🌐 <strong>Digital Transformation:</strong> E-commerce +35% | Social Engagement +42% • 
                    ⚡ <strong>AI-Powered Insights:</strong> 8 Predictive Models | Real-time Decision Support • 
                    🏆 <strong>Enterprise Performance:</strong> 99.8% Uptime | SOC 2 Certified | Multi-Tenant Secure
                </marquee>
            </div>
        </div>
        """, unsafe_allow_html=True)

    def render_current_page(self):
        """Render the current selected page with enhanced error handling"""
        current_page = st.session_state.get('current_page', '00_Welcome')  # Default to Welcome
        current_module = st.session_state.get('current_module', '🎯 Welcome & Onboarding')
        current_submodule = st.session_state.get('current_submodule', '🏠 Welcome Center')

        try:
            page_module = self.import_page_module(current_page)
            if page_module:
                page_module.render()
            else:
                # Fallback to welcome page with user feedback
                st.error(f"Unable to load {current_submodule}. Redirecting to welcome page...")
                if current_page != "00_Welcome":
                    st.session_state.current_page = "00_Welcome"
                    st.session_state.current_module = "🎯 Welcome & Onboarding"
                    st.session_state.current_submodule = "🏠 Welcome Center"
                    st.rerun()

        except Exception as e:
            logger.error(f"Critical error rendering page {current_page}: {str(e)}")
            st.error("""
            🚨 **Critical Application Error**
            
            The requested module could not be loaded. This could be due to:
            - Missing module files
            - Syntax errors in the module
            - Import dependencies not being met
            
            Please contact system administration.
            """)
            
            # Provide debugging information
            with st.expander("Technical Details"):
                st.code(f"""
                Error: {str(e)}
                Page: {current_page}
                Module: {current_module}
                Submodule: {current_submodule}
                """)

    def run(self):
        """Main application runner"""
        try:
            # Force scroll to top on every page load
            st.markdown("""
            <script>
            // Always scroll to top when page loads
            window.addEventListener('load', function() {
                window.scrollTo(0, 0);
            });
            // Also scroll to top when navigating
            setTimeout(function() {
                window.scrollTo(0, 0);
            }, 100);
            </script>
            """, unsafe_allow_html=True)

            # Always start with welcome page if not launched
            if not st.session_state.get('dashboard_launched', False):
                self.render_welcome_page()
                return

            with st.sidebar:
                self.render_sidebar()

            self.render_current_page()

        except Exception as e:
            logger.error(f"Application error: {str(e)}")
            st.error("A system error occurred. Our team has been notified.")

class AnalyticsEngine:
    """Core analytics and AI functionality"""
    
    def __init__(self, data_generator):
        self.dg = data_generator
        # Initialize with fallbacks for all data types
        try:
            self.sales_data = data_generator.sales_data
        except:
            self.sales_data = pd.DataFrame()
            
        try:
            self.inventory_data = data_generator.inventory_data
        except:
            self.inventory_data = pd.DataFrame()
            
        try:
            self.logistics_data = data_generator.logistics_data
        except:
            self.logistics_data = pd.DataFrame()
        
        # Initialize analytics engines with fallbacks
        try:
            from logistics_core.analytics.forecasting import DemandForecaster
            self.forecaster = DemandForecaster()
        except:
            self.forecaster = None
        
        try:
            from logistics_core.analytics.optimization import RouteOptimizer, InventoryOptimizer
            self.route_optimizer = RouteOptimizer()
            self.inventory_optimizer = InventoryOptimizer()
        except:
            self.route_optimizer = None
            self.inventory_optimizer = None

    def calculate_kpis(self):
        """Calculate key performance indicators"""
        try:
            # Sales KPIs
            if not self.sales_data.empty:
                total_revenue = (self.sales_data['quantity'] * self.sales_data['unit_price']).sum()
                recent_sales = self.sales_data[self.sales_data['date'] >= (datetime.now().date() - timedelta(days=30))]
                recent_revenue = (recent_sales['quantity'] * recent_sales['unit_price']).sum()
            else:
                total_revenue = 0
                recent_revenue = 0
            
            kpis = {
                'monthly_revenue': recent_revenue,
                'gross_margin_percent': np.random.uniform(18, 25),
                'inventory_turnover': np.random.uniform(6, 12),
                'stockout_rate': np.random.uniform(2, 8),
                'otif_rate': self.logistics_data['otif'].mean() * 100 if not self.logistics_data.empty else 85,
                'logistics_cost_per_case': np.random.uniform(45, 85),
                'fleet_utilization': np.random.uniform(65, 85),
                'customer_satisfaction': np.random.uniform(85, 95)
            }
            return kpis
        except Exception as e:
            logger.error(f"KPI calculation error: {e}")
            return {}

# Application entry point
if __name__ == "__main__":
    app = LogisticsProEnterprise()
    app.run()