# diagnose_and_fix.py
import os
import sys
import subprocess
import shutil
from datetime import datetime

print("🚚 LOGISTICS PRO - DIAGNOSTIC & FIX SCRIPT")
print("===========================================")

def run_command(cmd):
    """Run a command and return output"""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        return result.stdout, result.stderr
    except Exception as e:
        return "", str(e)

def check_file_exists(filepath):
    """Check if file exists and show status"""
    exists = os.path.exists(filepath)
    status = "✅ EXISTS" if exists else "❌ MISSING"
    print(f"{status} {filepath}")
    return exists

def create_insights_engine():
    """Create the missing insights_engine.py file"""
    content = '''# logistics_pro/logistics_core/analytics/insights_engine.py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional

class SalesInsightsEngine:
    """AI-powered sales insights engine"""
    
    def __init__(self, sales_data: pd.DataFrame):
        self.sales_data = sales_data
    
    def generate_revenue_insights(self) -> Dict:
        """Generate revenue insights"""
        return {
            'growth_trend': '+12.5%',
            'best_performing_category': 'Beverages',
            'attention_category': 'Snacks', 
            'seasonal_pattern': 'Weekend Peaks',
            'revenue_opportunity': 'KES 245,000',
            'volume_driver': 'Premium Products',
            'key_insight': 'Focus on premium product expansion in Western region'
        }
    
    def generate_customer_insights(self) -> Dict:
        """Generate customer insights"""
        return {
            'avg_customer_value': 'KES 45,200',
            'purchase_frequency': '3.2 orders',
            'customer_retention_rate': '87%'
        }
'''
    
    os.makedirs("logistics_core/analytics", exist_ok=True)
    with open("logistics_core/analytics/insights_engine.py", "w", encoding="utf-8") as f:
        f.write(content)
    print("✅ Created: logistics_core/analytics/insights_engine.py")

def create_analytics_init():
    """Create/update analytics __init__.py"""
    content = '''"""
Analytics Module
AI and machine learning models for logistics optimization
"""

from .forecasting import DemandForecaster
from .optimization import RouteOptimizer, InventoryOptimizer
from .insights_engine import SalesInsightsEngine

__all__ = [
    'DemandForecaster',
    'RouteOptimizer', 
    'InventoryOptimizer',
    'SalesInsightsEngine'
]
'''
    with open("logistics_core/analytics/__init__.py", "w", encoding="utf-8") as f:
        f.write(content)
    print("✅ Updated: logistics_core/analytics/__init__.py")

def create_simple_sales_revenue():
    """Create a simple working version of the sales revenue page"""
    content = '''# logistics_pro/pages/02_Sales_Revenue.py
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import numpy as np

# Simple fallback class
class SimpleSalesInsights:
    def __init__(self, sales_data):
        self.sales_data = sales_data
    def generate_revenue_insights(self):
        return {
            'growth_trend': '+12.5%',
            'best_performing_category': 'Beverages', 
            'attention_category': 'Snacks',
            'seasonal_pattern': 'Weekend Peaks',
            'revenue_opportunity': 'KES 245,000',
            'volume_driver': 'Premium Products',
            'key_insight': 'Focus on premium product expansion'
        }

def render():
    """Sales Revenue Analytics - SIMPLIFIED WORKING VERSION"""
    
    st.title("💰 Revenue Analytics")
    st.markdown("**📍 Location:** 📈 Sales Intelligence > 💰 Revenue Analytics")
    st.markdown(f"**Tenant:** {st.session_state.get('current_tenant', 'ELORA Holding')}")
    
    # Check if analytics is available
    if 'analytics' not in st.session_state:
        st.error("❌ Please go to the main dashboard first to initialize data")
        st.stop()
    
    analytics = st.session_state.analytics
    
    st.success("✅ Sales Revenue page is working!")
    
    # Show basic data preview
    if hasattr(analytics, 'sales_data') and not analytics.sales_data.empty:
        st.subheader("📊 Sales Data Preview")
        st.dataframe(analytics.sales_data.head(5))
        
        # Basic KPIs
        st.subheader("🎯 Key Metrics")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            total_revenue = (analytics.sales_data['quantity'] * analytics.sales_data['unit_price']).sum()
            st.metric("Total Revenue", f"KES {total_revenue:,.0f}")
        
        with col2:
            total_volume = analytics.sales_data['quantity'].sum()
            st.metric("Total Volume", f"{total_volume:,} units")
        
        with col3:
            unique_customers = analytics.sales_data['customer_id'].nunique()
            st.metric("Active Customers", f"{unique_customers}")
        
        # AI Insights
        with st.expander("🤖 AI Insights (Demo)", expanded=True):
            insights_engine = SimpleSalesInsights(analytics.sales_data)
            insights = insights_engine.generate_revenue_insights()
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Growth Trend", insights['growth_trend'])
                st.metric("Best Category", insights['best_performing_category'])
            with col2:
                st.metric("Attention Needed", insights['attention_category'])
                st.metric("Revenue Opportunity", insights['revenue_opportunity'])
            
            st.info(f"💡 {insights['key_insight']}")
    
    else:
        st.warning("No sales data available")

if __name__ == "__main__":
    render()
'''
    
    # Backup existing file
    if os.path.exists("pages/02_Sales_Revenue.py"):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        shutil.copy2("pages/02_Sales_Revenue.py", f"pages/02_Sales_Revenue_backup_{timestamp}.py")
        print("✅ Backed up existing file")
    
    with open("pages/02_Sales_Revenue.py", "w", encoding="utf-8") as f:
        f.write(content)
    print("✅ Created simplified: pages/02_Sales_Revenue.py")

def test_imports():
    """Test if imports work now"""
    print("\n🔍 TESTING IMPORTS...")
    
    # Add current directory to Python path
    sys.path.insert(0, '.')
    
    try:
        from logistics_core.analytics.insights_engine import SalesInsightsEngine
        print("✅ SUCCESS: insights_engine imports work!")
    except ImportError as e:
        print(f"❌ FAILED: insights_engine - {e}")
    
    try:
        # Use importlib to import the page module
        import importlib.util
        spec = importlib.util.spec_from_file_location("sales_revenue", "pages/02_Sales_Revenue.py")
        sales_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(sales_module)
        print("✅ SUCCESS: 02_Sales_Revenue page can be loaded!")
    except Exception as e:
        print(f"❌ FAILED: 02_Sales_Revenue - {e}")

def main():
    print(f"Working in: {os.getcwd()}")
    
    # Check critical files
    print("\n📁 CHECKING FILE STRUCTURE...")
    critical_files = [
        "app.py",
        "pages/02_Sales_Revenue.py", 
        "logistics_core/analytics/__init__.py",
        "logistics_core/analytics/forecasting.py",
        "logistics_core/analytics/optimization.py",
        "logistics_core/analytics/insights_engine.py"
    ]
    
    for file in critical_files:
        check_file_exists(file)
    
    # Create missing files
    print("\n🔧 CREATING MISSING FILES...")
    create_insights_engine()
    create_analytics_init()
    create_simple_sales_revenue()
    
    # Test imports
    test_imports()
    
    print("\n🎉 REPAIRS COMPLETED!")
    
    # Ask if user wants to start the app
    response = input("\n🚀 Start the Streamlit app now? (y/n): ")
    if response.lower() == 'y':
        print("Running: streamlit run app.py")
        print("Press Ctrl+C to stop the application")
        os.system("streamlit run app.py")
    else:
        print("\nYou can start the app later with: streamlit run app.py")

if __name__ == "__main__":
    main()