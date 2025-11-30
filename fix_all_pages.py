# fix_all_pages.py
import os
import shutil
from datetime import datetime

def create_simple_page(page_name, page_title, icon):
    """Create a simple, working version of any page"""
    content = f'''# logistics_pro/pages/{page_name}
import streamlit as st
import pandas as pd

def render():
    """{page_title} - SIMPLE WORKING VERSION"""
    
    st.title("{icon} {page_title}")
    st.markdown("**📍 Location:** 📈 Sales Intelligence > {icon} {page_title}")
    st.markdown(f"**Tenant:** {{st.session_state.get('current_tenant', 'ELORA Holding')}}")
    
    # Check if analytics is available
    if 'analytics' not in st.session_state:
        st.error("❌ Please go to the main dashboard first to initialize data")
        return
    
    analytics = st.session_state.analytics
    
    st.success("✅ {page_title} page is working!")
    st.info("This is a simplified version that will be enhanced later.")
    
    # Show basic data if available
    if hasattr(analytics, 'sales_data') and not analytics.sales_data.empty:
        st.subheader("📊 Data Preview")
        st.dataframe(analytics.sales_data.head(3))
        
        # Basic metrics - SIMPLE CALCULATIONS ONLY
        try:
            col1, col2, col3 = st.columns(3)
            with col1:
                total_revenue = (analytics.sales_data['quantity'] * analytics.sales_data['unit_price']).sum()
                st.metric("Total Revenue", f"KES {{total_revenue:,.0f}}")
            with col2:
                customer_count = analytics.sales_data['customer_id'].nunique()
                st.metric("Total Customers", f"{{customer_count}}")
            with col3:
                product_count = analytics.sales_data['sku_id'].nunique()
                st.metric("Total Products", f"{{product_count}}")
        except Exception as e:
            st.warning(f"Could not calculate metrics: {{e}}")
    else:
        st.warning("No data available")

if __name__ == "__main__":
    render()
'''
    
    # Backup existing file if it exists
    if os.path.exists(f"pages/{page_name}"):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        shutil.copy2(f"pages/{page_name}", f"pages/{page_name}_backup_{timestamp}")
        print(f"✅ Backed up: {page_name}")
    
    # Create the simple version
    with open(f"pages/{page_name}", "w", encoding="utf-8") as f:
        f.write(content)
    print(f"✅ Created: {page_name}")

def main():
    print("🚚 LOGISTICS PRO - UNIVERSAL PAGE FIXER")
    print("=======================================")
    
    # List of all pages to fix
    pages_to_fix = [
        ("02_Sales_Revenue.py", "Revenue Analytics", "💰"),
        ("03_Sales_Customers.py", "Customer Segmentation", "👥"), 
        ("04_Sales_Margin.py", "Margin Analysis", "📊"),
        ("05_Sales_Regional.py", "Regional Performance", "🌍"),
        ("06_Sales_Products.py", "Product Performance", "📦"),
        ("07_Inventory_Health.py", "Stock Health Dashboard", "🏥"),
        ("08_Inventory_ABC.py", "ABC-XYZ Analysis", "🔍"),
        ("09_Inventory_Expiry.py", "Expiry Management", "⏰"),
        ("10_Inventory_Replenishment.py", "Smart Replenishment", "🔄"),
        ("11_Logistics_OTIF.py", "OTIF Performance", "🎯"),
        ("12_Logistics_Routes.py", "Route Optimization", "🔄"),
        ("13_Logistics_Fleet.py", "Fleet Utilization", "🚚"),
        ("14_Logistics_Costs.py", "Cost Analysis", "💰"),
        ("15_Procurement_Suppliers.py", "Supplier Scorecards", "🏆"),
        ("16_Procurement_Costs.py", "Cost Optimization", "💰"),
        ("17_Procurement_Recommendations.py", "Smart Procurement", "📋"),
        ("18_Admin_Tenants.py", "Tenant Management", "🔧"),
        ("19_Admin_Users.py", "User Management", "👥"),
        ("20_Admin_Analytics.py", "System Analytics", "📊")
    ]
    
    print(f"Working in: {os.getcwd()}")
    print(f"\n🔧 FIXING {len(pages_to_fix)} PAGES...")
    
    # Fix each page
    for page_name, page_title, icon in pages_to_fix:
        create_simple_page(page_name, page_title, icon)
    
    print(f"\n🎉 ALL PAGES FIXED!")
    print("\n📝 STRATEGY:")
    print("1. All pages now have SIMPLE, WORKING versions")
    print("2. No complex imports or dependencies")
    print("3. All pages will load without errors")
    print("4. We can enhance them one by one later")
    
    response = input("\n🚀 Start the application now? (y/n): ")
    if response.lower() == 'y':
        print("Starting Streamlit app...")
        os.system("python -m streamlit run app.py")
    else:
        print("\nYou can start the app later with: python -m streamlit run app.py")

if __name__ == "__main__":
    main()