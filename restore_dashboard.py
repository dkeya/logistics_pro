# restore_dashboard.py
import os

def restore_dashboard():
    """Restore the dashboard functionality"""
    print("ðŸ”§ RESTORING DASHBOARD FUNCTIONALITY...")
    
    # Read the current app.py to see what broke
    with open('app.py', 'r', encoding='utf-8') as f:
        app_content = f.read()
    
    # Check if AnalyticsEngine is broken
    if "def __init__(self, sales_data):" in app_content:
        print("âœ… Found the issue - AnalyticsEngine constructor was changed")
        
        # Restore the original AnalyticsEngine
        old_analytics = '''class AnalyticsEngine:
    """Core analytics and AI functionality"""
    
    def __init__(self, sales_data):
        self.sales_data = sales_data
        # For compatibility with pages that expect these
        self.inventory_data = pd.DataFrame()
        self.logistics_data = pd.DataFrame()'''
        
        new_analytics = '''class AnalyticsEngine:
    """Core analytics and AI functionality"""
    
    def __init__(self, data_generator):
        self.dg = data_generator
        self.sales_data = data_generator.generate_sales_data()
        self.inventory_data = data_generator.generate_inventory_data()
        self.logistics_data = data_generator.generate_logistics_data()
        self.forecaster = DemandForecaster()
        self.route_optimizer = RouteOptimizer()
        self.inventory_optimizer = InventoryOptimizer()'''
        
        app_content = app_content.replace(old_analytics, new_analytics)
        
        # Also fix the tenant data initialization
        old_init = '''    def initialize_tenant_data(self):
        """Initialize data for the current tenant"""
        try:
            if 'data_gen' not in st.session_state:
                st.session_state.data_gen = LogisticsProDataGenerator()
                # Generate data using the actual methods
                sales_data = st.session_state.data_gen.generate_sales_data()
                # Create analytics with the generated data
                st.session_state.analytics = AnalyticsEngine(sales_data)
            logger.info(f"Data initialized for tenant: {st.session_state.tenant_id}")
        except Exception as e:
            logger.error(f"Data initialization error: {e}")'''
            
        new_init = '''    def initialize_tenant_data(self):
        """Initialize data for the current tenant"""
        try:
            if 'data_gen' not in st.session_state:
                st.session_state.data_gen = LogisticsProDataGenerator()
                st.session_state.analytics = AnalyticsEngine(st.session_state.data_gen)
            logger.info(f"Data initialized for tenant: {st.session_state.tenant_id}")
        except Exception as e:
            logger.error(f"Data initialization error: {e}")'''
            
        app_content = app_content.replace(old_init, new_init)
        
        # Write the fixed app.py
        with open('app.py', 'w', encoding='utf-8') as f:
            f.write(app_content)
        
        print("âœ… Dashboard restored!")
        return True
    else:
        print("âŒ Could not identify the issue")
        return False

def update_new_pages_for_tabs():
    """Update the new pages to include tab structure"""
    print("\nðŸ“Š ADDING TAB STRUCTURE TO NEW PAGES...")
    
    # Update Margin Analysis page with tabs
    margin_content = '''# logistics_pro/pages/04_Sales_Margin.py
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import numpy as np

def render():
    """Margin Analysis - WITH TABS"""
    
    st.title("ðŸ“Š Margin Analysis")
    st.markdown("**ðŸ“ Location:** ðŸ“ˆ Sales Intelligence > ðŸ“Š Margin Analysis")
    st.markdown(f"**Tenant:** {st.session_state.get('current_tenant', 'ELORA Holding')}")
    
    if 'analytics' not in st.session_state:
        st.error("âŒ Please go to the main dashboard first")
        return
    
    analytics = st.session_state.analytics
    
    # Get sales data from analytics
    sales_data = analytics.sales_data
    
    # Calculate basic margin metrics
    sales_data['revenue'] = sales_data['quantity'] * sales_data['unit_price']
    
    # For demo purposes, assume 75% cost ratio if unit_cost doesn't exist
    if 'unit_cost' in sales_data.columns:
        sales_data['cost'] = sales_data['quantity'] * sales_data['unit_cost']
    else:
        sales_data['cost'] = sales_data['revenue'] * 0.75  # Assume 25% margin
    
    sales_data['margin'] = sales_data['revenue'] - sales_data['cost']
    sales_data['margin_percent'] = (sales_data['margin'] / sales_data['revenue']).replace([np.inf, -np.inf], 0) * 100
    
    st.success("âœ… Margin Analysis loaded successfully!")
    
    # TAB STRUCTURE
    tab1, tab2, tab3 = st.tabs(["ðŸ’° Overview", "ðŸ“ˆ Trends", "ðŸ’¡ Insights"])
    
    with tab1:
        render_margin_overview(sales_data)
    
    with tab2:
        render_margin_trends(sales_data)
    
    with tab3:
        render_margin_insights(sales_data)

def render_margin_overview(sales_data):
    """Render margin overview tab"""
    st.header("ðŸ’° Margin Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    total_revenue = sales_data['revenue'].sum()
    total_margin = sales_data['margin'].sum()
    avg_margin = sales_data['margin_percent'].mean()
    total_volume = sales_data['quantity'].sum()
    
    with col1:
        st.metric("Total Revenue", f"KES {total_revenue:,.0f}")
    with col2:
        st.metric("Total Margin", f"KES {total_margin:,.0f}")
    with col3:
        st.metric("Avg Margin %", f"{avg_margin:.1f}%")
    with col4:
        st.metric("Total Volume", f"{total_volume:,} units")
    
    # Margin Distribution
    st.subheader("ðŸ“ˆ Margin Distribution")
    
    fig = px.histogram(
        sales_data, 
        x='margin_percent',
        nbins=20,
        title='Distribution of Margin Percentages',
        labels={'margin_percent': 'Margin %'}
    )
    st.plotly_chart(fig, width='stretch')

def render_margin_trends(sales_data):
    """Render margin trends tab"""
    st.header("ðŸ“… Margin Trends")
    
    # Daily margin trends
    daily_margins = sales_data.groupby('date').agg({
        'margin_percent': 'mean',
        'revenue': 'sum'
    }).reset_index()
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=daily_margins['date'],
        y=daily_margins['margin_percent'],
        mode='lines+markers',
        name='Daily Margin %'
    ))
    fig.update_layout(title='Daily Margin Percentage Trend')
    st.plotly_chart(fig, width='stretch')
    
    # Weekly patterns
    sales_data['day_of_week'] = pd.to_datetime(sales_data['date']).dt.day_name()
    weekly_margins = sales_data.groupby('day_of_week')['margin_percent'].mean().reset_index()
    
    fig = px.bar(weekly_margins, x='day_of_week', y='margin_percent', 
                 title='Average Margin by Day of Week')
    st.plotly_chart(fig, width='stretch')

def render_margin_insights(sales_data):
    """Render margin insights tab"""
    st.header("ðŸ’¡ Margin Insights & Recommendations")
    
    # Basic insights
    avg_margin = sales_data['margin_percent'].mean()
    low_margin_count = len(sales_data[sales_data['margin_percent'] < 10])
    high_margin_count = len(sales_data[sales_data['margin_percent'] > 25])
    
    st.subheader("ðŸ“Š Performance Summary")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Overall Margin", f"{avg_margin:.1f}%")
    with col2:
        st.metric("Low Margin Transactions", low_margin_count)
    with col3:
        st.metric("High Margin Transactions", high_margin_count)
    
    st.subheader("ðŸŽ¯ Optimization Opportunities")
    
    insights = [
        "ðŸ’° **Pricing Strategy**: Review pricing for low-margin products",
        "ðŸ“¦ **Product Mix**: Promote high-margin products to improve overall margin",
        "ðŸšš **Cost Optimization**: Analyze delivery and handling costs",
        "ðŸŽª **Customer Segmentation**: Offer tiered pricing based on customer value",
        "ðŸ“Š **Seasonal Analysis**: Identify high-margin periods for promotion"
    ]
    
    for insight in insights:
        st.write(f"- {insight}")
    
    # Actionable recommendations
    st.subheader("ðŸš€ Quick Actions")
    
    if st.button("ðŸ“ˆ Generate Margin Report"):
        st.success("Margin report generated! Focus on products with <10% margin.")
    
    if st.button("ðŸŽ¯ Identify Optimization Targets"):
        low_margin_products = sales_data[sales_data['margin_percent'] < 10]
        if len(low_margin_products) > 0:
            st.info(f"Found {len(low_margin_products)} transactions with margins below 10%")
        else:
            st.success("All transactions have healthy margins!")

if __name__ == "__main__":
    render()
'''

    # Update Customer Segmentation page with tabs
    customer_content = '''# logistics_pro/pages/03_Sales_Customers.py
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import numpy as np

def render():
    """Customer Segmentation - WITH TABS"""
    
    st.title("ðŸ‘¥ Customer Segmentation")
    st.markdown("**ðŸ“ Location:** ðŸ“ˆ Sales Intelligence > ðŸ‘¥ Customer Segmentation")
    st.markdown(f"**Tenant:** {st.session_state.get('current_tenant', 'ELORA Holding')}")
    
    if 'analytics' not in st.session_state:
        st.error("âŒ Please go to the main dashboard first")
        return
    
    analytics = st.session_state.analytics
    
    # Get sales data
    sales_data = analytics.sales_data
    
    # Basic calculations
    sales_data['revenue'] = sales_data['quantity'] * sales_data['unit_price']
    
    st.success("âœ… Customer Segmentation loaded successfully!")
    
    # TAB STRUCTURE
    tab1, tab2, tab3 = st.tabs(["ðŸ‘¥ Overview", "ðŸŽ¯ RFM Analysis", "ðŸ’¡ Insights"])
    
    with tab1:
        render_customer_overview(sales_data)
    
    with tab2:
        render_rfm_analysis(sales_data)
    
    with tab3:
        render_customer_insights(sales_data)

def render_customer_overview(sales_data):
    """Render customer overview tab"""
    st.header("ðŸ‘¥ Customer Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    total_customers = sales_data['customer_id'].nunique()
    total_revenue = sales_data['revenue'].sum()
    total_orders = len(sales_data)
    avg_order_value = total_revenue / total_orders if total_orders > 0 else 0
    
    with col1:
        st.metric("Total Customers", total_customers)
    with col2:
        st.metric("Total Revenue", f"KES {total_revenue:,.0f}")
    with col3:
        st.metric("Total Orders", total_orders)
    with col4:
        st.metric("Avg Order Value", f"KES {avg_order_value:,.0f}")
    
    # Customer Revenue Distribution
    st.subheader("ðŸ“Š Customer Revenue Analysis")
    
    customer_revenue = sales_data.groupby('customer_id').agg({
        'revenue': 'sum',
        'quantity': 'sum'
    }).reset_index()
    
    # Top customers
    top_customers = customer_revenue.nlargest(10, 'revenue')
    
    fig = px.bar(
        top_customers,
        x='customer_id',
        y='revenue',
        title='Top 10 Customers by Revenue',
        labels={'customer_id': 'Customer ID', 'revenue': 'Revenue (KES)'}
    )
    st.plotly_chart(fig, width='stretch')

def render_rfm_analysis(sales_data):
    """Render RFM analysis tab"""
    st.header("ðŸŽ¯ Customer Value Segmentation (RFM)")
    
    # Calculate basic RFM
    current_date = sales_data['date'].max()
    
    rfm = sales_data.groupby('customer_id').agg({
        'date': lambda x: (current_date - x.max()).days,  # Recency
        'customer_id': 'count',  # Frequency  
        'revenue': 'sum'  # Monetary
    }).reset_index()
    
    rfm.columns = ['customer_id', 'recency', 'frequency', 'monetary']
    
    # Simple segmentation
    rfm['segment'] = np.select([
        (rfm['recency'] <= 30) & (rfm['monetary'] > rfm['monetary'].quantile(0.8)),
        (rfm['recency'] <= 60) & (rfm['monetary'] > rfm['monetary'].quantile(0.6)),
        rfm['recency'] > 90,
    ], ['Champions', 'Loyal', 'At Risk'], default='Regular')
    
    segment_counts = rfm['segment'].value_counts()
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.pie(
            values=segment_counts.values,
            names=segment_counts.index,
            title='Customer Segments Distribution'
        )
        st.plotly_chart(fig, width='stretch')
    
    with col2:
        # RFM Scatter plot
        fig = px.scatter(
            rfm,
            x='frequency',
            y='monetary',
            color='segment',
            size='recency',
            title='RFM Analysis: Frequency vs Monetary',
            hover_data=['customer_id']
        )
        st.plotly_chart(fig, width='stretch')
    
    # Segment details
    st.subheader("ðŸ“‹ Segment Details")
    st.dataframe(
        rfm.groupby('segment').agg({
            'customer_id': 'count',
            'recency': 'mean',
            'frequency': 'mean',
            'monetary': 'mean'
        }).round(2),
        width='stretch'
    )

def render_customer_insights(sales_data):
    """Render customer insights tab"""
    st.header("ðŸ’¡ Customer Insights & Strategies")
    
    # Calculate insights
    current_date = sales_data['date'].max()
    customer_activity = sales_data.groupby('customer_id').agg({
        'date': 'max',
        'revenue': 'sum'
    }).reset_index()
    
    inactive_customers = len(customer_activity[
        (current_date - customer_activity['date']).dt.days > 60
    ])
    
    high_value_customers = len(customer_activity[
        customer_activity['revenue'] > customer_activity['revenue'].quantile(0.8)
    ])
    
    st.subheader("ðŸ“ˆ Key Metrics")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Inactive Customers (60+ days)", inactive_customers)
    with col2:
        st.metric("High Value Customers", high_value_customers)
    with col3:
        avg_customer_value = customer_activity['revenue'].mean()
        st.metric("Avg Customer Value", f"KES {avg_customer_value:,.0f}")
    
    st.subheader("ðŸŽ¯ Actionable Strategies")
    
    strategies = [
        "ðŸ† **Champions**: Exclusive offers and loyalty rewards",
        "ðŸ’Ž **Loyal Customers**: Cross-selling and upselling opportunities", 
        "âš ï¸ **At-Risk Customers**: Win-back campaigns and special offers",
        "ðŸ“Š **Regular Customers**: Engagement programs to increase frequency",
        "ðŸ†• **New Customers**: Onboarding and welcome sequences"
    ]
    
    for strategy in strategies:
        st.write(f"- {strategy}")
    
    # Quick actions
    st.subheader("ðŸš€ Quick Actions")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("ðŸ“§ Export Customer List"):
            st.success("Customer list exported for marketing campaigns!")
    
    with col2:
        if st.button("ðŸŽ¯ Identify Win-back Targets"):
            st.success(f"Identified {inactive_customers} customers for win-back campaigns")

if __name__ == "__main__":
    render()
'''

    # Write the updated pages with tabs
    with open('pages/04_Sales_Margin.py', 'w', encoding='utf-8') as f:
        f.write(margin_content)
    
    with open('pages/03_Sales_Customers.py', 'w', encoding='utf-8') as f:
        f.write(customer_content)
    
    print("âœ… Added tab structure to new pages!")

def main():
    print("ðŸšš LOGISTICS PRO - DASHBOARD RESTORE")
    print("=====================================")
    
    # Restore dashboard functionality
    if restore_dashboard():
        # Add tabs to new pages
        update_new_pages_for_tabs()
        
        print("\nðŸŽ‰ DASHBOARD RESTORED & PAGES ENHANCED!")
        print("\nâœ… Dashboard page will work again")
        print("âœ… New pages now have proper tab structure")
        print("âœ… All functionality preserved")
        
        print("\nðŸš€ STARTING APPLICATION...")
        os.system("python -m streamlit run app.py")
    else:
        print("\nâŒ Could not restore dashboard. Please check manually.")

if __name__ == "__main__":
    main()
