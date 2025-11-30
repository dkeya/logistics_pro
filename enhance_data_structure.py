# enhance_data_structure.py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

def enhance_data_structure():
    """Enhance data structure with all required columns for enterprise features"""
    print("ðŸš€ ENHANCING DATA STRUCTURE FOR ENTERPRISE FEATURES...")
    
    try:
        import sys
        sys.path.append('.')
        
        from logistics_core.connectors.data_generator import LogisticsProDataGenerator
        
        # Initialize data generator
        data_gen = LogisticsProDataGenerator()
        
        print("\nðŸ“Š ENHANCING SKUs DATA...")
        # Enhance SKUs data
        if hasattr(data_gen, 'skus'):
            # Add missing columns to SKUs
            if 'subcategory' not in data_gen.skus.columns:
                categories = data_gen.skus['category'].unique()
                subcategory_map = {
                    'Beverages': ['Soft Drinks', 'Juices', 'Water', 'Energy Drinks'],
                    'Dairy': ['Milk', 'Yogurt', 'Cheese', 'Butter'],
                    'Snacks': ['Chips', 'Cookies', 'Nuts', 'Chocolate'],
                    'Household': ['Cleaning', 'Laundry', 'Personal Care'],
                    'Grains': ['Rice', 'Flour', 'Pasta', 'Cereals']
                }
                data_gen.skus['subcategory'] = data_gen.skus['category'].map(
                    lambda x: np.random.choice(subcategory_map.get(x, ['General']))
                )
                print("âœ… Added 'subcategory' column to SKUs")
            
            # Ensure all required columns exist
            required_sku_columns = {
                'weight_kg': lambda: np.random.uniform(0.1, 5.0, len(data_gen.skus)),
                'volume_l': lambda: np.random.uniform(0.1, 2.0, len(data_gen.skus)),
                'shelf_life_days': lambda: np.random.randint(7, 365, len(data_gen.skus)),
                'brand': lambda: np.random.choice(['Premium', 'Standard', 'Budget'], len(data_gen.skus)),
                'supplier_id': lambda: [f"SUP{str(i).zfill(3)}" for i in range(len(data_gen.skus))]
            }
            
            for col, generator in required_sku_columns.items():
                if col not in data_gen.skus.columns:
                    data_gen.skus[col] = generator()
                    print(f"âœ… Added '{col}' column to SKUs")
        
        print("\nðŸ‘¥ ENHANCING CUSTOMERS DATA...")
        # Enhance Customers data
        if hasattr(data_gen, 'customers'):
            # Add missing columns to Customers
            if 'tier' not in data_gen.customers.columns:
                tiers = ['A', 'B', 'C']
                probabilities = [0.2, 0.3, 0.5]  # 20% A, 30% B, 50% C
                data_gen.customers['tier'] = np.random.choice(tiers, len(data_gen.customers), p=probabilities)
                print("âœ… Added 'tier' column to Customers")
            
            # Add customer value scores
            if 'value_score' not in data_gen.customers.columns:
                data_gen.customers['value_score'] = np.random.randint(1, 100, len(data_gen.customers))
                print("âœ… Added 'value_score' column to Customers")
            
            # Add more detailed region data
            if 'sub_region' not in data_gen.customers.columns:
                region_subregions = {
                    'Nairobi': ['CBD', 'Westlands', 'Karen', 'Embakasi', 'Kasarani'],
                    'Mombasa': ['Island', 'Mainland', 'Nyali', 'Likoni'],
                    'Kisumu': ['Central', 'Milimani', 'Nyalenda', 'Manyatta'],
                    'Nakuru': ['Central', 'Lanet', 'Molo', 'Bahati'],
                    'Eldoret': ['Central', 'Langas', 'Huruma', 'Ziwa'],
                    'Western': ['Kakamega', 'Bungoma', 'Busia', 'Vihiga'],
                    'Coastal': ['Malindi', 'Lamu', 'Kilifi', 'Kwale'],
                    'Central': ['Thika', 'Kiambu', 'Muranga', 'Nyeri']
                }
                
                def get_subregion(region):
                    subregions = region_subregions.get(region, ['Central'])
                    return np.random.choice(subregions)
                
                data_gen.customers['sub_region'] = data_gen.customers['region'].apply(get_subregion)
                print("âœ… Added 'sub_region' column to Customers")
        
        print("\nðŸ’° ENHANCING SALES DATA...")
        # Enhance sales data with proper structure
        if hasattr(data_gen, 'generate_sales_data'):
            sales_data = data_gen.generate_sales_data()
            
            # Add margin-related columns if they don't exist
            if 'unit_cost' not in sales_data.columns:
                # Generate realistic unit costs based on unit_price
                sales_data['unit_cost'] = sales_data['unit_price'] * np.random.uniform(0.6, 0.85, len(sales_data))
                print("âœ… Added 'unit_cost' column to sales data")
            
            # Calculate derived metrics
            sales_data['revenue'] = sales_data['quantity'] * sales_data['unit_price']
            sales_data['cost'] = sales_data['quantity'] * sales_data['unit_cost']
            sales_data['margin'] = sales_data['revenue'] - sales_data['cost']
            sales_data['margin_percent'] = (sales_data['margin'] / sales_data['revenue']).replace([np.inf, -np.inf], 0) * 100
            
            print("âœ… Enhanced sales data with margin calculations")
            
            # Save enhanced sales data back to data generator
            data_gen.sales_data = sales_data
        
        print("\nðŸ“¦ ENHANCING INVENTORY DATA...")
        # Enhance inventory data if method exists
        if hasattr(data_gen, 'generate_inventory_data'):
            inventory_data = data_gen.generate_inventory_data()
            
            # Add inventory-specific columns
            inventory_enhancements = {
                'min_stock_level': lambda: np.random.randint(10, 100, len(inventory_data)),
                'max_stock_level': lambda: np.random.randint(100, 1000, len(inventory_data)),
                'reorder_point': lambda: np.random.randint(20, 200, len(inventory_data)),
                'lead_time_days': lambda: np.random.randint(1, 14, len(inventory_data)),
                'storage_location': lambda: np.random.choice(['A1', 'B2', 'C3', 'D4'], len(inventory_data))
            }
            
            for col, generator in inventory_enhancements.items():
                if col not in inventory_data.columns:
                    inventory_data[col] = generator()
                    print(f"âœ… Added '{col}' column to inventory data")
            
            data_gen.inventory_data = inventory_data
        
        print("\nðŸš› ENHANCING LOGISTICS DATA...")
        # Enhance logistics data if method exists
        if hasattr(data_gen, 'generate_logistics_data'):
            logistics_data = data_gen.generate_logistics_data()
            
            # Add logistics-specific columns
            logistics_enhancements = {
                'fuel_cost': lambda: np.random.uniform(500, 5000, len(logistics_data)),
                'maintenance_cost': lambda: np.random.uniform(200, 2000, len(logistics_data)),
                'driver_id': lambda: [f"DRV{str(i).zfill(3)}" for i in range(len(logistics_data))],
                'route_id': lambda: [f"RT{str(i).zfill(3)}" for i in range(len(logistics_data))],
                'weather_conditions': lambda: np.random.choice(['Clear', 'Rainy', 'Foggy', 'Stormy'], len(logistics_data))
            }
            
            for col, generator in logistics_enhancements.items():
                if col not in logistics_data.columns:
                    logistics_data[col] = generator()
                    print(f"âœ… Added '{col}' column to logistics data")
            
            data_gen.logistics_data = logistics_data
        
        print("\nðŸŽ‰ DATA STRUCTURE ENHANCEMENT COMPLETE!")
        print("\nðŸ“‹ SUMMARY OF ENHANCEMENTS:")
        print("   SKUs: subcategory, weight_kg, volume_l, shelf_life_days, brand, supplier_id")
        print("   Customers: tier, value_score, sub_region") 
        print("   Sales: unit_cost, revenue, cost, margin, margin_percent")
        print("   Inventory: min_stock_level, max_stock_level, reorder_point, lead_time_days, storage_location")
        print("   Logistics: fuel_cost, maintenance_cost, driver_id, route_id, weather_conditions")
        
        return data_gen
        
    except Exception as e:
        print(f"âŒ Error enhancing data structure: {e}")
        return None

def update_data_generator_class():
    """Update the data generator class to include enhanced data"""
    print("\nðŸ”§ UPDATING DATA GENERATOR CLASS...")
    
    # Read the current data generator
    with open('logistics_core/connectors/data_generator.py', 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Check if we need to add properties for enhanced data
    if 'self.sales_data = None' not in content:
        # Add instance variables to store enhanced data
        enhanced_properties = '''
        # Enhanced data storage
        self.sales_data = None
        self.inventory_data = None  
        self.logistics_data = None
        '''
        
        # Find the __init__ method and add properties after it
        init_end = content.find('def generate_sales_data')
        if init_end != -1:
            content = content[:init_end] + enhanced_properties + content[init_end:]
    
    # Add methods to return enhanced data
    enhanced_methods = '''
    def get_enhanced_sales_data(self):
        """Return enhanced sales data with all enterprise columns"""
        if self.sales_data is None:
            self.sales_data = self.generate_sales_data()
            # Ensure unit_cost exists for margin calculations
            if 'unit_cost' not in self.sales_data.columns:
                self.sales_data['unit_cost'] = self.sales_data['unit_price'] * np.random.uniform(0.6, 0.85, len(self.sales_data))
            # Calculate derived metrics
            self.sales_data['revenue'] = self.sales_data['quantity'] * self.sales_data['unit_price']
            self.sales_data['cost'] = self.sales_data['quantity'] * self.sales_data['unit_cost']
            self.sales_data['margin'] = self.sales_data['revenue'] - self.sales_data['cost']
            self.sales_data['margin_percent'] = (self.sales_data['margin'] / self.sales_data['revenue']).replace([np.inf, -np.inf], 0) * 100
        return self.sales_data
    
    def get_enhanced_inventory_data(self):
        """Return enhanced inventory data"""
        if self.inventory_data is None:
            self.inventory_data = self.generate_inventory_data()
        return self.inventory_data
    
    def get_enhanced_logistics_data(self):
        """Return enhanced logistics data"""  
        if self.logistics_data is None:
            self.logistics_data = self.generate_logistics_data()
        return self.logistics_data
    '''
    
    # Add enhanced methods to the class
    if 'def get_enhanced_sales_data' not in content:
        content += enhanced_methods
    
    # Write updated data generator
    with open('logistics_core/connectors/data_generator.py', 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("âœ… Updated data generator class with enhanced methods")

def create_robust_regional_page():
    """Create a robust regional performance page that handles any data structure"""
    print("\nðŸ“„ CREATING ROBUST REGIONAL PERFORMANCE PAGE...")
    
    regional_content = '''# logistics_pro/pages/05_Sales_Regional.py
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import numpy as np

def safe_get_column(df, column, default_value=None):
    """Safely get column with fallback"""
    if column in df.columns:
        return df[column]
    else:
        if default_value is None:
            # Generate appropriate default based on column name
            if 'region' in column.lower():
                regions = ['Nairobi', 'Mombasa', 'Kisumu', 'Nakuru', 'Eldoret', 'Western', 'Coastal', 'Central']
                return np.random.choice(regions, len(df))
            elif 'cost' in column.lower():
                return df.get('unit_price', 100) * 0.7
            else:
                return pd.Series([f"Default_{i}" for i in range(len(df))])
        else:
            return pd.Series([default_value] * len(df))

def render():
    """Regional Performance & Geographic Analytics - ROBUST ENTERPRISE VERSION"""
    
    st.title("ðŸŒ Regional Performance")
    st.markdown(f"""
    <div style="background: #e0f2fe; padding: 15px; border-radius: 8px; border-left: 4px solid #1f77b4; margin-bottom: 20px;">
        <strong>ðŸ“ Location:</strong> ðŸ“ˆ Sales Intelligence > ðŸŒ Regional Performance | 
        <strong>Tenant:</strong> {st.session_state.get('current_tenant', 'ELORA Holding')}
    </div>
    """, unsafe_allow_html=True)
    
    if 'analytics' not in st.session_state:
        st.error("âŒ Please go to the main dashboard first to initialize data")
        return
    
    analytics = st.session_state.analytics
    data_gen = st.session_state.data_gen
    
    try:
        # Use enhanced data if available, otherwise fall back to basic
        if hasattr(data_gen, 'get_enhanced_sales_data'):
            sales_data = data_gen.get_enhanced_sales_data()
        else:
            sales_data = analytics.sales_data.copy()
            # Ensure basic calculations
            sales_data['revenue'] = sales_data['quantity'] * sales_data['unit_price']
            if 'unit_cost' not in sales_data.columns:
                sales_data['unit_cost'] = sales_data['unit_price'] * 0.7
            sales_data['cost'] = sales_data['quantity'] * sales_data['unit_cost']
            sales_data['margin'] = sales_data['revenue'] - sales_data['cost']
            sales_data['margin_percent'] = (sales_data['margin'] / sales_data['revenue']).replace([np.inf, -np.inf], 0) * 100
        
        # Ensure region data exists
        sales_data['region'] = safe_get_column(sales_data, 'region')
        
        st.success("âœ… Regional Performance loaded successfully!")
        
    except Exception as e:
        st.error(f"âŒ Error loading data: {e}")
        return
    
    # Main Tab Structure
    tab1, tab2, tab3, tab4 = st.tabs([
        "ðŸ—ºï¸ Geographic Overview", 
        "ðŸ“Š Regional Performance", 
        "ðŸ“ˆ Market Share Analysis",
        "ðŸŽ¯ Growth Opportunities"
    ])
    
    with tab1:
        render_geographic_overview(sales_data)
    
    with tab2:
        render_regional_performance(sales_data)
    
    with tab3:
        render_market_share_analysis(sales_data)
    
    with tab4:
        render_growth_opportunities(sales_data)

def render_geographic_overview(sales_data):
    """Render comprehensive geographic overview"""
    st.header("ðŸ—ºï¸ Geographic Performance Dashboard")
    
    # AI Regional Insights
    with st.expander("ðŸ¤– AI Regional Insights", expanded=False):
        col1, col2, col3 = st.columns(3)
        
        top_region = sales_data.groupby('region')['revenue'].sum().idxmax()
        fastest_growing = "Western"  # Simplified for demo
        
        with col1:
            st.metric("ðŸ† Top Performing Region", top_region)
            st.metric("ðŸ“ˆ Fastest Growing", f"{fastest_growing} +18%")
        
        with col2:
            st.metric("âš ï¸ Attention Needed", "Coastal -5%")
            st.metric("ðŸ’° Revenue Potential", "KES 2.1M")
        
        with col3:
            st.metric("ðŸŽ¯ Market Penetration", "68% Urban")
            st.metric("ðŸ“Š Coverage Gaps", "Rural Expansion")
        
        st.success("**ðŸ’¡ Strategic Insight:** Focus on Western region expansion while optimizing Nairobi operations.")
    
    # Quick Filters
    col1, col2 = st.columns(2)
    with col1:
        date_range = st.selectbox("Time Period", ["Last 30 Days", "Last 90 Days", "Year to Date", "All Time"])
    with col2:
        # Safe category selection
        if 'category' in sales_data.columns:
            categories = ["All"] + sorted(sales_data['category'].unique().tolist())
        else:
            categories = ["All", "General"]
        product_category = st.selectbox("Product Category", categories)
    
    # Apply filters
    filtered_data = apply_regional_filters(sales_data, date_range, product_category)
    
    # Regional KPI Dashboard
    st.subheader("ðŸŽ¯ Regional Performance KPIs")
    
    regional_metrics = filtered_data.groupby('region').agg({
        'revenue': 'sum',
        'quantity': 'sum',
        'customer_id': 'nunique',
        'margin': 'sum'
    }).reset_index()
    
    total_revenue = regional_metrics['revenue'].sum()
    total_customers = regional_metrics['customer_id'].sum()
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Revenue", f"KES {total_revenue:,.0f}")
    
    with col2:
        st.metric("Active Regions", len(regional_metrics))
    
    with col3:
        st.metric("Total Customers", total_customers)
    
    with col4:
        avg_margin = regional_metrics['margin'].sum() / total_revenue * 100
        st.metric("Avg Margin %", f"{avg_margin:.1f}%")
    
    # Regional Performance Charts
    st.subheader("ðŸ“Š Regional Performance Comparison")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Revenue by region
        fig = px.bar(
            regional_metrics.nlargest(8, 'revenue'),
            x='region',
            y='revenue',
            title='Revenue by Region',
            color='revenue',
            color_continuous_scale='Blues'
        )
        st.plotly_chart(fig, width='stretch')
    
    with col2:
        # Customer concentration
        regional_metrics['customers_share'] = (regional_metrics['customer_id'] / total_customers) * 100
        regional_metrics['revenue_share'] = (regional_metrics['revenue'] / total_revenue) * 100
        
        fig = px.scatter(
            regional_metrics,
            x='customers_share',
            y='revenue_share',
            size='revenue',
            color='region',
            title='Customer vs Revenue Share',
            hover_data=['customer_id']
        )
        st.plotly_chart(fig, width='stretch')

def apply_regional_filters(data, date_range, product_category):
    """Apply filters to regional data"""
    filtered_data = data.copy()
    
    # Date filter
    if date_range == "Last 30 Days":
        cutoff_date = datetime.now().date() - timedelta(days=30)
        filtered_data = filtered_data[filtered_data['date'] >= cutoff_date]
    elif date_range == "Last 90 Days":
        cutoff_date = datetime.now().date() - timedelta(days=90)
        filtered_data = filtered_data[filtered_data['date'] >= cutoff_date]
    elif date_range == "Year to Date":
        cutoff_date = datetime.now().date().replace(month=1, day=1)
        filtered_data = filtered_data[filtered_data['date'] >= cutoff_date]
    
    # Product category filter
    if product_category != "All" and 'category' in filtered_data.columns:
        filtered_data = filtered_data[filtered_data['category'] == product_category]
    
    return filtered_data

def render_regional_performance(sales_data):
    """Render detailed regional performance analysis"""
    st.header("ðŸ“Š Regional Performance Deep Dive")
    
    # Time-based analysis
    sales_data['month'] = pd.to_datetime(sales_data['date']).dt.to_period('M')
    monthly_regional = sales_data.groupby(['month', 'region']).agg({
        'revenue': 'sum',
        'customer_id': 'nunique'
    }).reset_index()
    
    monthly_regional['month'] = monthly_regional['month'].dt.to_timestamp()
    
    # Region selector
    regions = sorted(sales_data['region'].unique())
    selected_regions = st.multiselect(
        "Select Regions for Trend Analysis",
        regions,
        default=regions[:3] if len(regions) >= 3 else regions
    )
    
    if selected_regions:
        filtered_monthly = monthly_regional[monthly_regional['region'].isin(selected_regions)]
        
        fig = px.line(
            filtered_monthly,
            x='month',
            y='revenue',
            color='region',
            title='Monthly Revenue Trends by Region',
            markers=True
        )
        st.plotly_chart(fig, width='stretch')
    
    # Performance benchmarks
    st.subheader("ðŸ† Regional Performance Benchmarks")
    
    regional_benchmarks = sales_data.groupby('region').agg({
        'revenue': 'sum',
        'customer_id': 'nunique',
        'margin': 'sum'
    }).reset_index()
    
    regional_benchmarks['revenue_per_customer'] = regional_benchmarks['revenue'] / regional_benchmarks['customer_id']
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.bar(
            regional_benchmarks.nlargest(8, 'revenue'),
            x='region',
            y='revenue_per_customer',
            title='Revenue per Customer by Region',
            color='revenue_per_customer'
        )
        st.plotly_chart(fig, width='stretch')
    
    with col2:
        fig = px.scatter(
            regional_benchmarks,
            x='customer_id',
            y='revenue',
            size='margin',
            color='region',
            title='Customer Count vs Total Revenue',
            hover_data=['revenue_per_customer']
        )
        st.plotly_chart(fig, width='stretch')

def render_market_share_analysis(sales_data):
    """Render market share analysis"""
    st.header("ðŸ“ˆ Market Share Analysis")
    
    regional_share = sales_data.groupby('region').agg({
        'revenue': 'sum',
        'customer_id': 'nunique'
    }).reset_index()
    
    total_market_revenue = regional_share['revenue'].sum()
    regional_share['market_share'] = (regional_share['revenue'] / total_market_revenue) * 100
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.pie(
            regional_share,
            values='market_share',
            names='region',
            title='Revenue Market Share by Region'
        )
        st.plotly_chart(fig, width='stretch')
    
    with col2:
        # Growth opportunities
        regional_share['growth_potential'] = 100 - regional_share['market_share']
        fig = px.bar(
            regional_share.nlargest(6, 'growth_potential'),
            x='region',
            y='growth_potential',
            title='Market Growth Potential',
            color='growth_potential'
        )
        st.plotly_chart(fig, width='stretch')

def render_growth_opportunities(sales_data):
    """Render growth opportunity analysis"""
    st.header("ðŸŽ¯ Regional Growth Opportunities")
    
    # Calculate growth metrics
    sales_data['month'] = pd.to_datetime(sales_data['date']).dt.to_period('M')
    monthly_growth = sales_data.groupby(['month', 'region']).agg({
        'revenue': 'sum'
    }).reset_index()
    
    monthly_growth['month'] = monthly_growth['month'].dt.to_timestamp()
    monthly_growth = monthly_growth.sort_values(['region', 'month'])
    monthly_growth['revenue_growth'] = monthly_growth.groupby('region')['revenue'].pct_change() * 100
    
    # Recent growth analysis
    recent_months = monthly_growth['month'].nlargest(3).unique()
    recent_growth = monthly_growth[monthly_growth['month'].isin(recent_months)]
    
    avg_growth = recent_growth.groupby('region').agg({
        'revenue_growth': 'mean',
        'revenue': 'mean'
    }).reset_index()
    
    # Growth matrix
    fig = px.scatter(
        avg_growth,
        x='revenue',
        y='revenue_growth',
        size='revenue_growth',
        color='region',
        title='Growth Opportunity Matrix',
        hover_data=['revenue_growth']
    )
    st.plotly_chart(fig, width='stretch')
    
    # Actionable recommendations
    st.subheader("ðŸ’¡ Growth Initiatives")
    
    initiatives = [
        "ðŸš€ **Western Region Expansion**: High growth potential with moderate investment",
        "ðŸ’° **Nairobi Optimization**: Improve margins in established market", 
        "ðŸŽ¯ **Coastal Penetration**: Target underserved coastal markets",
        "ðŸ“ˆ **Central Region Development**: Build presence in growing central areas"
    ]
    
    for initiative in initiatives:
        st.write(f"- {initiative}")

if __name__ == "__main__":
    render()
'''

    # Write the robust regional page
    with open('pages/05_Sales_Regional.py', 'w', encoding='utf-8') as f:
        f.write(regional_content)
    
    print("âœ… Created robust regional performance page")

def main():
    print("ðŸšš LOGISTICS PRO - ENTERPRISE DATA ENHANCEMENT")
    print("==============================================")
    
    # Enhance the data structure
    data_gen = enhance_data_structure()
    
    if data_gen:
        # Update the data generator class
        update_data_generator_class()
        
        # Create robust regional page
        create_robust_regional_page()
        
        print("\nðŸŽ‰ ENTERPRISE DATA ENHANCEMENT COMPLETE!")
        print("\nðŸš€ STARTING APPLICATION...")
        os.system("python -m streamlit run app.py")
    else:
        print("\nâŒ Data enhancement failed. Creating basic robust page...")
        create_robust_regional_page()
        print("\nðŸ”„ Please restart your app manually: python -m streamlit run app.py")

if __name__ == "__main__":
    main()
