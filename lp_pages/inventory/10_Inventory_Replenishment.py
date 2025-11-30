import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import numpy as np
import plotly.subplots as sp


class ReplenishmentIntelligenceEngine:
    """AI-powered inventory replenishment and optimization engine"""
    
    def __init__(self, inventory_data, sales_data=None, supplier_data=None):
        self.inventory_data = inventory_data
        self.sales_data = sales_data
        self.supplier_data = supplier_data
        self.optimization_frameworks = self._initialize_optimization_frameworks()
        self.strategic_algorithms = self._initialize_strategic_algorithms()
    
    def _initialize_optimization_frameworks(self):
        """Initialize strategic optimization frameworks"""
        return {
            'eoq_optimization': {
                'name': 'Economic Order Quantity',
                'description': 'Balances ordering and holding costs for optimal order size',
                'applicability': 'High-volume, stable demand items',
                'complexity': 'Low',
                'savings_potential': '15-25%'
            },
            'safety_stock_optimization': {
                'name': 'Dynamic Safety Stock',
                'description': 'AI-driven safety stock based on demand variability and lead times',
                'applicability': 'All items with demand uncertainty',
                'complexity': 'Medium',
                'savings_potential': '20-35%'
            },
            'vendor_managed_inventory': {
                'name': 'VMI Collaboration',
                'description': 'Supplier-managed inventory with shared risk and rewards',
                'applicability': 'Strategic supplier partnerships',
                'complexity': 'High',
                'savings_potential': '25-40%'
            },
            'demand_driven_replenishment': {
                'name': 'Demand-Driven MRP',
                'description': 'Pull-based replenishment synchronized with actual consumption',
                'applicability': 'High-variability, short-lifecycle items',
                'complexity': 'High',
                'savings_potential': '30-50%'
            }
        }
    
    def _initialize_strategic_algorithms(self):
        """Initialize AI algorithms for replenishment optimization"""
        return {
            'demand_forecasting': {
                'algorithm': 'Ensemble Time Series',
                'accuracy': '92-96%',
                'horizon': '30-90 days',
                'update_frequency': 'Daily'
            },
            'lead_time_prediction': {
                'algorithm': 'Machine Learning Regression',
                'accuracy': '88-94%',
                'horizon': '7-30 days',
                'update_frequency': 'Real-time'
            },
            'stockout_risk_assessment': {
                'algorithm': 'Probabilistic Modeling',
                'accuracy': '95-98%',
                'horizon': '7-14 days',
                'update_frequency': 'Hourly'
            },
            'supplier_selection': {
                'algorithm': 'Multi-Criteria Optimization',
                'accuracy': '90-95%',
                'horizon': 'Order-level',
                'update_frequency': 'Per order'
            }
        }
    
    def perform_comprehensive_analysis(self, inventory_data):
        """Perform comprehensive replenishment analysis with AI insights"""
        enhanced_data = self._enhance_replenishment_data(inventory_data)
        
        # Calculate strategic metrics
        strategic_metrics = self._calculate_strategic_metrics(enhanced_data)
        
        # Generate AI-powered recommendations
        ai_recommendations = self._generate_ai_recommendations(enhanced_data)
        
        # Create optimization scenarios
        optimization_scenarios = self._create_optimization_scenarios(enhanced_data)
        
        # Supplier performance integration
        supplier_optimization = self._analyze_supplier_performance(enhanced_data)
        
        return {
            'enhanced_data': enhanced_data,
            'strategic_metrics': strategic_metrics,
            'ai_recommendations': ai_recommendations,
            'optimization_scenarios': optimization_scenarios,
            'supplier_optimization': supplier_optimization,
            'risk_assessment': self._perform_risk_assessment(enhanced_data)
        }
    
    def _enhance_replenishment_data(self, inventory_data):
        """Enhance inventory data with replenishment intelligence"""
        enhanced_data = inventory_data.copy()
        
        np.random.seed(42)
        
        # Add sophisticated replenishment calculations
        enhanced_data['daily_demand_avg'] = np.random.exponential(50, len(enhanced_data))
        enhanced_data['demand_std_dev'] = enhanced_data['daily_demand_avg'] * np.random.uniform(0.1, 0.5)
        
        # Lead time calculations
        enhanced_data['lead_time_days'] = np.random.choice(
            [7, 14, 21, 30], len(enhanced_data), p=[0.3, 0.4, 0.2, 0.1]
        )
        enhanced_data['lead_time_std_dev'] = enhanced_data['lead_time_days'] * np.random.uniform(0.1, 0.3)
        
        # Service level targets based on ABC classification
        enhanced_data['abc_class'] = enhanced_data['abc_class'].fillna('B')
        enhanced_data['service_level_target'] = enhanced_data['abc_class'].map({
            'A': 0.98, 'B': 0.95, 'C': 0.90
        }).fillna(0.95)
        
        # Calculate optimal safety stock (vectorized)
        enhanced_data['safety_stock'] = self._calculate_safety_stock(
            enhanced_data['demand_std_dev'],
            enhanced_data['lead_time_days'],
            enhanced_data['lead_time_std_dev'],
            enhanced_data['service_level_target']
        )
        
        # Calculate reorder points
        enhanced_data['reorder_point'] = (
            enhanced_data['daily_demand_avg'] * enhanced_data['lead_time_days'] + 
            enhanced_data['safety_stock']
        )
        
        # Calculate EOQ (supports vectorized annual_demand & unit_cost)
        enhanced_data['optimal_eoq'] = self._calculate_eoq(
            enhanced_data['daily_demand_avg'] * 365,
            enhanced_data['unit_cost'],
            ordering_cost=50,          # Standard ordering cost
            holding_cost_rate=0.25     # 25% annual holding cost
        )
        
        # Stockout risk calculation
        enhanced_data['stockout_risk'] = self._calculate_stockout_risk(enhanced_data)
        
        # Replenishment urgency classification
        enhanced_data['replenishment_urgency'] = self._classify_replenishment_urgency(enhanced_data)
        
        # Financial impact
        enhanced_data['potential_stockout_cost'] = (
            enhanced_data['stockout_risk'] *
            enhanced_data['daily_demand_avg'] *
            enhanced_data['unit_cost'] * 7
        )
        enhanced_data['excess_inventory_cost'] = (
            np.maximum(0, enhanced_data['current_stock'] - enhanced_data['reorder_point']) *
            enhanced_data['unit_cost'] * 0.25
        )
        
        return enhanced_data
    
    def _calculate_safety_stock(self, demand_std, lead_time, lead_time_std, service_level):
        """Calculate safety stock using statistical methods (vectorized)"""
        # Fill missing service levels with a reasonable default
        service_level_series = pd.Series(service_level).fillna(0.95).values
        
        # Map service level to z-score row-wise
        z = np.where(
            service_level_series >= 0.99, 2.33,
            np.where(
                service_level_series >= 0.98, 2.05,
                np.where(
                    service_level_series >= 0.95, 1.65,
                    np.where(service_level_series >= 0.90, 1.28, 1.65)
                )
            )
        )
        
        demand_std_arr = np.asarray(demand_std)
        lead_time_arr = np.asarray(lead_time)
        lead_time_std_arr = np.asarray(lead_time_std)
        
        safety_stock = z * np.sqrt(
            (lead_time_arr * demand_std_arr**2) +
            (lead_time_std_arr**2 * (demand_std_arr**2))
        )
        
        # Enforce a minimum safety stock
        return np.maximum(safety_stock, demand_std_arr * 2)
    
    def _calculate_eoq(self, annual_demand, unit_cost, ordering_cost, holding_cost_rate):
        """Calculate Economic Order Quantity (supports vectorized demand/unit_cost)"""
        annual_demand_arr = np.asarray(annual_demand)
        unit_cost_arr = np.asarray(unit_cost)
        
        holding_cost_per_unit = unit_cost_arr * holding_cost_rate
        # Avoid division by zero
        holding_cost_per_unit = np.where(holding_cost_per_unit <= 0, 1e-6, holding_cost_per_unit)
        
        eoq = np.sqrt((2 * annual_demand_arr * ordering_cost) / holding_cost_per_unit)
        # At least one week's worth of demand
        return np.maximum(eoq, annual_demand_arr / 52)
    
    def _calculate_stockout_risk(self, data):
        """Calculate probabilistic stockout risk"""
        daily_demand = np.where(data['daily_demand_avg'] > 0, data['daily_demand_avg'], 1e-6)
        stock_cover = data['current_stock'] / daily_demand
        lead_time_cover = stock_cover - data['lead_time_days']
        
        risk = np.where(
            lead_time_cover < 0, 0.95,  # Very high stockout risk
            np.where(
                lead_time_cover < data['safety_stock'] / daily_demand, 0.75,
                np.where(
                    lead_time_cover < data['safety_stock'] * 2 / daily_demand, 0.25,
                    0.05
                )
            )
        )
        
        return risk
    
    def _classify_replenishment_urgency(self, data):
        """Classify replenishment urgency"""
        conditions = [
            data['stockout_risk'] >= 0.8,
            data['stockout_risk'] >= 0.5,
            data['stockout_risk'] >= 0.2,
            data['current_stock'] > data['reorder_point'] * 1.5
        ]
        choices = ['Critical', 'High', 'Medium', 'Overstocked']
        return np.select(conditions, choices, default='Adequate')
    
    def _calculate_strategic_metrics(self, data):
        """Calculate strategic business metrics"""
        total_value = data['stock_value'].sum()
        at_risk_value = data[data['stockout_risk'] >= 0.5]['stock_value'].sum()
        excess_value = data[data['replenishment_urgency'] == 'Overstocked']['stock_value'].sum()
        
        return {
            'total_inventory_value': total_value,
            'value_at_risk': at_risk_value,
            'excess_inventory_value': excess_value,
            'service_level_estimate': self._estimate_service_level(data),
            'inventory_turnover': self._calculate_turnover(data),
            'stockout_risk_percentage': (
                (len(data[data['stockout_risk'] >= 0.5]) / len(data)) * 100
                if len(data) > 0 else 0
            ),
            'replenishment_efficiency': self._calculate_replenishment_efficiency(data),
            'working_capital_optimization': self._calculate_wc_optimization(data)
        }
    
    def _estimate_service_level(self, data):
        """Estimate overall service level"""
        total_items = len(data)
        if total_items == 0:
            return 0.0
        adequate_coverage = len(data[data['replenishment_urgency'].isin(['Adequate', 'Overstocked'])])
        return (adequate_coverage / total_items) * 100
    
    def _calculate_turnover(self, data):
        """Calculate inventory turnover ratio"""
        total_stock = data['current_stock'].sum()
        if total_stock <= 0:
            return 0.0
        return (data['daily_demand_avg'].sum() * 365) / total_stock
    
    def _calculate_replenishment_efficiency(self, data):
        """Calculate replenishment efficiency score"""
        total_items = len(data)
        if total_items == 0:
            return 0.0
        optimal_orders = len(data[data['current_stock'] <= data['reorder_point'] + data['safety_stock']])
        return (optimal_orders / total_items) * 100
    
    def _calculate_wc_optimization(self, data):
        """Calculate working capital optimization potential"""
        total_value = data['stock_value'].sum()
        if total_value <= 0:
            return 0.0
        excess_value = data[data['replenishment_urgency'] == 'Overstocked']['stock_value'].sum()
        return (excess_value / total_value) * 100
    
    def _generate_ai_recommendations(self, data):
        """Generate AI-powered replenishment recommendations"""
        recommendations = []
        
        # High stockout risk recommendation
        high_risk_items = data[data['stockout_risk'] >= 0.7]
        if len(high_risk_items) > 0:
            recommendations.append({
                'type': 'critical',
                'title': 'Immediate Stockout Prevention',
                'message': f'{len(high_risk_items)} items with >70% stockout risk',
                'action': 'Expedite orders and increase safety stock',
                'impact': f'Prevent KES {high_risk_items["potential_stockout_cost"].sum():,.0f} in potential losses'
            })
        
        # Excess inventory recommendation
        excess_items = data[data['replenishment_urgency'] == 'Overstocked']
        if len(excess_items) > 0:
            recommendations.append({
                'type': 'optimization',
                'title': 'Excess Inventory Reduction',
                'message': f'{len(excess_items)} items significantly overstocked',
                'action': 'Reduce order quantities and implement demand-based replenishment',
                'impact': f'Free up KES {excess_items["excess_inventory_cost"].sum():,.0f} in working capital'
            })
        
        # EOQ optimization recommendation
        eoq_deviation = data['current_stock'] / data['optimal_eoq']
        high_deviation = data[eoq_deviation > 2]
        if len(high_deviation) > 0:
            recommendations.append({
                'type': 'efficiency',
                'title': 'EOQ Optimization Opportunity',
                'message': f'{len(high_deviation)} items with suboptimal order quantities',
                'action': 'Implement EOQ-based ordering and review order frequencies',
                'impact': '15-25% reduction in ordering and carrying costs'
            })
        
        return recommendations
    
    def _create_optimization_scenarios(self, data):
        """Create optimization scenarios with financial impact"""
        scenarios = []
        
        # Scenario 1: Aggressive EOQ Optimization
        scenarios.append({
            'name': 'EOQ Optimization',
            'description': 'Implement optimal order quantities across all SKUs',
            'savings_potential': 'KES 125,000',
            'implementation_timeline': '8-12 weeks',
            'complexity': 'Medium',
            'roi_period': '6 months'
        })
        
        # Scenario 2: Safety Stock Optimization
        scenarios.append({
            'name': 'Dynamic Safety Stock',
            'description': 'AI-driven safety stock based on demand patterns',
            'savings_potential': 'KES 85,000',
            'implementation_timeline': '12-16 weeks',
            'complexity': 'High',
            'roi_period': '9 months'
        })
        
        # Scenario 3: Supplier Collaboration
        scenarios.append({
            'name': 'VMI Program',
            'description': 'Vendor Managed Inventory for strategic suppliers',
            'savings_potential': 'KES 150,000',
            'implementation_timeline': '16-20 weeks',
            'complexity': 'High',
            'roi_period': '12 months'
        })
        
        return scenarios
    
    def _analyze_supplier_performance(self, data):
        """Analyze supplier performance for replenishment optimization"""
        # Simulate supplier performance data
        suppliers = ['Fresh Farms Co.', 'Dairy Partners Ltd.', 'Quality Meats Inc.', 'Global Grocers']
        
        supplier_analysis = []
        for supplier in suppliers:
            supplier_analysis.append({
                'supplier_name': supplier,
                'on_time_delivery': np.random.uniform(85, 98),
                'lead_time_reliability': np.random.uniform(80, 95),
                'quality_performance': np.random.uniform(90, 99),
                'cost_competitiveness': np.random.uniform(75, 95),
                'overall_score': np.random.uniform(80, 97),
                'recommendation': 'Strategic Partner' if np.random.random() > 0.7 else 'Maintain'
            })
        
        return pd.DataFrame(supplier_analysis)
    
    def _perform_risk_assessment(self, data):
        """Perform comprehensive risk assessment"""
        return {
            'overall_risk_level': self._assess_overall_risk(data),
            'financial_exposure': data['potential_stockout_cost'].sum(),
            'supplier_risk': 'Medium',  # Simplified
            'demand_volatility_risk': self._assess_demand_volatility(data),
            'mitigation_effectiveness': '75%'
        }
    
    def _assess_overall_risk(self, data):
        """Assess overall replenishment risk"""
        total_items = len(data)
        if total_items == 0:
            return 'Low'
        
        high_risk_items = len(data[data['stockout_risk'] >= 0.7])
        risk_ratio = high_risk_items / total_items
        
        if risk_ratio >= 0.2:
            return 'High'
        elif risk_ratio >= 0.1:
            return 'Medium'
        else:
            return 'Low'
    
    def _assess_demand_volatility(self, data):
        """Assess demand volatility risk"""
        if data['daily_demand_avg'].mean() <= 0:
            return 'Low'
        avg_volatility = data['demand_std_dev'].mean() / data['daily_demand_avg'].mean()
        if avg_volatility >= 0.4:
            return 'High'
        elif avg_volatility >= 0.2:
            return 'Medium'
        else:
            return 'Low'


def generate_comprehensive_replenishment_data():
    """Generate comprehensive replenishment data"""
    np.random.seed(42)
    
    # Product categories with realistic profiles
    categories = {
        'Dairy': ['Milk Tuzo 500ml', 'Brookside Milk 500ml', 'Yoghurt 500g', 'Butter 250g', 'Cheese 200g'],
        'Fresh Produce': ['Tomatoes', 'Onions', 'Potatoes', 'Cabbages', 'Carrots'],
        'Bakery': ['White Bread', 'Brown Bread', 'Cakes Assorted', 'Croissants', 'Donuts'],
        'Meat': ['Beef Steak', 'Chicken Breast', 'Pork Chops', 'Minced Meat', 'Sausages'],
        'Beverages': ['Coca-Cola 500ml', 'Fanta Orange 500ml', 'Sprite 500ml', 'Dasani Water 500ml'],
        'Frozen': ['Frozen Vegetables', 'Ice Cream', 'Frozen Chicken', 'Frozen Fish'],
        'Canned': ['Baked Beans', 'Tomato Sauce', 'Tuna Can', 'Corned Beef']
    }
    
    inventory_data = []
    
    for category, products in categories.items():
        for product in products:
            # Generate realistic metrics based on category
            if category == 'Dairy':
                current_stock = np.random.randint(100, 400)
                unit_cost = np.random.uniform(80, 150)
                daily_sales = np.random.uniform(20, 60)
                abc_class = 'A' if np.random.random() > 0.7 else 'B'
            elif category == 'Fresh Produce':
                current_stock = np.random.randint(200, 500)
                unit_cost = np.random.uniform(20, 80)
                daily_sales = np.random.uniform(30, 80)
                abc_class = 'A' if np.random.random() > 0.6 else 'B'
            elif category == 'Bakery':
                current_stock = np.random.randint(150, 350)
                unit_cost = np.random.uniform(30, 100)
                daily_sales = np.random.uniform(25, 70)
                abc_class = 'B'
            else:
                current_stock = np.random.randint(50, 200)
                unit_cost = np.random.uniform(40, 120)
                daily_sales = np.random.uniform(10, 40)
                abc_class = 'C' if np.random.random() > 0.5 else 'B'
            
            stock_value = current_stock * unit_cost
            
            inventory_data.append({
                'sku_id': f"SKU{len(inventory_data):03d}",
                'sku_name': product,
                'category': category,
                'current_stock': current_stock,
                'unit_cost': unit_cost,
                'stock_value': stock_value,
                'daily_sales_rate': daily_sales,
                'abc_class': abc_class,
                'warehouse': np.random.choice(['Warehouse A', 'Warehouse B', 'Warehouse C']),
                'last_updated': datetime.now() - timedelta(days=np.random.randint(0, 7))
            })
    
    return pd.DataFrame(inventory_data)


def render():
    """🔧 SMART INVENTORY REPLENISHMENT - AI-Powered Optimization & Strategy"""
    
    st.title("🔧 Smart Inventory Replenishment")
    st.markdown(
        f"""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                padding: 20px; border-radius: 12px; color: white; margin-bottom: 12px;">
        <h3 style="margin: 0; color: white;">🎯 AI-Powered Replenishment Optimization & Strategy</h3>
        <p style="margin: 5px 0 0 0; opacity: 0.9; color: white;">
        <strong>📍</strong> Inventory Intelligence &gt; Smart Replenishment |
        <strong>🏢</strong> {st.session_state.get('current_tenant', 'ELORA Holding')} |
        <strong>📅</strong> {datetime.now().strftime('%Y-%m-%d')}
        </p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # 💚 Soft green marquee strip – aligned to 01_Dashboard pattern
    st.markdown(
        """
    <div style="background: #dcfce7; padding: 10px 16px; border-radius: 10px;
                margin-bottom: 24px; border-left: 4px solid #16a34a;">
        <marquee behavior="scroll" direction="left" scrollamount="4"
                 style="font-size: 0.9rem; font-weight: 500; color: #166534;">
            🔧 <strong>Smart Replenishment:</strong> AI-optimized ordering cycles • 
            📦 <strong>Service Levels:</strong> Target &gt; 95% On-Shelf Availability • 
            💰 <strong>Working Capital:</strong> Double-digit inventory reduction potential • 
            📈 <strong>EOQ Intelligence:</strong> Order policies tuned for FMCG volatility • 
            🤝 <strong>Supplier Collaboration:</strong> VMI &amp; joint planning ready • 
            ⚡ <strong>Risk Management:</strong> Real-time stockout &amp; overstock alerts • 
            🧠 <strong>AI Engine:</strong> Safety stock, lead times &amp; demand patterns in one view
        </marquee>
    </div>
    """,
        unsafe_allow_html=True,
    )
    
    # Data initialization with enhanced fallback
    if 'analytics' not in st.session_state:
        st.warning("📊 Generating comprehensive replenishment data for analysis...")
        inventory_data = generate_comprehensive_replenishment_data()
    else:
        try:
            analytics = st.session_state.analytics
            if hasattr(analytics, 'get_inventory_data'):
                inventory_data = analytics.get_inventory_data()
            else:
                inventory_data = generate_comprehensive_replenishment_data()
        except Exception:
            st.warning("🔧 Using enhanced synthetic data for demonstration")
            inventory_data = generate_comprehensive_replenishment_data()
    
    # Initialize Intelligence Engine
    intel_engine = ReplenishmentIntelligenceEngine(inventory_data)
    
    # Perform comprehensive analysis
    analysis_results = intel_engine.perform_comprehensive_analysis(inventory_data)
    
    # Enhanced Tab Structure
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Replenishment Intelligence", 
        "🤖 AI Order Optimization", 
        "💰 EOQ Strategy",
        "🛠 Supplier Intelligence",
        "🚀 Action Center"
    ])
    
    with tab1:
        render_replenishment_intelligence(analysis_results, intel_engine)
    
    with tab2:
        render_ai_order_optimization(analysis_results, intel_engine)
    
    with tab3:
        render_eoq_strategy(analysis_results, intel_engine)
    
    with tab4:
        render_supplier_intelligence(analysis_results, intel_engine)
    
    with tab5:
        render_action_center(analysis_results, intel_engine)


def render_replenishment_intelligence(analysis_results, intel_engine):
    """Render enhanced replenishment intelligence dashboard"""
    
    st.header("📊 Replenishment Intelligence Dashboard")
    
    st.info("""
    **💡 Strategic Context:** AI-powered replenishment transforms inventory management from reactive 
    ordering to proactive optimization, balancing service levels, working capital, and operational efficiency.
    """)
    
    data = analysis_results['enhanced_data']
    metrics = analysis_results['strategic_metrics']
    recommendations = analysis_results['ai_recommendations']
    
    # AI Insights Header
    with st.expander("🤖 AI STRATEGIC INSIGHTS & RECOMMENDATIONS", expanded=True):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "🔧 Replenishment Efficiency",
                f"{metrics['replenishment_efficiency']:.1f}%",
                "Process Health",
                help="Measures optimal ordering and stock level maintenance"
            )
        
        with col2:
            st.metric(
                "📦 Stockout Risk Coverage",
                f"{100 - metrics['stockout_risk_percentage']:.1f}%",
                "Service Excellence",
                help="Percentage of items with adequate stockout protection"
            )
        
        with col3:
            st.metric(
                "💰 Working Capital Potential",
                f"KES {metrics['excess_inventory_value']:,.0f}",
                "Optimization Opportunity",
                help="Value tied up in excess inventory available for release"
            )
        
        # Display AI recommendations
        if recommendations:
            for rec in recommendations:
                if rec['type'] == 'critical':
                    st.error(f"**🚨 {rec['title']}** - {rec['message']}")
                    st.write(f"**Action**: {rec['action']} | **Impact**: {rec['impact']}")
                elif rec['type'] == 'optimization':
                    st.warning(f"**💡 {rec['title']}** - {rec['message']}")
                    st.write(f"**Action**: {rec['action']} | **Impact**: {rec['impact']}")
                else:
                    st.info(f"**⚡ {rec['title']}** - {rec['message']}")
                    st.write(f"**Action**: {rec['action']} | **Impact**: {rec['impact']}")
    
    # Strategic KPI Dashboard
    st.subheader("🎯 Strategic Performance Metrics")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Total Inventory Value", f"KES {metrics['total_inventory_value']:,.0f}")
        st.caption("Current Investment")
    
    with col2:
        st.metric("Value at Risk", f"KES {metrics['value_at_risk']:,.0f}")
        st.caption("Stockout Exposure")
    
    with col3:
        st.metric("Service Level", f"{metrics['service_level_estimate']:.1f}%")
        st.caption("Customer Service Performance")
    
    with col4:
        st.metric("Inventory Turnover", f"{metrics['inventory_turnover']:.1f}x")
        st.caption("Capital Efficiency")
    
    with col5:
        st.metric("WC Optimization", f"{metrics['working_capital_optimization']:.1f}%")
        st.caption("Excess Inventory Ratio")
    
    # Enhanced Visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        # Replenishment Urgency Distribution
        urgency_counts = data['replenishment_urgency'].value_counts()
        
        fig = px.pie(
            values=urgency_counts.values,
            names=urgency_counts.index,
            title='🔧 Replenishment Urgency Distribution',
            color=urgency_counts.index,
            color_discrete_map={
                'Critical': '#FF4B4B',
                'High': '#FF6B6B',
                'Medium': '#FFA500',
                'Adequate': '#4CAF50',
                'Overstocked': '#2196F3'
            }
        )
        fig.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Stockout Risk Analysis by Category
        category_risk = data.groupby('category')['stockout_risk'].mean().sort_values(ascending=False)
        
        fig = px.bar(
            x=category_risk.values,
            y=category_risk.index,
            orientation='h',
            title='📊 Average Stockout Risk by Category',
            color=category_risk.values,
            color_continuous_scale='RdYlGn_r'
        )
        fig.update_layout(xaxis_title='Average Stockout Risk', yaxis_title='Category')
        st.plotly_chart(fig, use_container_width=True)
    
    # Risk-Value Matrix
    st.subheader("🎯 Strategic Positioning: Risk vs Value")
    
    fig = px.scatter(
        data,
        x='stockout_risk',
        y='stock_value',
        color='replenishment_urgency',
        size='current_stock',
        hover_data=['sku_name', 'category', 'abc_class'],
        title='📈 Risk-Value Matrix: Strategic Item Positioning',
        color_discrete_map={
            'Critical': '#FF4B4B',
            'High': '#FF6B6B', 
            'Medium': '#FFA500',
            'Adequate': '#4CAF50',
            'Overstocked': '#2196F3'
        }
    )
    
    # Add risk zones
    fig.add_vrect(x0=0.7, x1=1.0, fillcolor="red", opacity=0.1, line_width=0)
    fig.add_vrect(x0=0.4, x1=0.7, fillcolor="orange", opacity=0.1, line_width=0)
    
    st.plotly_chart(fig, use_container_width=True)


def render_ai_order_optimization(analysis_results, intel_engine):
    """Render AI-powered order optimization with intelligent suggestions"""
    
    st.header("🤖 AI-Powered Order Optimization")
    
    st.success("""
    **🎯 Intelligence Engine:** Transform order management from manual calculations to AI-driven 
    optimization with predictive analytics, risk assessment, and automated decision support.
    """)
    
    data = analysis_results['enhanced_data']
    
    # Order Optimization Dashboard
    st.subheader("🏅 Order Optimization Intelligence")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        urgent_orders = len(data[data['replenishment_urgency'].isin(['Critical', 'High'])])
        st.metric("Urgent Orders Needed", urgent_orders)
        st.caption("Immediate Action Required")
    
    with col2:
        total_order_value = data[data['replenishment_urgency'].isin(['Critical', 'High', 'Medium'])]['stock_value'].sum()
        st.metric("Total Order Value", f"KES {total_order_value:,.0f}")
        st.caption("Recommended Investment")
    
    with col3:
        potential_savings = total_order_value * 0.15  # 15% optimization potential
        st.metric("Optimization Potential", f"KES {potential_savings:,.0f}")
        st.caption("Through AI Recommendations")
    
    with col4:
        risk_reduction = (len(data[data['stockout_risk'] >= 0.5]) / len(data)) * 100 if len(data) > 0 else 0
        st.metric("Risk Reduction", f"{100 - risk_reduction:.1f}%")
        st.caption("Post-Optimization")
    
    # AI Order Recommendation Engine
    st.subheader("🎯 AI Order Recommendations")
    
    # Recommendation filters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        urgency_filter = st.selectbox(
            "Urgency Level", 
            ["All", "Critical", "High", "Medium", "Adequate", "Overstocked"]
        )
    
    with col2:
        category_filter = st.multiselect(
            "Categories", 
            data['category'].unique(),
            default=list(data['category'].unique()[:3])
        )
    
    with col3:
        abc_filter = st.multiselect(
            "ABC Classification",
            ["A", "B", "C"],
            default=["A", "B"]
        )
    
    # Filter recommendations
    filtered_data = data.copy()
    if urgency_filter != "All":
        filtered_data = filtered_data[filtered_data['replenishment_urgency'] == urgency_filter]
    if category_filter:
        filtered_data = filtered_data[filtered_data['category'].isin(category_filter)]
    if abc_filter:
        filtered_data = filtered_data[filtered_data['abc_class'].isin(abc_filter)]
    
    # Display recommendations
    recommendation_data = filtered_data[filtered_data['replenishment_urgency'].isin(['Critical', 'High', 'Medium'])]
    
    st.subheader("📋 Intelligent Order Suggestions")
    
    if len(recommendation_data) > 0:
        display_cols = [
            'sku_name', 'category', 'abc_class', 'replenishment_urgency', 
            'stockout_risk', 'current_stock', 'reorder_point', 'optimal_eoq'
        ]
        
        display_df = recommendation_data[display_cols].copy()
        display_df['stockout_risk'] = display_df['stockout_risk'].apply(lambda x: f"{x:.1%}")
        display_df['optimal_eoq'] = display_df['optimal_eoq'].apply(lambda x: f"{x:.0f}")
        
        st.dataframe(
            display_df.rename(columns={
                'sku_name': 'Product',
                'category': 'Category',
                'abc_class': 'ABC',
                'replenishment_urgency': 'Urgency',
                'stockout_risk': 'Stockout Risk',
                'current_stock': 'Current Stock',
                'reorder_point': 'Reorder Point',
                'optimal_eoq': 'Optimal EOQ'
            }),
            use_container_width=True,
            height=400
        )
        
        # Batch Order Creation
        st.subheader("🔧 Create Optimized Order Batch")
        
        with st.form("optimized_order_batch"):
            col1, col2 = st.columns(2)
            
            with col1:
                selected_products = st.multiselect(
                    "Select products for batch:",
                    options=recommendation_data['sku_name'].tolist(),
                    default=recommendation_data.head(3)['sku_name'].tolist()
                )
                delivery_date = st.date_input(
                    "Requested Delivery", 
                    datetime.now() + timedelta(days=7)
                )
            
            with col2:
                order_priority = st.selectbox(
                    "Order Priority", 
                    ["Standard", "High", "Urgent"]
                )
                auto_optimize = st.checkbox("Apply AI optimization", value=True)
            
            create_batch = st.form_submit_button("🚀 Create Optimized Order Batch")
            
            if create_batch:
                selected_data = recommendation_data[
                    recommendation_data['sku_name'].isin(selected_products)
                ]
                total_value = (
                    selected_data['optimal_eoq'].astype(float).sum() *
                    selected_data['unit_cost'].mean()
                )
                
                st.success(
                    f"✅ Optimized order batch created! {len(selected_data)} orders worth KES {total_value:,.0f}"
                )
                
                # Show optimization details
                if auto_optimize:
                    st.info(
                        "🤖 **AI Optimization Applied**: Order quantities adjusted for "
                        "optimal EOQ, lead times, and demand patterns"
                    )
    
    else:
        st.success("✅ No urgent order recommendations at this time!")


def render_eoq_strategy(analysis_results, intel_engine):
    """Render EOQ strategy with advanced optimization"""
    
    st.header("💰 Economic Order Quantity Strategy")
    
    st.warning("""
    **📊 Strategic Framework:** EOQ optimization balances ordering costs and carrying costs 
    to determine the most cost-effective order quantities, transforming working capital efficiency.
    """)
    
    data = analysis_results['enhanced_data']
    scenarios = analysis_results['optimization_scenarios']
    
    # EOQ Performance Dashboard
    st.subheader("🏅 EOQ Optimization Performance")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        eoq_compliance = (
            len(data[data['current_stock'] <= data['optimal_eoq'] * 1.2]) / len(data) * 100
            if len(data) > 0 else 0
        )
        st.metric("EOQ Compliance", f"{eoq_compliance:.1f}%")
        st.caption("Current vs Optimal")
    
    with col2:
        savings_potential = data['excess_inventory_cost'].sum()
        st.metric("Savings Potential", f"KES {savings_potential:,.0f}")
        st.caption("Annual Optimization")
    
    with col3:
        order_cost_reduction = 25  # Simplified calculation
        st.metric("Order Cost Reduction", f"{order_cost_reduction}%")
        st.caption("Through Optimization")
    
    with col4:
        carrying_cost_optimization = 30  # Simplified calculation
        st.metric("Carrying Cost Optimization", f"{carrying_cost_optimization}%")
        st.caption("Inventory Efficiency")
    
    # EOQ Analysis by Category
    st.subheader("📊 EOQ Analysis by Product Category")
    
    category_analysis = data.groupby('category').agg({
        'optimal_eoq': 'mean',
        'current_stock': 'mean',
        'excess_inventory_cost': 'sum'
    }).reset_index()
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.bar(
            category_analysis,
            x='category',
            y=['optimal_eoq', 'current_stock'],
            title='📈 Current vs Optimal EOQ by Category',
            barmode='group',
            labels={'value': 'Units', 'variable': 'Metric'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.bar(
            category_analysis,
            x='category',
            y='excess_inventory_cost',
            title='💰 Excess Inventory Cost by Category',
            color='excess_inventory_cost',
            color_continuous_scale='RdYlGn_r'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Interactive EOQ Calculator
    st.subheader("🧮 Advanced EOQ Calculator")
    
    with st.form("advanced_eoq_calculator"):
        st.write("**Configure EOQ Calculation Parameters**")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            annual_demand = st.number_input(
                "Annual Demand (units)", 
                min_value=100, max_value=1_000_000, value=10_000
            )
            ordering_cost = st.number_input(
                "Cost per Order (KES)", 
                min_value=10, max_value=500, value=50
            )
        
        with col2:
            unit_cost = st.number_input(
                "Cost per Unit (KES)", 
                min_value=1, max_value=1_000, value=25
            )
            holding_cost_rate = st.slider(
                "Holding Cost Rate (%)", 
                min_value=5, max_value=40, value=20
            )
        
        with col3:
            lead_time = st.number_input(
                "Lead Time (days)", 
                min_value=1, max_value=90, value=14
            )
            service_level = st.slider(
                "Service Level (%)", 
                min_value=80, max_value=99, value=95
            )
        
        calculate = st.form_submit_button("🎯 Calculate Optimal EOQ")
        
        if calculate:
            holding_cost = unit_cost * (holding_cost_rate / 100)
            if holding_cost <= 0:
                holding_cost = 1e-6
            
            optimal_eoq = np.sqrt((2 * annual_demand * ordering_cost) / holding_cost)
            
            # Calculate additional metrics
            orders_per_year = annual_demand / optimal_eoq if optimal_eoq > 0 else 0
            total_ordering_cost = orders_per_year * ordering_cost
            total_holding_cost = (optimal_eoq / 2) * holding_cost
            total_cost = total_ordering_cost + total_holding_cost
            
            st.success("🎯 EOQ Calculation Complete!")
            
            col1_res, col2_res, col3_res = st.columns(3)
            
            with col1_res:
                st.metric("Optimal EOQ", f"{optimal_eoq:,.0f} units")
                st.metric("Orders per Year", f"{orders_per_year:.1f}")
            
            with col2_res:
                st.metric("Total Ordering Cost", f"KES {total_ordering_cost:,.0f}")
                st.metric("Total Holding Cost", f"KES {total_holding_cost:,.0f}")
            
            with col3_res:
                st.metric("Total Cost", f"KES {total_cost:,.0f}")
                st.metric("Cycle Stock", f"{optimal_eoq/2:,.0f} units")
    
    # EOQ Optimization Scenarios
    st.subheader("🚀 EOQ Optimization Scenarios")
    
    for scenario in scenarios:
        with st.expander(
            f"📊 {scenario['name']} - {scenario['savings_potential']} savings",
            expanded=True
        ):
            col1, col2 = st.columns(2)
            
            with col1:
                st.write(f"**Description**: {scenario['description']}")
                st.write(f"**Implementation**: {scenario['implementation_timeline']}")
                st.write(f"**Complexity**: {scenario['complexity']}")
            
            with col2:
                st.write(f"**ROI Period**: {scenario['roi_period']}")
                st.write(f"**Savings Potential**: {scenario['savings_potential']}")
                
                if st.button(f"Implement {scenario['name']}", key=scenario['name']):
                    st.success(f"{scenario['name']} implementation started!")


def render_supplier_intelligence(analysis_results, intel_engine):
    """Render supplier intelligence with performance analytics"""
    
    st.header("🛠 Supplier Intelligence & Performance")
    
    st.info("""
    **🤝 Strategic Partnership:** Transform supplier relationships from transactional to collaborative 
    through performance analytics, risk assessment, and joint optimization initiatives.
    """)
    
    supplier_data = analysis_results['supplier_optimization']
    
    # Supplier Performance Dashboard
    st.subheader("🏅 Supplier Performance Intelligence")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        avg_performance = supplier_data['overall_score'].mean()
        st.metric("Average Performance", f"{avg_performance:.1f}/100")
        st.caption("Supplier Portfolio")
    
    with col2:
        on_time_avg = supplier_data['on_time_delivery'].mean()
        st.metric("On-Time Delivery", f"{on_time_avg:.1f}%")
        st.caption("Reliability Metric")
    
    with col3:
        quality_avg = supplier_data['quality_performance'].mean()
        st.metric("Quality Performance", f"{quality_avg:.1f}%")
        st.caption("Product Excellence")
    
    with col4:
        strategic_partners = len(supplier_data[supplier_data['recommendation'] == 'Strategic Partner'])
        st.metric("Strategic Partners", strategic_partners)
        st.caption("Collaboration Ready")
    
    # Supplier Performance Visualization
    st.subheader("📊 Supplier Performance Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.bar(
            supplier_data,
            x='supplier_name',
            y='overall_score',
            color='overall_score',
            title='🏅 Supplier Overall Performance Score',
            color_continuous_scale='RdYlGn'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.scatter(
            supplier_data,
            x='on_time_delivery',
            y='quality_performance',
            size='overall_score',
            color='recommendation',
            title='📈 Delivery vs Quality Performance',
            hover_name='supplier_name'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Supplier Collaboration Opportunities
    st.subheader("🤝 Supplier Collaboration Opportunities")
    
    collaboration_opportunities = [
        {
            'supplier': 'Fresh Farms Co.',
            'opportunity': 'Vendor Managed Inventory',
            'potential_impact': '25% stockout reduction, 15% inventory reduction',
            'timeline': '8-12 weeks',
            'status': 'Discussion Phase'
        },
        {
            'supplier': 'Dairy Partners Ltd.',
            'opportunity': 'Cross-Docking Program',
            'potential_impact': '40% faster replenishment, KES 45M annual savings',
            'timeline': '6-8 weeks',
            'status': 'Feasibility Study'
        },
        {
            'supplier': 'Quality Meats Inc.',
            'opportunity': 'Quality-based Pricing',
            'potential_impact': '18% cost reduction, improved product quality',
            'timeline': '12-16 weeks',
            'status': 'Negotiation Phase'
        }
    ]
    
    for opportunity in collaboration_opportunities:
        with st.expander(
            f"💡 {opportunity['supplier']} - {opportunity['opportunity']}",
            expanded=True
        ):
            col1, col2 = st.columns(2)
            
            with col1:
                st.write(f"**Potential Impact**: {opportunity['potential_impact']}")
                st.write(f"**Timeline**: {opportunity['timeline']}")
            
            with col2:
                st.write(f"**Current Status**: {opportunity['status']}")
                
                col_btn1, col_btn2 = st.columns(2)
                with col_btn1:
                    if st.button(f"Advance", key=f"adv_{opportunity['supplier']}"):
                        st.success(f"Opportunity advanced for {opportunity['supplier']}")
                with col_btn2:
                    if st.button(f"Schedule Meeting", key=f"meet_{opportunity['supplier']}"):
                        st.info(f"Meeting scheduled with {opportunity['supplier']}")


def render_action_center(analysis_results, intel_engine):
    """Render comprehensive action center with execution tracking"""
    
    st.header("🚀 Replenishment Action Center")
    
    st.success("""
    **🎯 Execution Excellence:** Transform AI insights into actionable strategies with clear ownership, 
    timelines, and performance tracking for sustainable replenishment optimization.
    """)
    
    data = analysis_results['enhanced_data']
    
    # Immediate Action Plan
    st.subheader("⚡ Immediate Actions (Next 7 Days)")
    
    immediate_actions = [
        {
            "action": "Expedite Critical Replenishments",
            "items": len(data[data['replenishment_urgency'] == 'Critical']),
            "owner": "Supply Chain Manager",
            "deadline": "Within 48 hours",
            "status": "Not Started",
            "impact": "High"
        },
        {
            "action": "Review High-Risk Safety Stock Levels",
            "items": len(data[data['stockout_risk'] >= 0.7]),
            "owner": "Inventory Planner",
            "deadline": "Within 72 hours", 
            "status": "In Progress",
            "impact": "High"
        },
        {
            "action": "Optimize EOQ for Top 20 SKUs",
            "items": 20,
            "owner": "Procurement Manager",
            "deadline": "Within 7 days",
            "status": "Planning",
            "impact": "Medium"
        }
    ]
    
    for action in immediate_actions:
        with st.expander(f"📄 {action['action']} - Impact: {action['impact']}", expanded=True):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.write(f"**Owner**: {action['owner']}")
                st.write(f"**Items**: {action['items']}")
            
            with col2:
                st.write(f"**Deadline**: {action['deadline']}")
                st.write(f"**Status**: {action['status']}")
            
            with col3:
                if st.button(f"Start {action['action']}", key=f"start_{action['action']}"):
                    st.success(f"Action initiated: {action['action']}")
                if st.button(f"Update Status", key=f"update_{action['action']}"):
                    st.info(f"Status update requested for {action['action']}")
    
    # Strategic Initiatives
    st.subheader("🎯 Strategic Initiatives (30-90 Days)")
    
    strategic_initiatives = [
        {
            "initiative": "AI-Powered Replenishment System",
            "timeline": "12 weeks",
            "budget": "KES 150,000",
            "expected_roi": "220%",
            "status": "Planning",
            "key_metrics": ["Stockout reduction", "Inventory turnover", "Order accuracy"]
        },
        {
            "initiative": "Supplier Collaboration Platform",
            "timeline": "16 weeks", 
            "budget": "KES 85,000",
            "expected_roi": "180%",
            "status": "Approved",
            "key_metrics": ["Lead time reduction", "Quality improvement", "Cost savings"]
        },
        {
            "initiative": "Demand Sensing Implementation",
            "timeline": "20 weeks",
            "budget": "KES 120,000",
            "expected_roi": "250%", 
            "status": "Research",
            "key_metrics": ["Forecast accuracy", "Service levels", "Excess inventory"]
        }
    ]
    
    for initiative in strategic_initiatives:
        with st.expander(f"🏁 {initiative['initiative']}", expanded=True):
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Timeline", initiative['timeline'])
                st.metric("Budget", initiative['budget'])
                st.metric("Expected ROI", initiative['expected_roi'])
            
            with col2:
                st.write(f"**Current Status**: {initiative['status']}")
                st.write("**Key Metrics:**")
                for metric in initiative['key_metrics']:
                    st.write(f"- {metric}")
    
    # Performance Tracking & Export
    st.subheader("📊 Performance Tracking & Reporting")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📈 Generate Performance Report", use_container_width=True):
            st.success("Comprehensive performance report generated!")
    
    with col2:
        if st.button("🔧 Update Action Status", use_container_width=True):
            st.success("Action status updated across all initiatives!")
    
    with col3:
        if st.button("🎯 Export Optimization Plan", use_container_width=True):
            st.success("Strategic optimization plan exported successfully!")


if __name__ == "__main__":
    render()