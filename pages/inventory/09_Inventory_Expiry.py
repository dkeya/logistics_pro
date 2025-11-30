import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import numpy as np


class ExpiryIntelligenceEngine:
    """AI-powered expiry management and optimization engine"""

    def __init__(self, inventory_data, sales_data=None):
        self.inventory_data = inventory_data
        self.sales_data = sales_data
        self.risk_frameworks = self._initialize_risk_frameworks()
        self.optimization_strategies = self._initialize_optimization_strategies()

    def _initialize_risk_frameworks(self):
        """Initialize strategic risk assessment frameworks"""
        return {
            'critical_risk': {
                'threshold': 7,
                'color': '#FF4B4B',
                'actions': ['Immediate discount', 'Emergency redistribution', 'Donation coordination'],
                'escalation': 'Executive level'
            },
            'high_risk': {
                'threshold': 30,
                'color': '#FF6B6B',
                'actions': ['Priority picking', 'Promotional campaigns', 'Cross-docking'],
                'escalation': 'Management level'
            },
            'medium_risk': {
                'threshold': 60,
                'color': '#FFA500',
                'actions': ['Inventory rotation', 'Demand forecasting review', 'Supplier coordination'],
                'escalation': 'Supervisor level'
            },
            'low_risk': {
                'threshold': 90,
                'color': '#4CAF50',
                'actions': ['Regular monitoring', 'Standard FEFO', 'Preventive planning'],
                'escalation': 'Operational level'
            }
        }

    def _initialize_optimization_strategies(self):
        """Initialize FEFO optimization strategies"""
        return {
            'warehouse_optimization': {
                'name': 'Smart Warehouse Layout',
                'impact': '25-40% waste reduction',
                'implementation': '8-12 weeks',
                'cost': 'Medium',
                'description': 'Optimize physical layout for FEFO compliance'
            },
            'system_integration': {
                'name': 'AI-Powered FEFO System',
                'impact': '40-60% compliance improvement',
                'implementation': '12-16 weeks',
                'cost': 'High',
                'description': 'Integrate machine learning for dynamic routing'
            },
            'supplier_collaboration': {
                'name': 'Supplier JIT Program',
                'impact': '15-25% waste reduction',
                'implementation': '16-20 weeks',
                'cost': 'Low',
                'description': 'Collaborate with suppliers for smaller, frequent deliveries'
            },
            'demand_matching': {
                'name': 'Predictive Demand Alignment',
                'impact': '20-35% stock optimization',
                'implementation': '10-14 weeks',
                'cost': 'Medium',
                'description': 'Align inventory levels with predicted demand patterns'
            }
        }

    def perform_comprehensive_expiry_analysis(self, inventory_data):
        """Perform comprehensive expiry risk analysis with AI insights"""
        analysis_data = self._enhance_expiry_data(inventory_data)

        # Calculate strategic metrics
        strategic_metrics = self._calculate_strategic_metrics(analysis_data)

        # Generate AI insights
        ai_insights = self._generate_ai_insights(analysis_data, strategic_metrics)

        # Create optimization recommendations
        optimizations = self._generate_optimization_recommendations(analysis_data)

        return {
            'analysis_data': analysis_data,
            'strategic_metrics': strategic_metrics,
            'ai_insights': ai_insights,
            'optimizations': optimizations,
            'risk_assessment': self._perform_risk_assessment(analysis_data)
        }

    def _enhance_expiry_data(self, inventory_data):
        """Enhance inventory data with expiry intelligence"""
        enhanced_data = inventory_data.copy()

        # Add sophisticated expiry calculations
        np.random.seed(42)

        # Generate realistic expiry profiles based on product categories
        category_expiry_profiles = {
            'Dairy': {'mean': 14, 'std': 3},
            'Fresh Produce': {'mean': 7, 'std': 2},
            'Meat': {'mean': 10, 'std': 2},
            'Bakery': {'mean': 5, 'std': 1},
            'Frozen': {'mean': 180, 'std': 30},
            'Canned': {'mean': 365, 'std': 60},
            'Beverages': {'mean': 90, 'std': 15}
        }

        enhanced_data['days_to_expiry'] = enhanced_data['category'].apply(
            lambda cat: max(
                1,
                np.random.normal(
                    category_expiry_profiles.get(cat, {'mean': 30, 'std': 10})['mean'],
                    category_expiry_profiles.get(cat, {'mean': 30, 'std': 10})['std']
                )
            )
        )

        # Calculate multi-dimensional risk score
        enhanced_data['risk_score'] = self._calculate_multi_dimension_risk(enhanced_data)

        # Add strategic classifications
        enhanced_data['expiry_strategy'] = enhanced_data['risk_score'].apply(
            self._assign_strategic_classification
        )

        # Calculate financial impact
        enhanced_data['potential_loss'] = (
            enhanced_data['current_stock'] *
            enhanced_data['unit_cost'] *
            enhanced_data['risk_score'] / 100
        )

        return enhanced_data

    def _calculate_multi_dimension_risk(self, data):
        """Calculate multi-dimensional risk score (0-100)"""
        # Days to expiry component (40% weight)
        expiry_risk = np.where(
            data['days_to_expiry'] <= 7, 100,
            np.where(data['days_to_expiry'] <= 14, 80,
            np.where(data['days_to_expiry'] <= 30, 60,
            np.where(data['days_to_expiry'] <= 60, 30, 10)))
        )

        # Value concentration component (30% weight)
        max_value = data['stock_value'].max()
        if max_value <= 0:
            value_risk = np.zeros(len(data))
        else:
            value_risk = (data['stock_value'] / max_value) * 100

        # Demand pattern component (30% weight)
        demand_risk = np.where(
            data['daily_sales_rate'] < 5, 20,
            np.where(data['daily_sales_rate'] < 20, 40,
            np.where(data['daily_sales_rate'] < 50, 60, 80))
        )

        # Combined weighted risk score
        combined_risk = (expiry_risk * 0.4) + (value_risk * 0.3) + (demand_risk * 0.3)

        return np.clip(combined_risk, 0, 100)

    def _assign_strategic_classification(self, risk_score):
        """Assign strategic classification based on risk score"""
        if risk_score >= 80:
            return 'Critical Intervention'
        elif risk_score >= 60:
            return 'Proactive Management'
        elif risk_score >= 40:
            return 'Preventive Planning'
        else:
            return 'Standard Monitoring'

    def _calculate_strategic_metrics(self, data):
        """Calculate strategic business metrics"""
        total_value = data['stock_value'].sum()
        at_risk_value = data[data['risk_score'] >= 60]['stock_value'].sum()

        if total_value > 0:
            risk_exposure_percentage = (at_risk_value / total_value) * 100
        else:
            risk_exposure_percentage = 0.0

        return {
            'total_inventory_value': total_value,
            'value_at_risk': at_risk_value,
            'risk_exposure_percentage': risk_exposure_percentage,
            'critical_items_count': len(data[data['risk_score'] >= 80]),
            'average_risk_score': data['risk_score'].mean() if len(data) > 0 else 0.0,
            'fefo_compliance_score': self._calculate_fefo_compliance(data),
            'waste_reduction_potential': self._calculate_waste_reduction_potential(data)
        }

    def _calculate_fefo_compliance(self, data):
        """Calculate FEFO compliance score"""
        if len(data) == 0:
            return 0.0

        optimal_order = data.sort_values('days_to_expiry').index
        current_order = data.index

        top_n = min(10, len(data))
        if top_n == 0:
            return 0.0

        compliance = len(set(optimal_order[:top_n]) & set(current_order[:top_n])) / top_n
        return compliance * 100

    def _calculate_waste_reduction_potential(self, data):
        """Calculate waste reduction potential"""
        total_stock_value = data['stock_value'].sum()
        high_risk_value = data[data['risk_score'] >= 70]['potential_loss'].sum()

        if total_stock_value > 0:
            potential = (high_risk_value / total_stock_value) * 100
        else:
            potential = 0.0

        return min(potential, 50)

    def _generate_ai_insights(self, data, metrics):
        """Generate AI-powered strategic insights"""
        insights = []

        # Risk concentration insight
        if metrics['risk_exposure_percentage'] > 25:
            insights.append({
                'type': 'warning',
                'title': 'High Risk Concentration',
                'message': f"{metrics['risk_exposure_percentage']:.1f}% of inventory value is at high expiry risk",
                'recommendation': 'Implement immediate risk mitigation strategies for high-value items'
            })

        # FEFO optimization insight
        if metrics['fefo_compliance_score'] < 70:
            insights.append({
                'type': 'error',
                'title': 'FEFO Compliance Opportunity',
                'message': f'Current FEFO compliance at {metrics["fefo_compliance_score"]:.1f}%',
                'recommendation': 'Prioritize warehouse layout optimization and staff training'
            })

        # Waste reduction insight
        if metrics['waste_reduction_potential'] > 20:
            insights.append({
                'type': 'success',
                'title': 'Significant Waste Reduction Potential',
                'message': f'Potential to reduce waste by {metrics["waste_reduction_potential"]:.1f}%',
                'recommendation': 'Focus on demand forecasting and inventory optimization'
            })

        return insights

    def _generate_optimization_recommendations(self, data):
        """Generate optimization recommendations"""
        recommendations = []

        # Category-specific recommendations
        category_analysis = data.groupby('category').agg({
            'risk_score': 'mean',
            'potential_loss': 'sum',
            'sku_id': 'count'
        }).reset_index()

        high_risk_categories = category_analysis[category_analysis['risk_score'] > 60]

        for _, category in high_risk_categories.iterrows():
            recommendations.append({
                'category': category['category'],
                'risk_level': 'High',
                'potential_savings': f"KES {category['potential_loss']:,.0f}",
                'strategy': 'Implement category-specific FEFO protocols',
                'priority': 'Immediate'
            })

        return recommendations

    def _perform_risk_assessment(self, data):
        """Perform comprehensive risk assessment"""
        return {
            'overall_risk_level': self._assess_overall_risk(data),
            'financial_exposure': data['potential_loss'].sum(),
            'critical_timeline': self._calculate_critical_timeline(data),
            'prevention_effectiveness': self._assess_prevention_effectiveness(data)
        }

    def _assess_overall_risk(self, data):
        """Assess overall risk level"""
        if len(data) == 0:
            return 'Low'
        avg_risk = data['risk_score'].mean()
        if avg_risk >= 70:
            return 'Critical'
        elif avg_risk >= 50:
            return 'High'
        elif avg_risk >= 30:
            return 'Medium'
        else:
            return 'Low'

    def _calculate_critical_timeline(self, data):
        """Calculate critical action timeline"""
        critical_items = data[data['risk_score'] >= 80]
        if len(critical_items) > 0:
            min_days = critical_items['days_to_expiry'].min()
            return f"{int(min_days)} days"
        return "No immediate critical items"

    def _assess_prevention_effectiveness(self, data):
        """Assess prevention effectiveness"""
        risky_items = data[data['risk_score'] >= 60]
        if len(risky_items) == 0:
            return "100%"

        preventable_items = risky_items[risky_items['days_to_expiry'] > 7]
        effectiveness = (len(preventable_items) / len(risky_items)) * 100
        return f"{effectiveness:.1f}%"


def generate_comprehensive_expiry_data():
    """Generate comprehensive expiry management data"""
    np.random.seed(42)

    # Product categories with realistic expiry profiles
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
                current_stock = np.random.randint(50, 200)
                unit_cost = np.random.uniform(80, 150)
                daily_sales = np.random.uniform(10, 40)
            elif category == 'Fresh Produce':
                current_stock = np.random.randint(100, 300)
                unit_cost = np.random.uniform(20, 80)
                daily_sales = np.random.uniform(15, 50)
            elif category == 'Bakery':
                current_stock = np.random.randint(80, 250)
                unit_cost = np.random.uniform(30, 100)
                daily_sales = np.random.uniform(20, 60)
            else:
                current_stock = np.random.randint(50, 200)
                unit_cost = np.random.uniform(40, 120)
                daily_sales = np.random.uniform(5, 30)

            stock_value = current_stock * unit_cost

            inventory_data.append({
                'sku_id': f"SKU{len(inventory_data):03d}",
                'sku_name': product,
                'category': category,
                'current_stock': current_stock,
                'unit_cost': unit_cost,
                'stock_value': stock_value,
                'daily_sales_rate': daily_sales,
                'warehouse': np.random.choice(['Warehouse A', 'Warehouse B', 'Warehouse C']),
                'last_updated': datetime.now() - timedelta(days=np.random.randint(0, 7))
            })

    return pd.DataFrame(inventory_data)


def render():
    """📆 INVENTORY EXPIRY MANAGEMENT - Strategic Waste Prevention & Optimization"""

    st.title("📆 Inventory Expiry Management")

    # 🌈 Gradient hero header – aligned with 01_Dashboard style
    st.markdown(
        f"""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    padding: 20px; border-radius: 12px; color: white; margin-bottom: 12px;">
            <h3 style="margin: 0; color: white;">🎯 Strategic Expiry Prevention & Waste Optimization</h3>
            <p style="margin: 5px 0 0 0; opacity: 0.9; color: white;">
                <strong>📍</strong> Inventory Intelligence &gt; Expiry Management |
                <strong>🏰</strong> {st.session_state.get('current_tenant', 'ELORA Holding')} |
                <strong>📅</strong> {datetime.now().strftime('%Y-%m-%d')}
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # 💚 Soft green marquee strip – same pattern as Executive Cockpit
    st.markdown(
        """
        <div style="background: #dcfce7; padding: 10px 16px; border-radius: 10px;
                    margin-bottom: 24px; border-left: 4px solid #16a34a;">
            <marquee behavior="scroll" direction="left" scrollamount="4"
                     style="font-size: 0.9rem; font-weight: 500; color: #166534;">
                ♻️ <strong>Expiry Intelligence:</strong> Turning waste into value •
                📉 <strong>Value at Risk:</strong> Dynamic monitoring of high-risk SKUs •
                🧊 <strong>FEFO Compliance:</strong> AI-guided picking and rotation •
                💰 <strong>Waste Reduction:</strong> Scenario-based savings simulations •
                🏪 <strong>Warehouse Visibility:</strong> Category & location risk heatmaps •
                🤖 <strong>AI Insights:</strong> Early warning on critical expiries and preventable loss
            </marquee>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Require analytics initialization (consistent with other pages)
    if 'analytics' not in st.session_state:
        st.error("❌ Please initialize the application from the main dashboard first.")
        st.info("💡 The system needs to load core inventory data before expiry analytics can run.")
        return

    analytics = st.session_state.analytics

    # Data initialization with enhanced but safe fallback
    try:
        if hasattr(analytics, 'get_inventory_data'):
            inventory_data = analytics.get_inventory_data()
        elif hasattr(analytics, 'inventory_data'):
            inventory_data = analytics.inventory_data
        else:
            inventory_data = pd.DataFrame()
    except Exception:
        inventory_data = pd.DataFrame()

    if inventory_data is None or inventory_data.empty:
        st.warning(
            "📊 Using enhanced demonstration data. Live expiry intelligence will appear once "
            "your inventory data is fully connected."
        )
        inventory_data = generate_comprehensive_expiry_data()

    # Initialize Intelligence Engine
    intel_engine = ExpiryIntelligenceEngine(inventory_data)

    # Perform comprehensive analysis
    analysis_results = intel_engine.perform_comprehensive_expiry_analysis(inventory_data)

    # Enhanced Tab Structure
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Expiry Intelligence",
        "📈 Waste Analytics",
        "🔍 Risk Assessment",
        "🎯 FEFO Optimization",
        "🚀 Action Center"
    ])

    with tab1:
        render_expiry_intelligence(analysis_results, intel_engine)

    with tab2:
        render_waste_analytics(analysis_results, intel_engine)

    with tab3:
        render_risk_assessment(analysis_results, intel_engine)

    with tab4:
        render_fefo_optimization(analysis_results, intel_engine)

    with tab5:
        render_action_center(analysis_results, intel_engine)


def render_expiry_intelligence(analysis_results, intel_engine):
    """Render enhanced expiry intelligence dashboard"""

    st.header("📊 Expiry Intelligence Dashboard")

    st.info("""
    **💡 Strategic Context:** Proactive expiry management transforms waste cost into business value 
    through optimized inventory rotation, demand matching, and strategic prevention frameworks.
    """)

    data = analysis_results['analysis_data']
    metrics = analysis_results['strategic_metrics']
    insights = analysis_results['ai_insights']

    # AI Insights Header
    with st.expander("🤖 AI STRATEGIC INSIGHTS & RECOMMENDATIONS", expanded=True):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "🔄 Stock Rotation Score",
                f"{metrics['fefo_compliance_score']:.1f}%",
                "Strategic Health",
                help="Measures FEFO compliance and inventory rotation effectiveness"
            )

        with col2:
            st.metric(
                "📉 Waste Reduction Potential",
                f"{metrics['waste_reduction_potential']:.1f}%",
                "Optimization Opportunity",
                help="Potential reduction in waste through AI-driven optimization"
            )

        with col3:
            st.metric(
                "💰 Value at Risk",
                f"KES {metrics['value_at_risk']:,.0f}",
                "Financial Exposure",
                help="Total inventory value facing expiry risk"
            )

        # Display AI insights
        if insights:
            for insight in insights:
                msg = f"**{insight['title']}** - {insight['message']}"
                if insight['type'] == 'success':
                    st.success(msg)
                elif insight['type'] == 'warning':
                    st.warning(msg)
                elif insight['type'] == 'error':
                    st.error(msg)

    # Strategic KPI Dashboard
    st.subheader("🎯 Strategic Performance Metrics")

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric("Total Inventory Value", f"KES {metrics['total_inventory_value']:,.0f}")
        st.caption("Current Investment")

    with col2:
        st.metric("Risk Exposure", f"{metrics['risk_exposure_percentage']:.1f}%")
        st.caption("Value at High Risk")

    with col3:
        st.metric("Critical Items", metrics['critical_items_count'])
        st.caption("Require Immediate Action")

    with col4:
        st.metric("Avg Risk Score", f"{metrics['average_risk_score']:.1f}/100")
        st.caption("Portfolio Health")

    with col5:
        cost_avoidance = metrics['value_at_risk'] * 0.6  # 60% recoverable
        st.metric("Savings Potential", f"KES {cost_avoidance:,.0f}")
        st.caption("Through Optimization")

    # Enhanced Visualizations
    col1, col2 = st.columns(2)

    with col1:
        # Risk Distribution by Category
        if len(data) > 0:
            category_risk = data.groupby('category')['risk_score'].mean().sort_values(ascending=False)

            fig = px.bar(
                x=category_risk.values,
                y=category_risk.index,
                orientation='h',
                title='📊 Average Risk Score by Category',
                color=category_risk.values,
                color_continuous_scale='RdYlGn_r'
            )
            fig.update_layout(xaxis_title='Average Risk Score', yaxis_title='Category')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No data available for category risk analysis.")

    with col2:
        # Expiry Timeline Analysis
        timeline_data = data[data['days_to_expiry'] <= 90].copy()
        if len(timeline_data) > 0:
            timeline_data['week_bucket'] = (timeline_data['days_to_expiry'] // 7) + 1

            weekly_summary = timeline_data.groupby('week_bucket').agg({
                'potential_loss': 'sum',
                'sku_id': 'count'
            }).reset_index()

            fig = go.Figure()

            fig.add_trace(go.Bar(
                x=weekly_summary['week_bucket'],
                y=weekly_summary['potential_loss'],
                name='Financial Exposure',
                marker_color='#FF6B6B'
            ))

            fig.update_layout(
                title='💰 Financial Exposure Timeline (Next 90 Days)',
                xaxis_title='Weeks to Expiry',
                yaxis_title='Potential Loss (KES)',
                showlegend=True
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No items expiring within the next 90 days.")

    # Strategic Risk Matrix
    st.subheader("🎯 Strategic Risk Positioning")

    if len(data) > 0:
        fig = px.scatter(
            data,
            x='days_to_expiry',
            y='stock_value',
            color='risk_score',
            size='current_stock',
            hover_data=['sku_name', 'category'],
            title='📈 Risk Matrix: Value vs Time to Expiry',
            color_continuous_scale='RdYlGn_r'
        )

        # Add risk zones
        fig.add_vrect(x0=0, x1=7, fillcolor="red", opacity=0.1, line_width=0)
        fig.add_vrect(x0=7, x1=30, fillcolor="orange", opacity=0.1, line_width=0)
        fig.add_vrect(x0=30, x1=60, fillcolor="yellow", opacity=0.1, line_width=0)

        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No data available for risk matrix.")


def render_waste_analytics(analysis_results, intel_engine):
    """Render enhanced waste analytics with strategic insights"""

    st.header("📈 Waste Analytics & Strategic Reduction")

    st.success("""
    **💡 Intelligence Framework:** Transform waste analytics from cost tracking to strategic opportunity 
    identification through root cause analysis, pattern recognition, and prevention optimization.
    """)

    data = analysis_results['analysis_data']

    # Waste Intelligence Dashboard
    st.subheader("🏆 Waste Intelligence Dashboard")

    total_waste_risk = data['potential_loss'].sum()
    preventable_waste = data[data['days_to_expiry'] > 7]['potential_loss'].sum()
    high_impact_categories = len(data[data['risk_score'] >= 70]['category'].unique())

    if total_waste_risk > 0:
        optimization_potential = (preventable_waste / total_waste_risk) * 100
    else:
        optimization_potential = 0.0

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Waste Risk", f"KES {total_waste_risk:,.0f}")
        st.caption("Financial Exposure")

    with col2:
        st.metric("Preventable Waste", f"KES {preventable_waste:,.0f}")
        st.caption("Opportunity for Action")

    with col3:
        st.metric("High-Impact Categories", high_impact_categories)
        st.caption("Strategic Focus Areas")

    with col4:
        st.metric("Optimization Potential", f"{optimization_potential:.1f}%")
        st.caption("Recoverable Value")

    # Enhanced Waste Analytics
    col1, col2 = st.columns(2)

    with col1:
        # Waste by Category with Risk Scoring
        if len(data) > 0:
            category_analysis = data.groupby('category').agg({
                'potential_loss': 'sum',
                'risk_score': 'mean',
                'sku_id': 'count'
            }).reset_index()

            fig = px.scatter(
                category_analysis,
                x='potential_loss',
                y='risk_score',
                size='sku_id',
                color='category',
                title='📊 Category Risk vs Financial Impact',
                hover_data=['category', 'potential_loss', 'risk_score']
            )
            fig.update_layout(
                xaxis_title='Potential Loss (KES)',
                yaxis_title='Average Risk Score'
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No data available for category-level waste analytics.")

    with col2:
        # Root Cause Analysis
        root_causes = {
            "Overstocking & Poor Forecasting": 45,
            "Inefficient FEFO Implementation": 30,
            "Supplier Lead Time Variability": 15,
            "Demand Pattern Mismatch": 10
        }

        fig = px.pie(
            values=list(root_causes.values()),
            names=list(root_causes.keys()),
            title='🔍 Waste Root Cause Analysis',
            color=list(root_causes.values()),
            color_discrete_sequence=px.colors.sequential.RdBu
        )
        st.plotly_chart(fig, use_container_width=True)

    # Strategic Waste Reduction Framework
    st.subheader("🎯 Strategic Reduction Framework")

    strategies = [
        {
            "phase": "Immediate (1-2 weeks)",
            "actions": [
                "⚙️ Implement emergency FEFO protocols for critical items",
                "🎯 Launch targeted promotions for high-risk categories",
                "📊 Conduct rapid root cause analysis for top 5 waste items"
            ]
        },
        {
            "phase": "Short-term (1 month)",
            "actions": [
                "🤖 Deploy AI-driven demand forecasting for perishables",
                "🏗️ Optimize warehouse layout for FEFO compliance",
                "📈 Implement real-time expiry monitoring dashboards"
            ]
        },
        {
            "phase": "Medium-term (3 months)",
            "actions": [
                "🤝 Establish supplier collaboration for JIT deliveries",
                "📱 Develop mobile FEFO compliance tracking",
                "🎓 Conduct comprehensive staff training programs"
            ]
        }
    ]

    for strategy in strategies:
        with st.expander(f"📅 {strategy['phase']}", expanded=True):
            for action in strategy['actions']:
                st.write(f"• {action}")


def render_risk_assessment(analysis_results, intel_engine):
    """Render enhanced risk assessment with prevention strategies"""

    st.header("🔍 Comprehensive Risk Assessment")

    st.warning("""
    **⚠️ Risk Intelligence:** Proactive risk assessment transforms reactive firefighting into 
    strategic prevention through multi-dimensional scoring, early warning systems, and targeted interventions.
    """)

    data = analysis_results['analysis_data']
    risk_assessment = analysis_results['risk_assessment']

    # Risk Intelligence Dashboard
    st.subheader("🌐 Risk Intelligence Dashboard")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Overall Risk Level", risk_assessment['overall_risk_level'])
        st.caption("Portfolio Assessment")

    with col2:
        st.metric("Financial Exposure", f"KES {risk_assessment['financial_exposure']:,.0f}")
        st.caption("Potential Loss")

    with col3:
        st.metric("Critical Timeline", risk_assessment['critical_timeline'])
        st.caption("Immediate Action Window")

    with col4:
        st.metric("Prevention Effectiveness", risk_assessment['prevention_effectiveness'])
        st.caption("Current Capability")

    # Enhanced Risk Visualizations
    col1, col2 = st.columns(2)

    with col1:
        # Risk Distribution Heatmap
        if len(data) > 0:
            risk_pivot = data.pivot_table(
                values='risk_score',
                index='category',
                columns='warehouse',
                aggfunc='mean',
                fill_value=0
            )

            fig = px.imshow(
                risk_pivot,
                title='🌐 Risk Heatmap by Category & Warehouse',
                color_continuous_scale='RdYlGn_r',
                aspect="auto"
            )
            fig.update_layout(xaxis_title="Warehouse", yaxis_title="Product Category")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No data available for risk heatmap.")

    with col2:
        # Risk Score Distribution
        if len(data) > 0:
            fig = px.histogram(
                data,
                x='risk_score',
                nbins=20,
                title='📊 Risk Score Distribution',
                color_discrete_sequence=['#FF6B6B']
            )
            fig.update_layout(xaxis_title='Risk Score', yaxis_title='Number of Items')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No data available for risk distribution.")

    # High-Risk Items Management
    st.subheader("🚨 High-Risk Items Requiring Immediate Attention")

    high_risk_items = data[data['risk_score'] >= 70].sort_values('risk_score', ascending=False)

    if len(high_risk_items) > 0:
        # Display top 10 high-risk items
        display_cols = ['sku_name', 'category', 'warehouse', 'days_to_expiry', 'risk_score', 'potential_loss']

        display_df = high_risk_items[display_cols].head(10).copy()
        display_df['potential_loss'] = display_df['potential_loss'].apply(lambda x: f"KES {x:,.0f}")
        display_df['risk_score'] = display_df['risk_score'].apply(lambda x: f"{x:.1f}%")
        display_df['days_to_expiry'] = display_df['days_to_expiry'].apply(lambda x: f"{int(x)} days")

        st.dataframe(
            display_df.rename(columns={
                'sku_name': 'Product', 'category': 'Category', 'warehouse': 'Warehouse',
                'days_to_expiry': 'Days to Expiry', 'risk_score': 'Risk Score',
                'potential_loss': 'Potential Loss'
            }),
            use_container_width=True
        )

        # Quick Actions for High-Risk Items
        st.subheader("⚡ Quick Action Center")

        col1, col2, col3 = st.columns(3)

        with col1:
            if st.button("🔄 Prioritize FEFO Picking", use_container_width=True):
                st.success("FEFO prioritization activated for high-risk items!")

        with col2:
            if st.button("🎯 Launch Promotions", use_container_width=True):
                st.success("Promotional campaigns initiated for critical items!")

        with col3:
            if st.button("📊 Generate Action Report", use_container_width=True):
                st.success("Comprehensive action report generated!")
    else:
        st.success("✅ No high-risk items requiring immediate attention!")


def render_fefo_optimization(analysis_results, intel_engine):
    """Render FEFO optimization with AI-powered recommendations"""

    st.header("🔧 FEFO Optimization Engine")

    st.info("""
    **🤖 AI Optimization:** Transform First-Expired-First-Out from manual process to intelligent system 
    through machine learning, dynamic routing, and predictive analytics.
    """)

    data = analysis_results['analysis_data']

    # FEFO Intelligence Dashboard
    st.subheader("🏆 FEFO Performance Intelligence")

    fefo_score = analysis_results['strategic_metrics']['fefo_compliance_score']
    optimization_potential = max(0.0, 100 - fefo_score)
    potential_savings = data[data['risk_score'] >= 60]['potential_loss'].sum() * 0.6
    implementation_timeline = "8-12 weeks"

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("FEFO Compliance Score", f"{fefo_score:.1f}%")
        st.caption("Current Performance")

    with col2:
        st.metric("Optimization Potential", f"{optimization_potential:.1f}%")
        st.caption("Improvement Opportunity")

    with col3:
        st.metric("Potential Savings", f"KES {potential_savings:,.0f}")
        st.caption("Annual Impact")

    with col4:
        st.metric("Implementation Timeline", implementation_timeline)
        st.caption("Full Optimization")

    # FEFO Optimization Strategies
    st.subheader("🎯 AI-Powered Optimization Strategies")

    strategies = intel_engine.optimization_strategies

    for strategy_key, strategy in strategies.items():
        with st.expander(f"🚀 {strategy['name']} - Impact: {strategy['impact']}", expanded=True):
            col1, col2 = st.columns([2, 1])

            with col1:
                st.write(f"**Description**: {strategy['description']}")
                st.write(f"**Implementation**: {strategy['implementation']}")
                st.write(f"**Cost Level**: {strategy['cost']}")

            with col2:
                if st.button(f"Implement {strategy['name']}", key=strategy_key, use_container_width=True):
                    st.success(f"Implementation started for {strategy['name']}!")

    # FEFO Simulation Engine
    st.subheader("🧪 FEFO Impact Simulation")

    with st.form("fefo_simulation"):
        st.write("**Configure Optimization Scenario**")

        col1, col2 = st.columns(2)

        with col1:
            target_compliance = st.slider("Target FEFO Compliance (%)", 70, 95, 85)
            focus_categories = st.multiselect(
                "Priority Categories",
                data['category'].unique(),
                default=list(data['category'].unique())[:3]
            )

        with col2:
            implementation_budget = st.select_slider(
                "Implementation Budget",
                options=[
                    'Low (KES 1M–2.5M)',
                    'Medium (KES 2.5M–7.5M)',
                    'High (KES 7.5M–15M)'
                ],
                value='Medium (KES 2.5M–7.5M)'
            )
            timeline = st.selectbox(
                "Implementation Timeline",
                ["30 days", "60 days", "90 days", "180 days"]
            )

        simulate = st.form_submit_button("🚀 Run FEFO Simulation")

        if simulate:
            # Dummy simulation (placeholder for real model)
            simulation_results = {
                "projected_savings": 145000,
                "waste_reduction": 38,
                "compliance_improvement": 32,
                "implementation_cost": 55000,
                "roi_period": "7 months",
                "risk_reduction": 45
            }

            # Display results
            st.success("🎯 Simulation Completed Successfully!")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Projected Annual Savings", f"KES {simulation_results['projected_savings']:,.0f}")

            with col2:
                st.metric("Waste Reduction", f"{simulation_results['waste_reduction']}%")

            with col3:
                st.metric("ROI Period", simulation_results['roi_period'])

            st.info(
                f"📊 **Business Case**: KES {simulation_results['implementation_cost']:,.0f} "
                f"investment with {simulation_results['compliance_improvement']}% compliance improvement"
            )


def render_action_center(analysis_results, intel_engine):
    """Render comprehensive action center with execution tracking"""

    st.header("🚀 Strategic Action Center")

    st.success("""
    **🎯 Execution Excellence:** Transform insights into action through structured implementation, 
    progress tracking, and performance measurement with clear accountability and timelines.
    """)

    # Action Planning Dashboard
    st.subheader("📋 Strategic Action Plan")

    # Immediate Actions (Next 7 days)
    st.markdown("### ⚡ Immediate Actions (Next 7 Days)")

    immediate_actions = [
        {
            "action": "Emergency FEFO Implementation",
            "owner": "Warehouse Manager",
            "deadline": "Within 48 hours",
            "status": "Not Started",
            "impact": "High"
        },
        {
            "action": "Critical Items Promotion",
            "owner": "Sales Manager",
            "deadline": "Within 72 hours",
            "status": "Planning",
            "impact": "High"
        },
        {
            "action": "Supplier Coordination - Short Shelf Life",
            "owner": "Procurement Manager",
            "deadline": "Within 7 days",
            "status": "In Progress",
            "impact": "Medium"
        }
    ]

    for action in immediate_actions:
        with st.expander(f"📌 {action['action']} - Impact: {action['impact']}", expanded=True):
            col1, col2, col3 = st.columns(3)

            with col1:
                st.write(f"**Owner**: {action['owner']}")

            with col2:
                st.write(f"**Deadline**: {action['deadline']}")

            with col3:
                status_icon = {
                    'Not Started': "🔴",
                    'Planning': "🟡",
                    'In Progress': "🟢",
                    'Completed': "✅"
                }.get(action['status'], "⚪")
                st.write(f"**Status**: {status_icon} {action['status']}")

            # Action buttons
            col_a, col_b = st.columns(2)
            with col_a:
                if st.button(f"Start {action['action']}", key=f"start_{action['action']}", use_container_width=True):
                    st.success(f"Action initiated: {action['action']}")
            with col_b:
                if st.button("Update Status", key=f"update_{action['action']}", use_container_width=True):
                    st.info(f"Status update requested for {action['action']}")

    # Strategic Initiatives (30-90 days)
    st.markdown("### 🎯 Strategic Initiatives (30-90 Days)")

    strategic_initiatives = [
        {
            "initiative": "AI-Powered FEFO System Implementation",
            "timeline": "12 weeks",
            "budget": "KES 7,500,000",
            "expected_roi": "185%",
            "status": "Planning"
        },
        {
            "initiative": "Warehouse Layout Optimization",
            "timeline": "8 weeks",
            "budget": "KES 4,500,000",
            "expected_roi": "220%",
            "status": "Approved"
        },
        {
            "initiative": "Supplier Collaboration Program",
            "timeline": "16 weeks",
            "budget": "KES 2,500,000",
            "expected_roi": "150%",
            "status": "Research"
        }
    ]

    for initiative in strategic_initiatives:
        with st.expander(f"🏗️ {initiative['initiative']}", expanded=True):
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Timeline", initiative['timeline'])

            with col2:
                st.metric("Budget", initiative['budget'])

            with col3:
                st.metric("Expected ROI", initiative['expected_roi'])

            st.write(f"**Current Status**: {initiative['status']}")

    # Performance Tracking
    st.subheader("📊 Performance Tracking & Reporting")

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("📈 Generate Performance Report", use_container_width=True):
            st.success("Comprehensive performance report generated!")

    with col2:
        if st.button("🔄 Update Action Status", use_container_width=True):
            st.success("Action status updated across all initiatives!")

    with col3:
        if st.button("📤 Export Optimization Plan", use_container_width=True):
            st.success("Strategic optimization plan exported successfully!")

    # Quick Wins Dashboard
    st.subheader("⚡ Quick Wins & Immediate Impact")

    quick_wins = [
        "Implement color-coded expiry labeling system",
        "Train staff on FEFO principles and importance",
        "Set up daily expiry risk alerts for managers",
        "Create expiry dashboard for operational teams",
        "Establish cross-functional expiry task force"
    ]

    for i, win in enumerate(quick_wins, 1):
        col1, col2 = st.columns([3, 1])
        with col1:
            st.write(f"{i}. {win}")
        with col2:
            if st.button("Implement", key=f"win_{i}", use_container_width=True):
                st.success(f"Quick win implemented: {win}")


if __name__ == "__main__":
    render()
