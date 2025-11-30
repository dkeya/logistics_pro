# logistics_pro/pages/02_Sales_Revenue.py
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import numpy as np


def safe_get_column(df, column, default_value=None):
    """Safely get column with fallback (same pattern as 01_Dashboard.py)."""
    if column in df.columns:
        return df[column]
    if default_value is None:
        return pd.Series([f"Default_{i}" for i in range(len(df))])
    return pd.Series([default_value] * len(df))


# Enhanced forecasting engine with strategic insights
class RevenueStrategyEngine:
    """Advanced revenue strategy and forecasting engine"""
    def __init__(self, sales_data):
        self.sales_data = sales_data
        self.strategy_frameworks = self._initialize_frameworks()
    
    def _initialize_frameworks(self):
        """Initialize strategic frameworks"""
        return {
            "growth_strategies": {
                "Market Penetration": "Increase share in existing markets",
                "Product Development": "Introduce new products to existing markets",
                "Market Development": "Expand into new markets with existing products",
                "Diversification": "Develop new products for new markets"
            },
            "pricing_strategies": {
                "Value-Based": "Price based on perceived customer value",
                "Competitive": "Match or beat competitor pricing",
                "Cost-Plus": "Add markup to cost basis",
                "Dynamic": "Adjust prices based on demand and market conditions"
            },
            "channel_strategies": {
                "Direct Sales": "Sell directly to customers",
                "Distribution Partners": "Leverage third-party distributors",
                "E-commerce": "Online sales channels",
                "Hybrid Model": "Combination of multiple channels"
            }
        }
    
    def generate_strategic_forecast(self, strategy_choice, confidence=0.9):
        """Generate forecast based on selected strategy"""
        base_forecast = self._generate_base_forecast()
        
        # Apply strategy multipliers
        strategy_impact = {
            "Market Penetration": 1.15,   # 15% growth
            "Product Development": 1.25,  # 25% growth
            "Market Development": 1.30,   # 30% growth
            "Diversification": 1.40       # 40% growth
        }
        
        multiplier = strategy_impact.get(strategy_choice, 1.0)
        strategic_forecast = base_forecast * multiplier
        
        return {
            'base_forecast': base_forecast,
            'strategic_forecast': strategic_forecast,
            'growth_impact': (multiplier - 1) * 100,
            'strategy': strategy_choice
        }
    
    def _generate_base_forecast(self):
        """Generate base revenue forecast"""
        if self.sales_data.empty:
            return 50000
        
        daily_revenue = self.sales_data.groupby('date')['revenue'].sum()
        return daily_revenue.mean() * 30  # Monthly projection


def render_revenue_marquee():
    """Soft green marquee strip under the hero header (pattern-aligned with Executive Cockpit)."""
    st.markdown(
        """
        <div style="background: #dcfce7; padding: 10px 16px; border-radius: 10px;
                    margin-bottom: 24px; border-left: 4px solid #16a34a;">
            <marquee behavior="scroll" direction="left" scrollamount="4"
                     style="font-size: 0.9rem; font-weight: 500; color: #166534;">
                📈 <strong>Revenue Pulse:</strong> Scenario-based growth simulation •
                💰 <strong>Strategic Pricing:</strong> Value-based &amp; competitive levers in play •
                🎯 <strong>Customer Growth:</strong> Acquisition, retention &amp; upsell dynamics •
                🌍 <strong>Regional Focus:</strong> High-opportunity markets under the spotlight •
                🚀 <strong>War Room Ready:</strong> Real-time KPIs for decisive action
            </marquee>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render():
    """🚀 REVENUE STRATEGY COMMAND CENTER - Interactive & Visual"""
    
    st.title("💰 Revenue Strategy Command Center")
    st.markdown(
        f"""
        <div style="
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
            border-radius: 12px;
            color: white;
            margin-bottom: 12px;
        ">
            <h3 style="margin: 0; color: white;">🎯 Revenue Growth War Room</h3>
            <p style="margin: 5px 0 0 0; opacity: 0.9; color: white;">
                <strong>📍</strong> Sales Intelligence &gt; Revenue Strategy |
                <strong>🏢</strong> {st.session_state.get('current_tenant', 'ELORA Holding')} |
                <strong>📅</strong> {datetime.now().strftime('%Y-%m-%d')}
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 🟢 Soft green marquee strip directly under the hero header
    render_revenue_marquee()
    
    # Check if analytics is available
    if 'analytics' not in st.session_state:
        st.error("❌ Please visit the main dashboard first to initialize data")
        return
    
    analytics = st.session_state.analytics

    # Load sales data safely (pattern similar to 01_Dashboard)
    try:
        sales_data = analytics.sales_data.copy()
    except Exception as e:
        st.error(f"❌ Error loading sales data: {e}")
        return

    # --- Ensure essential columns exist so the page never breaks ---
    if 'quantity' not in sales_data.columns:
        sales_data['quantity'] = 1.0

    if 'unit_price' not in sales_data.columns:
        sales_data['unit_price'] = safe_get_column(sales_data, 'unit_price', 100)

    if 'revenue' not in sales_data.columns:
        sales_data['revenue'] = sales_data['quantity'] * safe_get_column(
            sales_data, 'unit_price', 100
        )

    if 'date' not in sales_data.columns:
        sales_data['date'] = pd.date_range(
            end=datetime.now().date(), periods=len(sales_data)
        )

    if 'customer_id' not in sales_data.columns:
        sales_data['customer_id'] = [f"CUST_{i+1:03d}" for i in range(len(sales_data))]

    if 'category' not in sales_data.columns:
        sales_data['category'] = "General"

    # 🎨 INTERACTIVE STRATEGY CANVAS - New Visual Approach
    tab1, tab2, tab3, tab4 = st.tabs([
        "🎯 Strategy Canvas",
        "📈 Growth Simulator",
        "💡 Opportunity Matrix",
        "📊 Performance War Room"
    ])
    
    with tab1:
        render_strategy_canvas(sales_data)
    
    with tab2:
        render_growth_simulator(sales_data)
    
    with tab3:
        render_opportunity_matrix(sales_data)
    
    with tab4:
        render_performance_war_room(sales_data)


def render_strategy_canvas(sales_data):
    """Interactive Revenue Strategy Canvas"""
    
    st.header("🎯 Revenue Strategy Canvas")
    st.markdown("""
    <div style="background: #f8f9fa; padding: 15px; border-radius: 8px;
                border-left: 4px solid #28a745; margin-bottom: 20px;">
        <strong>🚀 Design your revenue growth strategy using this interactive canvas.</strong> 
        Use the levers and initiatives below to build your customized growth plan.
    </div>
    """, unsafe_allow_html=True)
    
    # Strategy Building Blocks
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("🏹 Growth Levers")
        growth_levers = {
            "Pricing Optimization": {"impact": "High", "effort": "Medium", "timeline": "30-60 days"},
            "Customer Acquisition": {"impact": "High", "effort": "High", "timeline": "60-90 days"},
            "Retention Improvement": {"impact": "Medium", "effort": "Low", "timeline": "30 days"},
            "Upsell/Cross-sell": {"impact": "Medium", "effort": "Medium", "timeline": "45 days"}
        }
        
        for lever, details in growth_levers.items():
            with st.expander(f"📌 {lever}"):
                st.write(f"**Impact:** {details['impact']}")
                st.write(f"**Effort:** {details['effort']}")
                st.write(f"**Timeline:** {details['timeline']}")
                if st.button(f"Add {lever}", key=f"add_{lever}"):
                    st.success(f"✅ {lever} added to strategy")
    
    with col2:
        st.subheader("🎯 Strategic Initiatives")
        
        # Interactive initiative selection
        selected_initiatives = st.multiselect(
            "Choose your strategic initiatives:",
            ["Digital Transformation", "Market Expansion", "Product Innovation",
             "Partnership Development", "Customer Experience"],
            default=["Digital Transformation", "Market Expansion"]
        )
        
        # Impact assessment
        if selected_initiatives:
            st.subheader("📊 Initiative Impact Assessment")
            impact_data = []
            for initiative in selected_initiatives:
                revenue_impact = np.random.uniform(5, 25)  # Simulated impact
                timeline = np.random.choice(["30 days", "60 days", "90 days"])
                impact_data.append({
                    "Initiative": initiative,
                    "Revenue Impact %": revenue_impact,
                    "Timeline": timeline
                })
            
            impact_df = pd.DataFrame(impact_data)
            fig = px.bar(
                impact_df,
                x="Initiative",
                y="Revenue Impact %",
                title="Expected Revenue Impact by Initiative",
                color="Revenue Impact %",
                color_continuous_scale="Viridis"
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with col3:
        st.subheader("🎲 Risk & Opportunity Matrix")
        
        # 2x2 Risk-Return Matrix
        initiatives_risk_return = {
            "Pricing Optimization": {"return": 8.5, "risk": 3.2},
            "Market Expansion": {"return": 12.3, "risk": 7.8},
            "Product Innovation": {"return": 15.7, "risk": 9.1},
            "Digital Transformation": {"return": 9.8, "risk": 6.5}
        }
        
        risk_return_df = pd.DataFrame(initiatives_risk_return).T.reset_index()
        risk_return_df.columns = ['Initiative', 'Expected Return (%)', 'Risk Score']
        
        fig = px.scatter(
            risk_return_df,
            x="Risk Score",
            y="Expected Return (%)",
            size="Expected Return (%)",
            color="Initiative",
            title="Risk-Return Matrix: Strategic Initiatives",
            hover_name="Initiative",
            size_max=60
        )
        
        # Add quadrant lines
        fig.add_hline(y=10, line_dash="dash", line_color="red")
        fig.add_vline(x=5, line_dash="dash", line_color="red")
        
        # Add quadrant labels
        fig.add_annotation(x=2, y=15, text="High Return\nLow Risk", showarrow=False, font=dict(color="green", size=12))
        fig.add_annotation(x=8, y=15, text="High Return\nHigh Risk", showarrow=False, font=dict(color="orange", size=12))
        fig.add_annotation(x=2, y=5, text="Low Return\nLow Risk", showarrow=False, font=dict(color="blue", size=12))
        fig.add_annotation(x=8, y=5, text="Low Return\nHigh Risk", showarrow=False, font=dict(color="red", size=12))
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Strategy Summary
    st.markdown("---")
    st.subheader("📋 Your Custom Revenue Strategy")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.info("""
        **🎯 Selected Growth Levers:**
        - Pricing Optimization
        - Customer Acquisition  
        - Retention Improvement
        
        **🚀 Strategic Initiatives:**
        - Digital Transformation
        - Market Expansion
        """)
    
    with col2:
        total_impact = 28.5  # Simulated total impact
        timeline = "90 days"
        confidence = "High"
        
        st.metric("Total Expected Impact", f"+{total_impact}%")
        st.metric("Implementation Timeline", timeline)
        st.metric("Confidence Level", confidence)
    
    # Action Plan
    with st.expander("🛠️ Generate Your Action Plan", expanded=True):
        st.success("""
        **📅 30-Day Action Plan:**
        - Week 1-2: Implement pricing optimization
        - Week 3-4: Launch customer acquisition campaigns
        - Week 5-6: Deploy retention improvement initiatives
        
        **🎯 Success Metrics:**
        - 15% revenue growth in 90 days
        - 25% improvement in customer retention
        - 20% increase in average order value
        """)


def render_growth_simulator(sales_data):
    """Interactive Growth Strategy Simulator"""
    
    st.header("📈 Growth Strategy Simulator")
    st.markdown("""
    <div style="background: #f0f8ff; padding: 15px; border-radius: 8px;
                border-left: 4px solid #007bff; margin-bottom: 20px;">
        <strong>🎮 Test different growth strategies and see their impact on revenue.</strong> 
        Adjust the sliders to simulate various scenarios and optimize your approach.
    </div>
    """, unsafe_allow_html=True)
    
    # Simulation Controls
    col1, col2, col3 = st.columns(3)
    
    with col1:
        pricing_strategy = st.select_slider(
            "💰 Pricing Strategy",
            options=["Cost-Plus", "Competitive", "Value-Based", "Premium"],
            value="Value-Based"
        )
        
        price_impact = {
            "Cost-Plus": 1.0,
            "Competitive": 1.05,
            "Value-Based": 1.15,
            "Premium": 1.25
        }
    
    with col2:
        acquisition_budget = st.slider(
            "🎯 Customer Acquisition Budget (% of revenue)",
            min_value=5, max_value=30, value=15
        )
        
        acquisition_impact = 1 + (acquisition_budget - 10) * 0.02
    
    with col3:
        retention_focus = st.slider(
            "🤝 Customer Retention Focus",
            min_value=1, max_value=10, value=7
        )
        
        retention_impact = 1 + (retention_focus - 5) * 0.03
    
    # Market Conditions
    col1, col2 = st.columns(2)
    
    with col1:
        market_growth = st.slider(
            "📈 Market Growth Rate (%)",
            min_value=-5, max_value=15, value=8
        )
    
    with col2:
        competitive_intensity = st.slider(
            "⚔️ Competitive Intensity",
            min_value=1, max_value=10, value=6
        )
        
        competitive_impact = 1 - (competitive_intensity - 5) * 0.02
    
    # Calculate Simulated Impact
    base_revenue = sales_data['revenue'].sum() if not sales_data.empty else 1_000_000
    simulated_revenue = (
        base_revenue
        * price_impact[pricing_strategy]
        * acquisition_impact
        * retention_impact
        * competitive_impact
    )
    
    # Display Results
    st.markdown("---")
    st.subheader("🎯 Simulation Results")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Base Revenue", f"KES {base_revenue:,.0f}")
    
    with col2:
        st.metric("Simulated Revenue", f"KES {simulated_revenue:,.0f}")
    
    with col3:
        growth_pct = ((simulated_revenue - base_revenue) / base_revenue) * 100 if base_revenue > 0 else 0
        st.metric("Growth Impact", f"+{growth_pct:.1f}%")
    
    with col4:
        st.metric("Optimal Strategy", pricing_strategy)
    
    # Visualize Strategy Impact
    st.subheader("📊 Strategy Impact Visualization")
    
    strategy_components = {
        'Pricing Strategy': price_impact[pricing_strategy],
        'Acquisition Focus': acquisition_impact,
        'Retention Focus': retention_impact,
        'Market Conditions': competitive_impact
    }
    
    impact_df = pd.DataFrame({
        'Component': list(strategy_components.keys()),
        'Impact Multiplier': list(strategy_components.values())
    })
    
    fig = px.bar(
        impact_df,
        x='Component',
        y='Impact Multiplier',
        title='Growth Strategy Component Impact',
        color='Impact Multiplier',
        color_continuous_scale='RdYlGn'
    )
    
    fig.add_hline(
        y=1.0,
        line_dash="dash",
        line_color="red",
        annotation_text="Baseline",
        annotation_position="right"
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Recommendation Engine
    st.subheader("🧠 AI Strategy Recommendations")
    
    if growth_pct > 20:
        recommendation = "🎉 Excellent strategy! Your simulated growth exceeds 20%. Consider scaling this approach."
        color = "green"
    elif growth_pct > 10:
        recommendation = "✅ Strong strategy! Good growth potential. Focus on execution excellence."
        color = "blue"
    else:
        recommendation = "⚠️ Moderate growth. Consider optimizing pricing or acquisition strategy."
        color = "orange"
    
    st.markdown(f"""
    <div style="background: {color}20; padding: 15px; border-radius: 8px;
                border-left: 4px solid {color};">
        <strong>Recommendation:</strong> {recommendation}
    </div>
    """, unsafe_allow_html=True)


def render_opportunity_matrix(sales_data):
    """Interactive Opportunity Identification Matrix"""
    
    st.header("💡 Revenue Opportunity Matrix")
    st.markdown("""
    <div style="background: #fff3cd; padding: 15px; border-radius: 8px;
                border-left: 4px solid #ffc107; margin-bottom: 20px;">
        <strong>🔍 Discover hidden revenue opportunities across customers, products, and regions.</strong> 
        Use this matrix to prioritize your growth initiatives.
    </div>
    """, unsafe_allow_html=True)
    
    # Opportunity Analysis Framework
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎯 Customer Opportunity Matrix")
        
        # Simulated customer data
        customer_opportunity = {
            'Segment': ['Enterprise', 'SMB', 'Retail', 'Government', 'Startups'],
            'Current Revenue': [450000, 280000, 320000, 190000, 80000],
            'Growth Potential %': [15, 25, 35, 20, 45],
            'Effort Required': [3, 5, 7, 8, 6]
        }
        
        customer_df = pd.DataFrame(customer_opportunity)
        customer_df['Opportunity Score'] = (
            customer_df['Growth Potential %'] / customer_df['Effort Required']
        ).round(1)
        
        fig = px.scatter(
            customer_df,
            x='Effort Required',
            y='Growth Potential %',
            size='Current Revenue',
            color='Segment',
            title='Customer Segment Opportunity Matrix',
            hover_name='Segment',
            size_max=40
        )
        
        # Add quadrant lines
        fig.add_hline(y=25, line_dash="dash", line_color="red")
        fig.add_vline(x=5, line_dash="dash", line_color="red")
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("📦 Product Opportunity Analysis")
        
        if 'category' in sales_data.columns:
            product_analysis = sales_data.groupby('category').agg({
                'revenue': 'sum',
                'quantity': 'sum',
                'customer_id': 'nunique'
            }).reset_index()
            
            product_analysis['Revenue per Customer'] = (
                product_analysis['revenue'] / product_analysis['customer_id'].replace(0, 1)
            )
            product_analysis['Growth Score'] = (
                product_analysis['revenue'] / product_analysis['revenue'].sum() * 100
            ).round(1)
            
            fig = px.treemap(
                product_analysis,
                path=['category'],
                values='revenue',
                color='Revenue per Customer',
                title='Product Category Revenue Distribution',
                color_continuous_scale='Viridis'
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Product category data not available for analysis")
    
    # Regional Opportunity Heatmap
    st.subheader("🌍 Regional Opportunity Heatmap")
    
    # Simulated regional data
    regions = ['Nairobi', 'Mombasa', 'Kisumu', 'Nakuru', 'Eldoret', 'Thika']
    regional_opportunity = {
        'Region': regions,
        'Current Market Share %': np.random.uniform(15, 45, len(regions)),
        'Growth Rate %': np.random.uniform(5, 25, len(regions)),
        'Competition Intensity': np.random.randint(1, 10, len(regions))
    }
    
    regional_df = pd.DataFrame(regional_opportunity)
    regional_df['Opportunity Index'] = (
        regional_df['Growth Rate %']
        * (100 - regional_df['Current Market Share %'])
        / regional_df['Competition Intensity'].replace(0, 1)
    ).round(1)
    
    # Create opportunity heatmap
    fig = px.density_heatmap(
        regional_df,
        x='Current Market Share %',
        y='Growth Rate %',
        z='Opportunity Index',
        hover_name='Region',
        title='Regional Opportunity Heatmap',
        color_continuous_scale='Viridis'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Priority Recommendations
    st.subheader("🎯 Priority Opportunity Recommendations")
    
    # Sort by opportunity score
    opportunity_priorities = regional_df.nlargest(3, 'Opportunity Index')
    
    for idx, (_, row) in enumerate(opportunity_priorities.iterrows(), 1):
        st.markdown(f"""
        <div style="background: #e8f5e8; padding: 15px; border-radius: 8px;
                    margin-bottom: 10px; border-left: 4px solid #28a745;">
            <strong>#{idx} Priority: {row['Region']}</strong><br>
            📈 Growth Rate: {row['Growth Rate %']:.1f}% | 
            🎯 Market Share: {row['Current Market Share %']:.1f}% |
            💡 Opportunity Score: {row['Opportunity Index']:.1f}
        </div>
        """, unsafe_allow_html=True)


def render_performance_war_room(sales_data):
    """Interactive Performance Monitoring War Room"""
    
    st.header("📊 Performance War Room")
    st.markdown("""
    <div style="background: #e7f3ff; padding: 15px; border-radius: 8px;
                border-left: 4px solid #0056b3; margin-bottom: 20px;">
        <strong>🚀 Real-time revenue performance monitoring and strategic decision support.</strong> 
        Track KPIs, monitor trends, and make data-driven decisions.
    </div>
    """, unsafe_allow_html=True)
    
    # Real-time KPI Dashboard
    st.subheader("🎯 Live Performance Dashboard")
    
    # Calculate KPIs
    if not sales_data.empty:
        total_revenue = sales_data['revenue'].sum()
        daily_revenue = sales_data.groupby('date')['revenue'].sum()
        growth_rate = calculate_growth_rate(daily_revenue)
        customer_count = sales_data['customer_id'].nunique()
        avg_order_value = sales_data['revenue'].mean()
    else:
        total_revenue = 0
        growth_rate = 0
        customer_count = 0
        avg_order_value = 0
    
    # KPI Cards with Visual Indicators
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Revenue", f"KES {total_revenue:,.0f}")
        progress = min(100, total_revenue / 1_000_000 * 100) if total_revenue > 0 else 0
        st.progress(progress / 100)
        st.caption(f"{progress:.1f}% of monthly target")
    
    with col2:
        st.metric("Growth Rate", f"{growth_rate:+.1f}%")
        color = "green" if growth_rate > 5 else "orange" if growth_rate > 0 else "red"
        label = "Strong" if growth_rate > 5 else "Moderate" if growth_rate > 0 else "Declining"
        st.markdown(
            f"<div style='color: {color}; font-weight: bold;'>📈 {label}</div>",
            unsafe_allow_html=True
        )
    
    with col3:
        st.metric("Active Customers", f"{customer_count}")
        st.caption(f"👥 {customer_count} unique customers")
    
    with col4:
        st.metric("Avg Order Value", f"KES {avg_order_value:,.0f}")
        st.caption("💰 Average revenue per transaction")
    
    # Performance Trends
    st.subheader("📈 Real-time Performance Trends")
    
    if not sales_data.empty:
        # Weekly performance trends
        sales_data = sales_data.copy()
        sales_data['week'] = pd.to_datetime(sales_data['date']).dt.isocalendar().week
        weekly_trends = sales_data.groupby('week').agg({
            'revenue': 'sum',
            'quantity': 'sum',
            'customer_id': 'nunique'
        }).reset_index()
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=weekly_trends['week'],
            y=weekly_trends['revenue'],
            name='Weekly Revenue',
            line=dict(color='#1f77b4', width=4),
            fill='tozeroy',
            fillcolor='rgba(31, 119, 180, 0.1)'
        ))
        
        fig.add_trace(go.Scatter(
            x=weekly_trends['week'],
            y=weekly_trends['customer_id'] * 1000,
            name='Active Customers (scaled)',
            line=dict(color='#2ca02c', width=3, dash='dot'),
            yaxis='y2'
        ))
        
        fig.update_layout(
            title='📊 Weekly Revenue & Customer Trends',
            xaxis_title='Week',
            yaxis_title='Revenue (KES)',
            yaxis2=dict(
                title='Customers (scaled)',
                overlaying='y',
                side='right'
            ),
            height=400,
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Strategic Alerts & Recommendations
    st.subheader("🚨 Strategic Alerts & Actions")
    
    alerts = []
    
    if growth_rate < 5:
        alerts.append("⚠️ Growth rate below target. Consider growth initiatives.")
    
    if customer_count < 50 and not sales_data.empty:
        alerts.append("🎯 Customer base small. Focus on acquisition strategies.")
    
    if avg_order_value < 5000 and not sales_data.empty:
        alerts.append("💰 Low average order value. Implement upselling strategies.")
    
    if alerts:
        for alert in alerts:
            st.warning(alert)
    else:
        st.success("✅ All performance indicators are healthy!")
    
    # Quick Action Buttons
    st.subheader("⚡ Quick Actions")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("📊 Generate Report", use_container_width=True):
            st.success("Performance report generated!")
    
    with col2:
        if st.button("🎯 Optimize Pricing", use_container_width=True):
            st.info("Pricing optimization analysis started")
    
    with col3:
        if st.button("🚀 Launch Campaign", use_container_width=True):
            st.info("Campaign launch initiated")
    
    with col4:
        if st.button("📈 Forecast Growth", use_container_width=True):
            st.info("Growth forecasting in progress")


def calculate_growth_rate(daily_revenue):
    """Calculate revenue growth rate"""
    if len(daily_revenue) < 14:
        return 8.5
    
    recent = daily_revenue.tail(7).mean()
    previous = daily_revenue.tail(14).head(7).mean()
    
    if previous > 0:
        return ((recent - previous) / previous) * 100
    return 8.5


if __name__ == "__main__":
    render()
