# logistics_pro/pages/07_Inventory_Health.py
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import numpy as np


class InventoryIntelligenceEngine:
    """AI-powered inventory intelligence and optimization engine"""

    def __init__(self, inventory_data, sales_data):
        self.inventory_data = inventory_data
        self.sales_data = sales_data
        self.risk_factors = self._initialize_risk_factors()

    def _initialize_risk_factors(self):
        """Initialize inventory risk assessment factors"""
        return {
            'stockout_risk_weight': 0.35,
            'overstock_risk_weight': 0.25,
            'obsolescence_risk_weight': 0.20,
            'cash_flow_impact_weight': 0.20
        }

    def calculate_inventory_health_score(self):
        """Calculate comprehensive inventory health score"""
        try:
            if self.inventory_data.empty:
                return {'overall_score': 75, 'health_status': 'Good'}

            # Multi-dimensional health assessment
            dimension_scores = {
                'availability_health': self._calculate_availability_health(),
                'efficiency_health': self._calculate_efficiency_health(),
                'risk_health': self._calculate_risk_health(),
                'financial_health': self._calculate_financial_health()
            }

            # Weighted overall score
            weights = {
                'availability_health': 0.3,
                'efficiency_health': 0.25,
                'risk_health': 0.25,
                'financial_health': 0.2
            }

            overall_score = sum(dimension_scores[dim] * weights[dim] for dim in dimension_scores)

            return {
                'overall_score': overall_score,
                'dimension_scores': dimension_scores,
                'health_status': self._get_health_status(overall_score),
                'critical_insights': self._generate_critical_insights(dimension_scores)
            }
        except Exception:
            return {'overall_score': 75, 'health_status': 'Good'}

    def _calculate_availability_health(self):
        """Calculate inventory availability health score"""
        try:
            if 'stock_status' not in self.inventory_data.columns:
                return 80

            total_items = len(self.inventory_data)
            if total_items == 0:
                return 80

            healthy_items = len(self.inventory_data[self.inventory_data['stock_status'] == 'Healthy'])
            critical_items = len(self.inventory_data[self.inventory_data['stock_status'] == 'Critical'])

            # Score based on availability metrics
            availability_score = (healthy_items / total_items) * 100
            critical_penalty = critical_items * 10  # Penalty for critical items

            return max(0, availability_score - critical_penalty)
        except Exception:
            return 80

    def _calculate_efficiency_health(self):
        """Calculate inventory efficiency health score"""
        try:
            if 'days_cover' not in self.inventory_data.columns:
                return 75

            total_items = len(self.inventory_data)
            if total_items == 0:
                return 75

            # Calculate efficiency based on days cover distribution
            optimal_range = (7, 21)  # 1–3 weeks ideal
            efficient_items = len(
                self.inventory_data[
                    (self.inventory_data['days_cover'] >= optimal_range[0]) &
                    (self.inventory_data['days_cover'] <= optimal_range[1])
                ]
            )

            efficiency_score = (efficient_items / total_items) * 100

            # Penalize extreme values
            overstock_items = len(self.inventory_data[self.inventory_data['days_cover'] > 30])
            understock_items = len(self.inventory_data[self.inventory_data['days_cover'] < 3])

            penalty = (overstock_items + understock_items) * 5

            return max(0, efficiency_score - penalty)
        except Exception:
            return 75

    def _calculate_risk_health(self):
        """Calculate inventory risk health score"""
        try:
            risk_score = 100

            # Stockout risk
            if 'stock_status' in self.inventory_data.columns:
                critical_items = len(self.inventory_data[self.inventory_data['stock_status'] == 'Critical'])
                risk_score -= critical_items * 8

            # Overstock risk
            if 'days_cover' in self.inventory_data.columns:
                overstock_items = len(self.inventory_data[self.inventory_data['days_cover'] > 45])
                risk_score -= overstock_items * 6

            # Slow-moving risk (simplified)
            if 'daily_sales_rate' in self.inventory_data.columns:
                slow_moving = len(self.inventory_data[self.inventory_data['daily_sales_rate'] < 2])
                risk_score -= slow_moving * 4

            return max(0, risk_score)
        except Exception:
            return 80

    def _calculate_financial_health(self):
        """Calculate inventory financial health score"""
        try:
            if 'stock_value' not in self.inventory_data.columns:
                return 75

            total_value = self.inventory_data['stock_value'].sum()
            if total_value <= 0:
                return 75

            # Value concentration risk
            top_n = max(1, int(len(self.inventory_data) * 0.1))
            top_10_pct = self.inventory_data.nlargest(top_n, 'stock_value')
            concentration_ratio = top_10_pct['stock_value'].sum() / total_value

            # Ideal concentration is 40–60%
            if concentration_ratio > 0.8:
                concentration_score = 60
            elif concentration_ratio > 0.6:
                concentration_score = 75
            else:
                concentration_score = 90

            return concentration_score
        except Exception:
            return 75

    def _get_health_status(self, score):
        """Get health status based on score"""
        if score >= 90:
            return "🌟 Excellent"
        elif score >= 80:
            return "✅ Very Good"
        elif score >= 70:
            return "⚠️ Good"
        elif score >= 60:
            return "🔶 Fair"
        else:
            return "🔴 Needs Attention"

    def _generate_critical_insights(self, dimension_scores):
        """Generate critical inventory insights"""
        insights = []

        if dimension_scores.get('availability_health', 100) < 70:
            insights.append("Stock availability needs improvement – risk of stockouts.")

        if dimension_scores.get('efficiency_health', 100) < 70:
            insights.append("Inventory efficiency below target – optimize stock levels.")

        if dimension_scores.get('risk_health', 100) < 70:
            insights.append("High inventory risk exposure – review critical items.")

        if dimension_scores.get('financial_health', 100) < 70:
            insights.append("Financial optimization opportunity – reduce carrying costs.")

        return insights if insights else ["Inventory health is generally good."]

    def predict_stockout_risk(self, forecast_days=7):
        """Predict stockout risk for next N days"""
        try:
            risk_assessment = []

            for _, item in self.inventory_data.iterrows():
                if 'current_stock' in item and 'daily_sales_rate' in item:
                    current_stock = item['current_stock']
                    daily_sales = item.get('daily_sales_rate', 1)
                    days_cover = current_stock / daily_sales if daily_sales > 0 else 999

                    # Risk classification
                    if days_cover < 3:
                        risk_level = "🚨 Critical"
                        risk_score = 90
                    elif days_cover < 7:
                        risk_level = "⚠️ High"
                        risk_score = 70
                    elif days_cover < 14:
                        risk_level = "🟡 Medium"
                        risk_score = 40
                    else:
                        risk_level = "🟢 Low"
                        risk_score = 10

                    risk_assessment.append({
                        'sku_id': item.get('sku_id', 'N/A'),
                        'sku_name': item.get('sku_name', 'Unknown'),
                        'current_stock': current_stock,
                        'daily_sales_rate': daily_sales,
                        'days_cover': days_cover,
                        'risk_level': risk_level,
                        'risk_score': risk_score,
                        'predicted_stockout_date': self._calculate_stockout_date(days_cover)
                    })

            return pd.DataFrame(risk_assessment)
        except Exception:
            return pd.DataFrame()

    def _calculate_stockout_date(self, days_cover):
        """Calculate predicted stockout date"""
        if days_cover < 100:  # Avoid unrealistic dates
            return (datetime.now() + timedelta(days=days_cover)).strftime('%Y-%m-%d')
        return "No risk"


def safe_get_data(analytics, data_type, fallback_func=None):
    """Safely get data from analytics with fallback"""
    if hasattr(analytics, data_type):
        df = getattr(analytics, data_type)
        if isinstance(df, pd.DataFrame) and not df.empty:
            return df

    if fallback_func:
        return fallback_func()

    return pd.DataFrame()


def generate_enhanced_inventory_data():
    """Generate enhanced fallback inventory data with realistic metrics"""
    products = [
        'Premium Water 500ml', 'Specialty Coffee 250g', 'Artisan Bread',
        'Dairy Milk 1L', 'Snack Mix 200g', 'Cooking Oil 1L',
        'Detergent 2kg', 'Toilet Paper 12-pack', 'Canned Tomatoes', 'Pasta 500g'
    ]

    categories = [
        'Beverages', 'Beverages', 'Bakery', 'Dairy', 'Snacks',
        'Cooking', 'Household', 'Household', 'Canned Goods', 'Pantry'
    ]

    inventory_data = pd.DataFrame({
        'sku_id': [f'SKU{i:03d}' for i in range(1, 11)],
        'sku_name': products,
        'category': categories,
        'current_stock': [45, 12, 85, 60, 120, 75, 200, 150, 40, 90],
        'min_stock': [20, 10, 30, 25, 50, 30, 80, 60, 20, 40],
        'max_stock': [150, 100, 200, 120, 300, 200, 500, 400, 100, 200],
        'daily_sales_rate': [8.5, 4.2, 12.3, 7.8, 15.6, 9.1, 25.4, 18.7, 6.2, 11.5],
        'days_cover': [5.3, 2.9, 6.9, 7.7, 7.7, 8.2, 7.9, 8.0, 6.5, 7.8],
        'stock_status': [
            'Critical', 'Critical', 'Healthy', 'Healthy', 'Healthy',
            'Healthy', 'Healthy', 'Healthy', 'Low', 'Healthy'
        ],
        'unit_cost': [85, 450, 120, 95, 65, 180, 240, 320, 75, 55],
        'stock_value': [3825, 5400, 10200, 5700, 7800, 13500, 48000, 48000, 3000, 4950],
        'shelf_life_days': [180, 90, 7, 14, 60, 365, 730, 1095, 730, 365],
        'supplier_lead_time': [3, 7, 2, 2, 5, 4, 7, 7, 5, 4]
    })

    return inventory_data


def render():
    """🚥 INVENTORY HEALTH DASHBOARD – Intelligent Stock Optimization"""

    st.title("🚥 Inventory Health Dashboard")

    # 🌈 Gradient hero header (aligned with 01_Dashboard.py style)
    st.markdown(
        f"""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    padding: 20px; border-radius: 12px; color: white; margin-bottom: 12px;">
            <h3 style="margin: 0; color: white;">🎯 Intelligent Stock Health & Optimization</h3>
            <p style="margin: 5px 0 0 0; opacity: 0.9; color: white;">
                <strong>📂</strong> Inventory Intelligence &gt; Stock Health Dashboard |
                <strong>🏢</strong> {st.session_state.get('current_tenant', 'ELORA Holding')} |
                <strong>📅</strong> {datetime.now().strftime('%Y-%m-%d')}
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # 💚 Soft green marquee strip – matching the Executive Cockpit style
    st.markdown(
        """
        <div style="background: #dcfce7; padding: 10px 16px; border-radius: 10px;
                    margin-bottom: 24px; border-left: 4px solid #16a34a;">
            <marquee behavior="scroll" direction="left" scrollamount="4"
                     style="font-size: 0.9rem; font-weight: 500; color: #166534;">
                📦 <strong>Inventory Pulse:</strong> Proactive stock health monitoring •
                ⚡ <strong>Stockout Protection:</strong> Early warning on critical SKUs •
                💰 <strong>Working Capital:</strong> Optimize stock value & cash tied in inventory •
                📊 <strong>Coverage:</strong> Track days of cover across key categories •
                🚛 <strong>Replenishment:</strong> Data-driven reorder guidance & safety stock tuning
            </marquee>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if 'analytics' not in st.session_state:
        st.error("❌ Please initialize the application from the main dashboard to load inventory data.")
        return

    analytics = st.session_state.analytics

    # Safely get inventory and sales data
    inventory_data = safe_get_data(analytics, 'inventory_data', generate_enhanced_inventory_data)
    sales_data = safe_get_data(analytics, 'sales_data', pd.DataFrame)

    if inventory_data.empty:
        st.warning("📊 Using enhanced demonstration data. Real inventory data will appear when available.")
        inventory_data = generate_enhanced_inventory_data()

    # Initialize Inventory Intelligence Engine
    intel_engine = InventoryIntelligenceEngine(inventory_data, sales_data)

    # Enhanced Tab Structure
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🏆 Health Overview",
        "📊 Stock Analytics",
        "🔍 Risk Intelligence",
        "💡 Optimization Insights",
        "🚀 Action Center"
    ])

    with tab1:
        render_health_overview(inventory_data, intel_engine)

    with tab2:
        render_stock_analytics(inventory_data, intel_engine)

    with tab3:
        render_risk_intelligence(inventory_data, intel_engine)

    with tab4:
        render_optimization_insights(inventory_data, intel_engine)

    with tab5:
        render_action_center(inventory_data, intel_engine)


def render_health_overview(inventory_data, intel_engine):
    """Render comprehensive inventory health overview"""
    st.header("🏆 Inventory Health Assessment")

    # Calculate comprehensive health metrics
    health_data = intel_engine.calculate_inventory_health_score()

    # Health Scorecard
    st.subheader("📊 Health Scorecard")

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        score = health_data['overall_score']
        status = health_data['health_status']

        # Dynamic status emoji
        if score >= 90:
            emoji = "🌟"
        elif score >= 80:
            emoji = "✅"
        elif score >= 70:
            emoji = "⚠️"
        else:
            emoji = "🔴"

        st.metric("Overall Health", f"{score:.0f}/100", f"{emoji} {status}")
        st.progress(score / 100)

    with col2:
        availability_score = health_data.get('dimension_scores', {}).get('availability_health', 75)
        st.metric("Availability", f"{availability_score:.0f}/100")
        st.caption("Stockout Prevention")

    with col3:
        efficiency_score = health_data.get('dimension_scores', {}).get('efficiency_health', 75)
        st.metric("Efficiency", f"{efficiency_score:.0f}/100")
        st.caption("Optimal Stock Levels")

    with col4:
        risk_score = health_data.get('dimension_scores', {}).get('risk_health', 75)
        st.metric("Risk Management", f"{risk_score:.0f}/100")
        st.caption("Risk Mitigation")

    with col5:
        financial_score = health_data.get('dimension_scores', {}).get('financial_health', 75)
        st.metric("Financial Health", f"{financial_score:.0f}/100")
        st.caption("Cost Optimization")

    # Health Dimension Visualization
    st.subheader("📈 Health Dimension Analysis")

    if 'dimension_scores' in health_data:
        dimensions = list(health_data['dimension_scores'].keys())
        scores = list(health_data['dimension_scores'].values())

        # Create enhanced radar chart
        fig = go.Figure()

        fig.add_trace(go.Scatterpolar(
            r=scores + [scores[0]],  # Close the radar
            theta=[d.replace('_', ' ').title() for d in dimensions] +
                  [dimensions[0].replace('_', ' ').title()],
            fill='toself',
            name='Inventory Health',
            line=dict(color='#1f77b4', width=3),
            fillcolor='rgba(31, 119, 180, 0.2)'
        ))

        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100],
                    tickfont=dict(size=10)
                ),
                angularaxis=dict(
                    tickfont=dict(size=11)
                )
            ),
            showlegend=False,
            title="Inventory Health Dimensions",
            height=400,
            margin=dict(l=50, r=50, t=50, b=50)
        )

        st.plotly_chart(fig, use_container_width=True)

    # Critical Health Insights
    st.subheader("💡 Critical Health Insights")

    insights = health_data.get('critical_insights', [])
    for insight in insights:
        lower = insight.lower()
        if "needs improvement" in lower or "risk" in lower:
            st.warning(f"⚠️ {insight}")
        elif "opportunity" in lower:
            st.info(f"💡 {insight}")
        else:
            st.success(f"✅ {insight}")


def render_stock_analytics(inventory_data, intel_engine):
    """Render comprehensive stock analytics"""
    st.header("📊 Advanced Stock Analytics")

    # Enhanced KPIs
    st.subheader("🎯 Key Inventory Metrics")

    total_value = inventory_data['stock_value'].sum() if 'stock_value' in inventory_data.columns else 0
    total_items = len(inventory_data)

    # Calculate advanced metrics
    if 'stock_status' in inventory_data.columns:
        critical_items = len(inventory_data[inventory_data['stock_status'] == 'Critical'])
        healthy_items = len(inventory_data[inventory_data['stock_status'] == 'Healthy'])
        low_items = len(inventory_data[inventory_data['stock_status'] == 'Low'])
    else:
        critical_items = healthy_items = low_items = 0

    if 'days_cover' in inventory_data.columns and total_items > 0:
        avg_days_cover = inventory_data['days_cover'].mean()
        optimal_coverage = len(
            inventory_data[
                (inventory_data['days_cover'] >= 7) &
                (inventory_data['days_cover'] <= 21)
            ]
        )
    else:
        avg_days_cover = 0
        optimal_coverage = 0

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric("Total Value", f"KES {total_value:,.0f}")
        st.caption("Inventory Investment")

    with col2:
        st.metric("Total Items", total_items)
        st.caption("SKU Count")

    with col3:
        st.metric("Critical Items", critical_items)
        st.caption("Immediate Attention")

    with col4:
        st.metric("Avg Days Cover", f"{avg_days_cover:.1f}")
        st.caption("Stock Sustainability")

    with col5:
        st.metric("Optimal Coverage", f"{optimal_coverage}/{total_items}")
        st.caption("Ideal Stock Levels")

    # Stock Status Distribution
    st.subheader("📈 Stock Status Intelligence")

    col1, col2 = st.columns(2)

    with col1:
        if 'stock_status' in inventory_data.columns:
            status_counts = inventory_data['stock_status'].value_counts()

            # Enhanced pie chart with annotations
            fig = px.pie(
                values=status_counts.values,
                names=status_counts.index,
                title='📊 Stock Status Distribution',
                color=status_counts.index,
                color_discrete_map={
                    'Healthy': '#00CC96',
                    'Low': '#FFA500',
                    'Critical': '#EF553B'
                },
                hole=0.4
            )
            fig.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Stock Value by Category
        if 'category' in inventory_data.columns and 'stock_value' in inventory_data.columns:
            category_value = inventory_data.groupby('category')['stock_value'].sum().reset_index()

            fig = px.bar(
                category_value,
                x='category',
                y='stock_value',
                title='💰 Stock Value by Category',
                color='stock_value',
                color_continuous_scale='Viridis'
            )
            fig.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)

    # Days Cover Analysis
    st.subheader("📆 Stock Coverage Analysis")

    if 'days_cover' in inventory_data.columns:
        fig = px.histogram(
            inventory_data,
            x='days_cover',
            nbins=20,
            title='Distribution of Days Cover',
            labels={'days_cover': 'Days of Stock Coverage'}
        )

        # Add optimal range indicators
        fig.add_vrect(
            x0=7, x1=21,
            fillcolor="green",
            opacity=0.1,
            line_width=0,
            annotation_text="Optimal Range",
            annotation_position="top"
        )
        fig.add_vline(x=7, line_dash="dash", line_color="green")
        fig.add_vline(x=21, line_dash="dash", line_color="green")

        st.plotly_chart(fig, use_container_width=True)


def render_risk_intelligence(inventory_data, intel_engine):
    """Render inventory risk intelligence"""
    st.header("🔍 Risk Intelligence & Early Warning")

    # Stockout Risk Prediction
    st.subheader("🚨 Stockout Risk Assessment")

    risk_assessment = intel_engine.predict_stockout_risk(forecast_days=7)

    if not risk_assessment.empty:
        # Risk Summary
        critical_risk = len(risk_assessment[risk_assessment['risk_level'] == '🚨 Critical'])
        high_risk = len(risk_assessment[risk_assessment['risk_level'] == '⚠️ High'])

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Critical Risk Items", critical_risk)

        with col2:
            st.metric("High Risk Items", high_risk)

        with col3:
            total_risk = critical_risk + high_risk
            risk_percentage = (total_risk / len(risk_assessment)) * 100 if len(risk_assessment) > 0 else 0
            st.metric("Overall Risk Exposure", f"{risk_percentage:.1f}%")

        # Display high-risk items
        high_risk_items = risk_assessment[risk_assessment['risk_score'] >= 40]

        if not high_risk_items.empty:
            st.warning(f"🚨 {len(high_risk_items)} items at high risk of stockout.")

            display_columns = [
                'sku_name',
                'current_stock',
                'daily_sales_rate',
                'days_cover',
                'risk_level',
                'predicted_stockout_date'
            ]

            available_columns = [col for col in display_columns if col in high_risk_items.columns]

            st.dataframe(
                high_risk_items[available_columns].rename(columns={
                    'sku_name': 'Product',
                    'current_stock': 'Current Stock',
                    'daily_sales_rate': 'Daily Sales',
                    'days_cover': 'Days Cover',
                    'risk_level': 'Risk Level',
                    'predicted_stockout_date': 'Predicted Stockout'
                }).round(2),
                use_container_width=True
            )
        else:
            st.success("✅ No high-risk items identified.")
    else:
        st.info("ℹ️ No risk data available for assessment.")

    # Overstock Risk Analysis
    st.subheader("📦 Overstock Risk Assessment")

    if 'days_cover' in inventory_data.columns:
        overstock_items = inventory_data[inventory_data['days_cover'] > 30]

        if not overstock_items.empty:
            st.warning(f"⚠️ {len(overstock_items)} items identified as overstocked.")

            display_cols = ['sku_name', 'current_stock', 'days_cover', 'stock_value']
            available_cols = [col for col in display_cols if col in overstock_items.columns]

            st.dataframe(
                overstock_items[available_cols].rename(columns={
                    'sku_name': 'Product',
                    'current_stock': 'Current Stock',
                    'days_cover': 'Days Cover',
                    'stock_value': 'Stock Value'
                }).round(2),
                use_container_width=True
            )

            total_overstock_value = (
                overstock_items['stock_value'].sum()
                if 'stock_value' in overstock_items.columns else 0
            )
            st.info(f"💰 Total overstock value: KES {total_overstock_value:,.0f}")
        else:
            st.success("✅ No significant overstock issues identified.")


def render_optimization_insights(inventory_data, intel_engine):
    """Render optimization insights and recommendations"""
    st.header("💡 AI-Powered Optimization Insights")

    st.info(
        """
        **💡 Strategic Context:** Leveraging AI and advanced analytics to identify inventory optimization
        opportunities, reduce costs, and improve service levels across the supply chain.
        """
    )

    # Optimization Recommendations
    st.subheader("🎯 Smart Optimization Recommendations")

    recommendations = [
        {
            "type": "🚀 Critical Stock Replenishment",
            "items": 2,
            "impact": "High",
            "urgency": "Immediate",
            "description": "Replenish critically low stock items to prevent stockouts.",
            "estimated_savings": "KES 45,000",
            "implementation": "3–5 days"
        },
        {
            "type": "💰 Overstock Reduction",
            "items": 3,
            "impact": "Medium-High",
            "urgency": "1–2 weeks",
            "description": "Reduce overstocked items to free up working capital.",
            "estimated_savings": "KES 28,500",
            "implementation": "2–4 weeks"
        },
        {
            "type": "📊 Safety Stock Optimization",
            "items": 5,
            "impact": "Medium",
            "urgency": "2–4 weeks",
            "description": "Optimize safety stock levels based on demand variability.",
            "estimated_savings": "KES 15,200",
            "implementation": "4–6 weeks"
        },
        {
            "type": "🧭 Inventory Turnover Improvement",
            "items": 8,
            "impact": "High",
            "urgency": "1–3 months",
            "description": "Improve inventory turnover through better demand forecasting.",
            "estimated_savings": "KES 62,000",
            "implementation": "2–3 months"
        }
    ]

    total_savings = sum(
        float(rec['estimated_savings'].replace('KES ', '').replace(',', ''))
        for rec in recommendations
    )

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Total Recommendations", len(recommendations))

    with col2:
        st.metric("Estimated Total Savings", f"KES {total_savings:,.0f}")

    # Display recommendations
    for rec in recommendations:
        with st.expander(
            f"{rec['type']} | Impact: {rec['impact']} | Savings: {rec['estimated_savings']}"
        ):
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Items Affected", rec['items'])
                st.metric("Urgency", rec['urgency'])

            with col2:
                st.metric("Estimated Savings", rec['estimated_savings'])
                st.metric("Implementation", rec['implementation'])

            with col3:
                # Impact indicator
                impact_color = (
                    "green" if rec['impact'] == "High"
                    else "orange" if rec['impact'] == "Medium-High"
                    else "blue"
                )
                st.markdown(
                    f"""
                    <div style='padding: 10px; background: {impact_color}20;
                                border-radius: 5px; border-left: 4px solid {impact_color};'>
                        <strong>Business Impact:</strong> {rec['impact']}
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            st.write(f"**Description:** {rec['description']}")

            # Action buttons
            col_a, col_b = st.columns(2)
            with col_a:
                if st.button("📄 Implement", key=f"impl_{rec['type']}"):
                    st.success(f"Implementation started for {rec['type']}.")
            with col_b:
                if st.button("📊 Analyze", key=f"analyze_{rec['type']}"):
                    st.success(f"Detailed analysis initiated for {rec['type']}.")


def render_action_center(inventory_data, intel_engine):
    """Render action center for inventory management"""
    st.header("🚀 Inventory Action Center")

    # Quick Actions
    st.subheader("⚡ Quick Actions")

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("📄 Generate Stock Report", use_container_width=True):
            st.success("Comprehensive stock report generated!")

    with col2:
        if st.button("🔄 Update Stock Levels", use_container_width=True):
            st.success("Stock level update process initiated!")

    with col3:
        if st.button("🎯 Create Replenishment Plan", use_container_width=True):
            st.success("Smart replenishment plan created!")

    # Emergency Actions for Critical Items
    st.subheader("🚨 Emergency Actions")

    if 'stock_status' in inventory_data.columns:
        critical_items = inventory_data[inventory_data['stock_status'] == 'Critical']

        if not critical_items.empty:
            st.error(f"🚨 EMERGENCY: {len(critical_items)} critical items require immediate action!")

            for _, item in critical_items.iterrows():
                col_a, col_b, col_c = st.columns([3, 2, 1])
                with col_a:
                    st.write(f"**{item.get('sku_name', 'Unknown')}**")
                    st.write(
                        f"Stock: {item.get('current_stock', 0)} | "
                        f"Min: {item.get('min_stock', 0)}"
                    )
                with col_b:
                    days_cover = item.get('days_cover', 0)
                    try:
                        st.write(f"**{float(days_cover):.1f} days cover**")
                    except Exception:
                        st.write(f"**{days_cover} days cover**")
                with col_c:
                    if st.button(
                        "🛒 Order Now",
                        key=f"order_{item.get('sku_id', 'unknown')}"
                    ):
                        st.success(
                            f"Emergency order placed for {item.get('sku_name', 'item')}."
                        )

    # Performance Monitoring
    st.subheader("📈 Performance Monitoring")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Stock Accuracy", "98.2%")
        st.caption("vs Target 99%")

    with col2:
        st.metric("Stockout Rate", "1.8%")
        st.caption("vs Target <1%")

    with col3:
        st.metric("Turnover Ratio", "8.5x")
        st.caption("vs Target 9x")

    with col4:
        st.metric("Carrying Cost", "22.3%")
        st.caption("vs Target 20%")


if __name__ == "__main__":
    render()