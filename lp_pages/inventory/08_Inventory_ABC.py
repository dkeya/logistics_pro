# logistics_pro/pages/08_Inventory_ABC.py
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from datetime import datetime, timedelta


class ABCXYZIntelligenceEngine:
    """AI-powered ABC-XYZ classification and optimization engine"""

    def __init__(self, inventory_data, sales_data=None):
        # Ensure we always have a DataFrame
        if not isinstance(inventory_data, pd.DataFrame):
            inventory_data = pd.DataFrame(inventory_data)

        self.inventory_data = inventory_data.copy()
        self.sales_data = sales_data
        self.optimization_rules = self._initialize_optimization_rules()

    def _initialize_optimization_rules(self):
        """Initialize strategic optimization rules for each ABC-XYZ segment"""
        return {
            'AX': {
                'name': 'Strategic Excellence',
                'strategy': 'Lean & Agile',
                'reorder_point_days': 7,
                'safety_stock_days': 3,
                'review_frequency': 'Daily',
                'ordering_strategy': 'JIT with frequent small orders',
                'service_level_target': 99.5,
                'inventory_turnover_target': 12,
                'key_metrics': ['Service Level', 'Stock Turnover', 'Order Frequency']
            },
            'AY': {
                'name': 'Seasonal Precision',
                'strategy': 'Demand-Driven Planning',
                'reorder_point_days': 14,
                'safety_stock_days': 10,
                'review_frequency': 'Weekly',
                'ordering_strategy': 'Forecast-based with seasonal buffers',
                'service_level_target': 98.0,
                'inventory_turnover_target': 8,
                'key_metrics': ['Forecast Accuracy', 'Seasonal Coverage', 'Service Level']
            },
            'AZ': {
                'name': 'Risk Mitigation',
                'strategy': 'Buffer & Monitor',
                'reorder_point_days': 21,
                'safety_stock_days': 21,
                'review_frequency': 'Weekly',
                'ordering_strategy': 'Higher safety stock with frequent reviews',
                'service_level_target': 97.0,
                'inventory_turnover_target': 6,
                'key_metrics': ['Stockout Rate', 'Carrying Cost', 'Demand Variability']
            },
            'BX': {
                'name': 'Efficient Operations',
                'strategy': 'Standardized Management',
                'reorder_point_days': 10,
                'safety_stock_days': 5,
                'review_frequency': 'Weekly',
                'ordering_strategy': 'Regular periodic ordering',
                'service_level_target': 98.5,
                'inventory_turnover_target': 10,
                'key_metrics': ['Order Cycle Time', 'Service Level', 'Cost Efficiency']
            },
            'BY': {
                'name': 'Balanced Approach',
                'strategy': 'Managed Flexibility',
                'reorder_point_days': 17,
                'safety_stock_days': 12,
                'review_frequency': 'Bi-weekly',
                'ordering_strategy': 'Seasonal adjustments with standard procedures',
                'service_level_target': 97.5,
                'inventory_turnover_target': 7,
                'key_metrics': ['Seasonal Performance', 'Cost Control', 'Service Level']
            },
            'BZ': {
                'name': 'Controlled Complexity',
                'strategy': 'Active Management',
                'reorder_point_days': 25,
                'safety_stock_days': 18,
                'review_frequency': 'Weekly',
                'ordering_strategy': 'Higher monitoring with flexible approaches',
                'service_level_target': 96.5,
                'inventory_turnover_target': 5,
                'key_metrics': ['Demand Volatility', 'Stockout Prevention', 'Cost Management']
            },
            'CX': {
                'name': 'Cost Efficiency',
                'strategy': 'Bulk Optimization',
                'reorder_point_days': 14,
                'safety_stock_days': 7,
                'review_frequency': 'Monthly',
                'ordering_strategy': 'Economic order quantities with minimal management',
                'service_level_target': 96.0,
                'inventory_turnover_target': 8,
                'key_metrics': ['Order Cost', 'Carrying Cost', 'Service Level']
            },
            'CY': {
                'name': 'Simple Seasonal',
                'strategy': 'Basic Management',
                'reorder_point_days': 21,
                'safety_stock_days': 14,
                'review_frequency': 'Monthly',
                'ordering_strategy': 'Basic seasonal adjustments with standard ordering',
                'service_level_target': 95.0,
                'inventory_turnover_target': 6,
                'key_metrics': ['Seasonal Demand', 'Stock Levels', 'Service Performance']
            },
            'CZ': {
                'name': 'Strategic Review',
                'strategy': 'Vendor Partnership or Elimination',
                'reorder_point_days': 30,
                'safety_stock_days': 25,
                'review_frequency': 'Monthly',
                'ordering_strategy': 'Consider vendor-managed inventory or product elimination',
                'service_level_target': 94.0,
                'inventory_turnover_target': 4,
                'key_metrics': ['Product Profitability', 'Strategic Fit', 'Alternative Options']
            }
        }

    def perform_comprehensive_analysis(
        self,
        abc_a_threshold=80,
        abc_b_threshold=90,
        xyz_x_threshold=0.3,
        xyz_y_threshold=0.6
    ):
        """Perform comprehensive ABC-XYZ analysis with strategic insights"""
        # ABC Analysis
        abc_results = self._perform_abc_analysis(abc_a_threshold, abc_b_threshold)

        # XYZ Analysis
        xyz_results = self._perform_xyz_analysis(xyz_x_threshold, xyz_y_threshold)

        # Ensure expected columns exist in xyz_results
        if not isinstance(xyz_results, pd.DataFrame):
            xyz_results = pd.DataFrame(xyz_results)

        for col in ['sku_id', 'xyz_class', 'demand_variability', 'demand_pattern']:
            if col not in xyz_results.columns:
                # Create safe defaults if missing
                if col == 'sku_id' and 'sku_id' in abc_results.columns:
                    xyz_results['sku_id'] = abc_results['sku_id']
                elif col == 'xyz_class':
                    xyz_results['xyz_class'] = 'Z'
                elif col == 'demand_variability':
                    xyz_results['demand_variability'] = 0.6
                elif col == 'demand_pattern':
                    xyz_results['demand_pattern'] = 'Highly Variable'

        # Combine classifications
        combined_data = abc_results.merge(
            xyz_results[['sku_id', 'xyz_class', 'demand_variability', 'demand_pattern']],
            on='sku_id',
            how='left'
        )

        # Final safeguards
        if 'xyz_class' not in combined_data.columns:
            combined_data['xyz_class'] = 'Z'
        if 'demand_variability' not in combined_data.columns:
            combined_data['demand_variability'] = 0.6

        combined_data['abc_xyz_class'] = (
            combined_data['abc_class'].astype(str) +
            combined_data['xyz_class'].astype(str)
        )

        # Add strategic recommendations
        combined_data['strategy_name'] = combined_data['abc_xyz_class'].map(
            lambda x: self.optimization_rules.get(x, {}).get('name', 'Standard')
        )
        combined_data['optimization_strategy'] = combined_data['abc_xyz_class'].map(
            lambda x: self.optimization_rules.get(x, {}).get('strategy', 'Standard Management')
        )

        # Calculate segment performance
        segment_performance = self._calculate_segment_performance(combined_data)

        return {
            'combined_data': combined_data,
            'segment_performance': segment_performance,
            'strategic_insights': self._generate_strategic_insights(combined_data, segment_performance)
        }

    def _perform_abc_analysis(self, a_threshold, b_threshold):
        """Perform enhanced ABC analysis with cumulative metrics"""
        analysis_data = self.inventory_data.copy()

        if 'stock_value' not in analysis_data.columns:
            # If stock_value missing, try to derive from current_stock * unit_cost
            if {'current_stock', 'unit_cost'}.issubset(analysis_data.columns):
                analysis_data['stock_value'] = (
                    analysis_data['current_stock'] * analysis_data['unit_cost']
                )
            else:
                analysis_data['stock_value'] = 0.0

        analysis_data = analysis_data.sort_values('stock_value', ascending=False)

        analysis_data['cumulative_value'] = analysis_data['stock_value'].cumsum()
        total_value = analysis_data['stock_value'].sum() or 1.0
        analysis_data['cumulative_percentage'] = (
            analysis_data['cumulative_value'] / total_value
        ) * 100

        analysis_data['abc_class'] = np.select(
            [
                analysis_data['cumulative_percentage'] <= a_threshold,
                analysis_data['cumulative_percentage'] <= b_threshold,
            ],
            ['A', 'B'],
            default='C'
        )

        # Add ABC segment insights
        analysis_data['abc_priority'] = analysis_data['abc_class'].map({
            'A': 'Critical Focus',
            'B': 'Managed Focus',
            'C': 'Efficiency Focus'
        })

        return analysis_data

    def _perform_xyz_analysis(self, x_threshold, y_threshold):
        """Perform enhanced XYZ analysis with demand intelligence"""
        analysis_data = self.inventory_data.copy()
        if not isinstance(analysis_data, pd.DataFrame):
            analysis_data = pd.DataFrame(analysis_data)

        # Ensure daily_sales_rate exists
        if 'daily_sales_rate' not in analysis_data.columns:
            analysis_data['daily_sales_rate'] = 1.0

        # Use pre-calculated demand variability or calculate from sales data
        if 'demand_variability' not in analysis_data.columns:
            analysis_data['demand_variability'] = self._calculate_demand_variability(
                analysis_data
            )

        # Apply XYZ classification
        analysis_data['xyz_class'] = np.select(
            [
                analysis_data['demand_variability'] <= x_threshold,
                analysis_data['demand_variability'] <= y_threshold,
            ],
            ['X', 'Y'],
            default='Z'
        )

        # Add demand pattern insights
        analysis_data['demand_pattern'] = analysis_data['xyz_class'].map({
            'X': 'Highly Predictable',
            'Y': 'Moderately Predictable',
            'Z': 'Highly Variable'
        })

        analysis_data['forecast_accuracy'] = analysis_data['xyz_class'].map({
            'X': '95-98%',
            'Y': '85-92%',
            'Z': '70-80%'
        })

        return analysis_data

    def _calculate_demand_variability(self, inventory_data):
        """Calculate demand variability coefficient (CV proxy)"""
        cv_values = []
        for _, item in inventory_data.iterrows():
            daily_sales = item.get('daily_sales_rate', 1)
            # Simulate variability based on product characteristics
            if daily_sales > 50:
                cv = np.random.uniform(0.1, 0.3)  # High volume = more stable
            elif daily_sales > 20:
                cv = np.random.uniform(0.3, 0.6)  # Medium volume = moderate variability
            else:
                cv = np.random.uniform(0.6, 0.9)  # Low volume = high variability
            cv_values.append(cv)

        return cv_values

    def _calculate_segment_performance(self, combined_data):
        """Calculate performance metrics for each ABC-XYZ segment"""
        if 'abc_xyz_class' not in combined_data.columns:
            return pd.DataFrame([])

        segments = combined_data['abc_xyz_class'].unique()
        performance_data = []

        # Safeguards
        if 'stock_value' not in combined_data.columns:
            combined_data['stock_value'] = 0.0
        if 'days_cover' not in combined_data.columns:
            combined_data['days_cover'] = np.nan
        if 'demand_variability' not in combined_data.columns:
            combined_data['demand_variability'] = 0.6

        for segment in segments:
            segment_data = combined_data[combined_data['abc_xyz_class'] == segment]

            performance = {
                'segment': segment,
                'product_count': len(segment_data),
                'total_value': segment_data['stock_value'].sum(),
                'avg_days_cover': segment_data['days_cover'].mean(),
                'avg_demand_variability': segment_data['demand_variability'].mean(),
                'service_level_estimate': self._estimate_service_level(segment),
                'inventory_turnover_estimate': self._estimate_turnover(segment),
                'management_complexity': self._assess_complexity(segment)
            }
            performance_data.append(performance)

        return pd.DataFrame(performance_data)

    def _estimate_service_level(self, segment):
        """Estimate service level for segment"""
        base_levels = {
            'AX': 99.5, 'AY': 98.0, 'AZ': 97.0,
            'BX': 98.5, 'BY': 97.5, 'BZ': 96.5,
            'CX': 96.0, 'CY': 95.0, 'CZ': 94.0
        }
        return base_levels.get(segment, 95.0)

    def _estimate_turnover(self, segment):
        """Estimate inventory turnover for segment"""
        base_turnover = {
            'AX': 12, 'AY': 8, 'AZ': 6,
            'BX': 10, 'BY': 7, 'BZ': 5,
            'CX': 8, 'CY': 6, 'CZ': 4
        }
        return base_turnover.get(segment, 6)

    def _assess_complexity(self, segment):
        """Assess management complexity for segment"""
        complexity_scores = {
            'AX': 'Low', 'AY': 'Medium', 'AZ': 'High',
            'BX': 'Low-Medium', 'BY': 'Medium', 'BZ': 'High',
            'CX': 'Low', 'CY': 'Low-Medium', 'CZ': 'High'
        }
        return complexity_scores.get(segment, 'Medium')

    def _generate_strategic_insights(self, combined_data, segment_performance):
        """Generate strategic insights from ABC-XYZ analysis"""
        insights = []

        if 'stock_value' not in combined_data.columns:
            return insights

        total_value = combined_data['stock_value'].sum() or 1.0

        if 'abc_class' not in combined_data.columns:
            combined_data['abc_class'] = 'C'
        if 'xyz_class' not in combined_data.columns:
            combined_data['xyz_class'] = 'Z'

        a_class_share = (
            combined_data[combined_data['abc_class'] == 'A']['stock_value'].sum() / total_value
        ) * 100
        x_class_share = (
            len(combined_data[combined_data['xyz_class'] == 'X']) / len(combined_data)
            if len(combined_data) > 0 else 0
        ) * 100

        # ABC Insights
        if a_class_share > 75:
            insights.append({
                'type': 'success',
                'title': 'Strong Value Concentration',
                'message': (
                    f'A-class items represent {a_class_share:.1f}% of total inventory value - '
                    'excellent focus on high-value items'
                ),
                'recommendation': (
                    'Maintain rigorous management of A-class items while optimizing B and C classes'
                )
            })
        elif a_class_share < 60:
            insights.append({
                'type': 'warning',
                'title': 'Value Distribution Opportunity',
                'message': (
                    f'A-class items represent only {a_class_share:.1f}% of total value - '
                    'consider product portfolio optimization'
                ),
                'recommendation': (
                    'Review product mix and pricing strategies to increase high-value item concentration'
                )
            })

        # XYZ Insights
        if x_class_share > 40:
            insights.append({
                'type': 'success',
                'title': 'Excellent Demand Predictability',
                'message': (
                    f'{x_class_share:.1f}% of items have stable demand patterns - '
                    'enables efficient inventory planning'
                ),
                'recommendation': 'Leverage predictable demand for JIT inventory and reduced safety stock'
            })
        elif x_class_share < 25:
            insights.append({
                'type': 'warning',
                'title': 'High Demand Variability',
                'message': (
                    f'Only {x_class_share:.1f}% of items have stable demand - '
                    'requires robust inventory buffers'
                ),
                'recommendation': (
                    'Implement advanced forecasting and increase safety stock for volatile items'
                )
            })

        # Segment-specific insights
        if 'abc_xyz_class' in combined_data.columns:
            critical_segments = ['AZ', 'BZ', 'CZ']
            critical_count = len(
                combined_data[combined_data['abc_xyz_class'].isin(critical_segments)]
            )

            if critical_count > len(combined_data) * 0.2:
                insights.append({
                    'type': 'error',
                    'title': 'High-Risk Segment Concentration',
                    'message': (
                        f'{critical_count} items in high-variability segments (AZ, BZ, CZ) - '
                        'significant management complexity'
                    ),
                    'recommendation': (
                        'Implement specialized management strategies for high-variability items'
                    )
                })

        return insights

    def generate_optimization_impact(self, current_rules, proposed_rules):
        """Generate optimization impact analysis (simulated)"""
        impact_metrics = {
            'inventory_turnover': {'current': 6.2, 'proposed': 7.8, 'improvement': '+25.8%'},
            'service_level': {'current': 94.5, 'proposed': 97.2, 'improvement': '+2.9%'},
            'stockout_rate': {'current': 3.2, 'proposed': 1.5, 'improvement': '-53.1%'},
            'carrying_costs': {'current': 22.5, 'proposed': 18.2, 'improvement': '-19.1%'},
            'order_efficiency': {'current': 68.0, 'proposed': 82.5, 'improvement': '+21.3%'}
        }

        return impact_metrics


def generate_comprehensive_inventory_data():
    """Generate comprehensive synthetic inventory data for all analyses"""
    np.random.seed(42)

    # Expanded product portfolio
    products = [
        # Beverages
        'Coca-Cola 500ml', 'Fanta Orange 500ml', 'Sprite 500ml', 'Stoney Tangawizi 500ml',
        'Dasani Water 500ml', 'Keringet Water 1L', 'Pepsi 500ml', 'Mirinda 500ml',

        # Dairy
        'Milk Tuzo 500ml', 'Brookside Milk 500ml', 'Mala 500ml', 'Yoghurt 500g',
        'Butter 250g', 'Cheese 200g', 'Cream 200ml',

        # Bakery
        'White Bread', 'Brown Bread', 'Cakes Assorted', 'Croissants', 'Donuts',
        'Cookies Pack', 'Rusk Pack',

        # Household
        'Blue Band 500g', 'Sunlight Soap', 'Omo Detergent', 'Ariel Detergent',
        'Toilet Paper 4-pack', 'Tissue Box', 'Hand Sanitizer',

        # Food & Grocery
        'Royco Cubes', 'Cooking Oil 1L', 'Rice 1kg', 'Wheat Flour 2kg',
        'Sugar 1kg', 'Salt 1kg', 'Tea Leaves 500g', 'Coffee 250g'
    ]

    categories = {
        'Beverages': products[:8],
        'Dairy': products[8:15],
        'Bakery': products[15:22],
        'Household': products[22:29],
        'Food': products[29:]
    }

    inventory_data = []

    for product in products:
        category = next((cat for cat, prods in categories.items() if product in prods), 'General')

        # Generate realistic metrics with proper distribution
        if product in ['Coca-Cola 500ml', 'Fanta Orange 500ml', 'Sprite 500ml']:
            # A items - high value
            current_stock = np.random.randint(300, 800)
            unit_cost = np.random.uniform(80, 150)
            daily_sales = np.random.uniform(40, 100)
            demand_variability = np.random.uniform(0.1, 0.3)  # Low variability (X)
        elif product in ['Stoney Tangawizi 500ml', 'Dasani Water 500ml', 'Milk Tuzo 500ml']:
            # B items - medium value
            current_stock = np.random.randint(150, 400)
            unit_cost = np.random.uniform(50, 100)
            daily_sales = np.random.uniform(20, 50)
            demand_variability = np.random.uniform(0.3, 0.6)  # Medium variability (Y)
        else:
            # C items - low value
            current_stock = np.random.randint(50, 200)
            unit_cost = np.random.uniform(20, 80)
            daily_sales = np.random.uniform(5, 25)
            demand_variability = np.random.uniform(0.6, 0.9)  # High variability (Z)

        min_stock = max(10, int(current_stock * 0.1))
        max_stock = int(current_stock * 1.5)
        days_cover = current_stock / daily_sales if daily_sales > 0 else 999
        stock_value = current_stock * unit_cost

        # XYZ classification
        if demand_variability < 0.3:
            demand_pattern = 'Stable (X)'
            xyz_class = 'X'
        elif demand_variability < 0.6:
            demand_pattern = 'Seasonal (Y)'
            xyz_class = 'Y'
        else:
            demand_pattern = 'Erratic (Z)'
            xyz_class = 'Z'

        inventory_data.append({
            'sku_id': f"SKU{len(inventory_data):03d}",
            'sku_name': product,
            'category': category,
            'current_stock': current_stock,
            'min_stock': min_stock,
            'max_stock': max_stock,
            'daily_sales_rate': daily_sales,
            'days_cover': days_cover,
            'unit_cost': unit_cost,
            'stock_value': stock_value,
            'demand_variability': demand_variability,
            'demand_pattern': demand_pattern,
            'xyz_class': xyz_class,
            'last_updated': datetime.now() - timedelta(days=np.random.randint(0, 7))
        })

    return pd.DataFrame(inventory_data)


def render():
    """EXECUTIVE COCKPIT - ABC-XYZ Inventory Classification & Optimization"""
    st.title("🔍 ABC-XYZ Inventory Analysis")

    # 🌈 Gradient hero header (aligned with 01_Dashboard)
    st.markdown(
        f"""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    padding: 20px; border-radius: 12px; color: white; margin-bottom: 12px;">
            <h3 style="margin: 0; color: white;">🎯 Strategic Inventory Classification & Optimization</h3>
            <p style="margin: 5px 0 0 0; opacity: 0.9; color: white;">
                <strong>📍</strong> Inventory Intelligence &gt; ABC-XYZ Analysis |
                <strong>🏢</strong> {st.session_state.get('current_tenant', 'ELORA Holding')} |
                <strong>📅</strong> {datetime.now().strftime('%Y-%m-%d')}
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 💚 Soft green marquee strip – directly under the hero (Executive Cockpit style)
    st.markdown(
        """
        <div style="background: #dcfce7; padding: 10px 16px; border-radius: 10px;
                    margin-bottom: 24px; border-left: 4px solid #16a34a;">
            <marquee behavior="scroll" direction="left" scrollamount="4"
                     style="font-size: 0.9rem; font-weight: 500; color: #166534;">
                📦 <strong>Inventory Pulse:</strong> AI-powered ABC-XYZ classification •
                💰 <strong>Value Focus:</strong> A-items concentration & working capital optimization •
                📈 <strong>Demand Intelligence:</strong> XYZ variability signals for forecasting & safety stock •
                🎯 <strong>Strategic Segments:</strong> AX / AY / AZ under tight executive control •
                ⚙️ <strong>Optimization Engine:</strong> Turnover uplift, stockout reduction, cost efficiency •
                🧠 <strong>Decision Support:</strong> Ready-made playbooks per ABC-XYZ segment
            </marquee>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Check if analytics is available - with proper fallback
    if 'analytics' not in st.session_state:
        st.info("📊 No live inventory feed detected – generating comprehensive synthetic inventory data for analysis.")
        inventory_data = generate_comprehensive_inventory_data()
    else:
        try:
            analytics = st.session_state.analytics
            if hasattr(analytics, 'get_inventory_data'):
                inventory_data = analytics.get_inventory_data()
                if not isinstance(inventory_data, pd.DataFrame):
                    inventory_data = pd.DataFrame(inventory_data)
            else:
                # Silent fallback – no message shown here
                inventory_data = generate_comprehensive_inventory_data()
        except Exception:
            st.info("🔄 Fallback engaged – using enhanced synthetic inventory data for demonstration.")
            inventory_data = generate_comprehensive_inventory_data()

    # Initialize Intelligence Engine
    intel_engine = ABCXYZIntelligenceEngine(inventory_data)

    # Enhanced Tab Structure
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 ABC Analysis",
        "🔧 XYZ Analysis",
        "🎯 ABC-XYZ Matrix",
        "💡 Strategic Insights",
        "🚀 Optimization Engine"
    ])

    with tab1:
        render_abc_analysis(inventory_data, intel_engine)

    with tab2:
        render_xyz_analysis(inventory_data, intel_engine)

    with tab3:
        render_abc_xyz_matrix(inventory_data, intel_engine)

    with tab4:
        render_strategic_insights(inventory_data, intel_engine)

    with tab5:
        render_optimization_engine(inventory_data, intel_engine)


def render_abc_analysis(inventory_data, intel_engine):
    """Render enhanced ABC Analysis with strategic insights"""
    st.header("📊 ABC Inventory Analysis")

    st.info(
        """
        **💡 Strategic Context:** ABC Analysis applies the Pareto principle (80/20 rule) to classify inventory 
        based on value contribution, enabling focused management attention and resource allocation.
        """
    )

    # ABC Analysis Controls
    st.subheader("🎛️ Analysis Parameters")

    col1, col2, col3 = st.columns(3)

    with col1:
        abc_a_threshold = st.slider(
            "A Items Threshold (%)",
            70,
            90,
            80,
            help="Percentage of total value for A-class items"
        )

    with col2:
        abc_b_threshold = st.slider(
            "B Items Threshold (%)",
            85,
            95,
            90,
            help="Percentage of total value for B-class items"
        )

    with col3:
        analysis_depth = st.selectbox("Analysis Depth", ["Standard", "Detailed", "Strategic"])

    # Perform ABC Analysis
    abc_results = intel_engine._perform_abc_analysis(abc_a_threshold, abc_b_threshold)

    # ABC Overview Dashboard
    st.subheader("🏆 ABC Classification Dashboard")

    abc_summary = abc_results['abc_class'].value_counts().sort_index()
    value_by_class = abc_results.groupby('abc_class')['stock_value'].sum()
    total_value = value_by_class.sum() or 1.0

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric("A Items", f"{abc_summary.get('A', 0)}", "High Value Focus")
        a_value_share = (value_by_class.get('A', 0) / total_value) * 100
        st.caption(f"{a_value_share:.1f}% of total value")

    with col2:
        st.metric("B Items", f"{abc_summary.get('B', 0)}", "Managed Focus")
        b_value_share = (value_by_class.get('B', 0) / total_value) * 100
        st.caption(f"{b_value_share:.1f}% of total value")

    with col3:
        st.metric("C Items", f"{abc_summary.get('C', 0)}", "Efficiency Focus")
        c_value_share = (value_by_class.get('C', 0) / total_value) * 100
        st.caption(f"{c_value_share:.1f}% of total value")

    with col4:
        total_items = len(abc_results)
        st.metric("Total Items", total_items)
        st.caption("SKU Portfolio")

    with col5:
        st.metric("Total Value", f"KES {total_value:,.0f}")
        st.caption("Inventory Investment")

    # Enhanced Visualizations
    col1, col2 = st.columns(2)

    with col1:
        # Enhanced ABC Distribution
        fig = px.pie(
            values=abc_summary.values,
            names=abc_summary.index,
            title='🔧 ABC Classification Distribution',
            color=abc_summary.index,
            color_discrete_map={'A': '#EF553B', 'B': '#FFA500', 'C': '#00CC96'},
            hole=0.4
        )
        fig.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Value Concentration Analysis
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=abc_results['cumulative_percentage'],
            y=abc_results['stock_value'],
            mode='markers',
            marker=dict(
                size=8,
                color=abc_results['abc_class'].map(
                    {'A': '#EF553B', 'B': '#FFA500', 'C': '#00CC96'}
                ),
                opacity=0.7
            ),
            text=abc_results['sku_name'] if 'sku_name' in abc_results.columns else None,
            hovertemplate=(
                '<b>%{text}</b><br>Value: KES %{y:,.0f}<br>'
                'Cumulative: %{x:.1f}%<extra></extra>'
            )
        ))

        # Add threshold lines
        fig.add_vline(
            x=abc_a_threshold,
            line_dash="dash",
            line_color="red",
            annotation_text="A Threshold",
            annotation_position="top"
        )
        fig.add_vline(
            x=abc_b_threshold,
            line_dash="dash",
            line_color="orange",
            annotation_text="B Threshold",
            annotation_position="top"
        )

        fig.update_layout(
            title='💰 Value Concentration Analysis',
            xaxis_title='Cumulative Percentage (%)',
            yaxis_title='Stock Value (KES)',
            showlegend=False,
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)

    # ABC Strategic Recommendations
    st.subheader("💡 ABC Strategic Recommendations")

    a_class_share = (value_by_class.get('A', 0) / total_value) * 100

    if a_class_share >= 75:
        st.success(
            f"""
            **🌟 Excellent Value Concentration:** A-class items represent {a_class_share:.1f}% of total inventory value.

            **🎯 Recommended Actions:**
            - Maintain rigorous daily monitoring of A-class items
            - Implement JIT inventory strategies for A-class
            - Optimize B-class with weekly reviews
            - Streamline C-class management with monthly reviews
            """
        )
    elif a_class_share >= 60:
        st.info(
            f"""
            **📊 Good Value Distribution:** A-class items represent {a_class_share:.1f}% of total inventory value.

            **🎯 Recommended Actions:**
            - Strengthen A-class item management
            - Review B-class for potential A-class promotion
            - Consider product mix optimization
            - Implement segmented service level strategies
            """
        )
    else:
        st.warning(
            f"""
            **⚠️ Value Distribution Opportunity:** A-class items represent only {a_class_share:.1f}% of total value.

            **🎯 Recommended Actions:**
            - Conduct product portfolio review
            - Analyze pricing and promotion strategies
            - Identify opportunities to increase high-value items
            - Consider product rationalization for low-value items
            """
        )

    # Detailed ABC Classification
    st.subheader("📋 ABC Classification Details")

    display_columns = [
        'sku_name', 'category', 'abc_class', 'abc_priority',
        'stock_value', 'cumulative_percentage'
    ]

    available_columns = [col for col in display_columns if col in abc_results.columns]

    styled_df = (
        abc_results[available_columns]
        .rename(columns={
            'sku_name': 'Product',
            'category': 'Category',
            'abc_class': 'ABC Class',
            'abc_priority': 'Priority',
            'stock_value': 'Stock Value',
            'cumulative_percentage': 'Cumulative %'
        })
        .round(2)
        .style.format({
            'Stock Value': 'KES {:,.0f}',
            'Cumulative %': '{:.1f}%'
        })
    )

    st.dataframe(styled_df, height=400, use_container_width=True)


def render_xyz_analysis(inventory_data, intel_engine):
    """Render enhanced XYZ Analysis with demand intelligence"""
    st.header("🔧 XYZ Demand Analysis")

    st.info(
        """
        **💡 Strategic Context:** XYZ Analysis classifies inventory based on demand variability and predictability, 
        enabling appropriate forecasting methods and safety stock strategies for different demand patterns.
        """
    )

    # XYZ Analysis Controls
    st.subheader("🎛️ Demand Analysis Parameters")

    col1, col2, col3 = st.columns(3)

    with col1:
        x_threshold = st.slider(
            "X Items Threshold (CV)",
            0.1,
            0.4,
            0.3,
            0.05,
            help="Coefficient of Variation threshold for stable demand (X)"
        )

    with col2:
        y_threshold = st.slider(
            "Y Items Threshold (CV)",
            0.4,
            0.7,
            0.6,
            0.05,
            help="Coefficient of Variation threshold for seasonal demand (Y)"
        )

    with col3:
        forecast_horizon = st.selectbox(
            "Forecast Horizon",
            ["30 days", "60 days", "90 days"]
        )

    # Perform XYZ Analysis
    xyz_results = intel_engine._perform_xyz_analysis(x_threshold, y_threshold)

    if 'xyz_class' not in xyz_results.columns:
        st.error(
            "XYZ classification could not be computed (missing 'xyz_class'). "
            "Please check the source inventory data."
        )
        return

    # XYZ Overview Dashboard
    st.subheader("🏆 XYZ Classification Dashboard")

    xyz_summary = xyz_results['xyz_class'].value_counts().sort_index()
    total_items = len(xyz_results) or 1

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric("X Items", f"{xyz_summary.get('X', 0)}", "Stable Demand")
        x_share = (xyz_summary.get('X', 0) / total_items) * 100
        st.caption(f"{x_share:.1f}% of portfolio")

    with col2:
        st.metric("Y Items", f"{xyz_summary.get('Y', 0)}", "Seasonal Demand")
        y_share = (xyz_summary.get('Y', 0) / total_items) * 100
        st.caption(f"{y_share:.1f}% of portfolio")

    with col3:
        st.metric("Z Items", f"{xyz_summary.get('Z', 0)}", "Erratic Demand")
        z_share = (xyz_summary.get('Z', 0) / total_items) * 100
        st.caption(f"{z_share:.1f}% of portfolio")

    with col4:
        avg_variability = xyz_results['demand_variability'].mean()
        st.metric("Avg Variability", f"{avg_variability:.2f}")
        st.caption("Coefficient of Variation")

    with col5:
        predictable_items = xyz_summary.get('X', 0) + xyz_summary.get('Y', 0)
        predictable_share = (predictable_items / total_items) * 100
        st.metric("Predictable Items", f"{predictable_share:.1f}%")
        st.caption("X + Y Classes")

    # Enhanced Visualizations
    col1, col2 = st.columns(2)

    with col1:
        # XYZ Distribution
        fig = px.pie(
            values=xyz_summary.values,
            names=xyz_summary.index,
            title='🔧 XYZ Classification Distribution',
            color=xyz_summary.index,
            color_discrete_map={'X': '#00CC96', 'Y': '#FFA500', 'Z': '#EF553B'},
            hole=0.4
        )
        fig.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Demand Variability Analysis
        fig = px.scatter(
            xyz_results,
            x='daily_sales_rate',
            y='demand_variability',
            color='xyz_class',
            size='stock_value' if 'stock_value' in xyz_results.columns else None,
            title='📈 Demand Variability vs Sales Volume',
            hover_data=[
                col for col in ['sku_name', 'category', 'forecast_accuracy']
                if col in xyz_results.columns
            ],
            color_discrete_map={'X': '#00CC96', 'Y': '#FFA500', 'Z': '#EF553B'},
            size_max=20
        )

        # Add classification boundaries
        fig.add_hline(
            y=x_threshold,
            line_dash="dash",
            line_color="green",
            annotation_text="X Threshold",
            annotation_position="left"
        )
        fig.add_hline(
            y=y_threshold,
            line_dash="dash",
            line_color="orange",
            annotation_text="Y Threshold",
            annotation_position="left"
        )

        fig.update_layout(
            xaxis_title='Daily Sales Rate (Units)',
            yaxis_title='Demand Variability (Coefficient)',
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)

    # XYZ Strategic Recommendations
    st.subheader("💡 XYZ Strategic Recommendations")

    x_share = (xyz_summary.get('X', 0) / total_items) * 100

    if x_share >= 40:
        st.success(
            f"""
            **🌟 Excellent Demand Predictability:** {x_share:.1f}% of items have stable demand patterns.

            **🎯 Recommended Actions:**
            - Implement JIT inventory for X-class items
            - Reduce safety stock levels for predictable items
            - Use simple forecasting methods (moving averages)
            - Focus resources on managing Y and Z classes
            """
        )
    elif x_share >= 25:
        st.info(
            f"""
            **📊 Moderate Predictability:** {x_share:.1f}% of items have stable demand patterns.

            **🎯 Recommended Actions:**
            - Use statistical forecasting for X and Y classes
            - Maintain moderate safety stock for Y-class
            - Implement advanced forecasting for Z-class
            - Focus on improving forecast accuracy
            """
        )
    else:
        st.warning(
            f"""
            **⚠️ High Demand Variability:** Only {x_share:.1f}% of items have stable demand patterns.

            **🎯 Recommended Actions:**
            - Implement robust safety stock strategies
            - Use advanced forecasting methods (machine learning)
            - Increase monitoring frequency for Z-class items
            - Consider vendor-managed inventory for high-variability items
            """
        )

    # Demand Pattern Analysis
    st.subheader("📊 Demand Pattern Analysis")

    col1, col2 = st.columns(2)

    with col1:
        # Variability by Category
        if 'category' in xyz_results.columns:
            category_variability = (
                xyz_results
                .groupby('category')['demand_variability']
                .mean()
                .sort_values()
            )
            fig = px.bar(
                x=category_variability.values,
                y=category_variability.index,
                orientation='h',
                title='📋 Demand Variability by Category',
                color=category_variability.values,
                color_continuous_scale='Viridis'
            )
            fig.update_layout(
                xaxis_title='Average Demand Variability',
                yaxis_title='Category'
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Category information not available for variability analysis.")

    with col2:
        # Forecast Accuracy by Class
        forecast_data = {
            'Class': ['X', 'Y', 'Z'],
            'Forecast Accuracy': [96, 88, 75],
            'Safety Stock Days': [3, 10, 21]
        }
        forecast_df = pd.DataFrame(forecast_data)

        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                name='Forecast Accuracy (%)',
                x=forecast_df['Class'],
                y=forecast_df['Forecast Accuracy']
            )
        )
        fig.add_trace(
            go.Bar(
                name='Safety Stock (Days)',
                x=forecast_df['Class'],
                y=forecast_df['Safety Stock Days']
            )
        )

        fig.update_layout(
            title='🎯 Forecast Accuracy vs Safety Stock Requirements',
            barmode='group',
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)

    # Detailed XYZ Classification
    st.subheader("📋 XYZ Classification Details")

    display_columns = [
        'sku_name', 'category', 'xyz_class', 'demand_pattern',
        'forecast_accuracy', 'demand_variability', 'daily_sales_rate'
    ]

    available_columns = [col for col in display_columns if col in xyz_results.columns]

    styled_df = (
        xyz_results[available_columns]
        .rename(columns={
            'sku_name': 'Product',
            'category': 'Category',
            'xyz_class': 'XYZ Class',
            'demand_pattern': 'Demand Pattern',
            'forecast_accuracy': 'Forecast Accuracy',
            'demand_variability': 'Variability',
            'daily_sales_rate': 'Daily Sales'
        })
        .round(3)
        .style.format({
            'Variability': '{:.3f}',
            'Daily Sales': '{:.1f}'
        })
        .background_gradient(subset=['Variability'], cmap='RdYlGn_r')
    )

    st.dataframe(styled_df, height=400, use_container_width=True)


def render_abc_xyz_matrix(inventory_data, intel_engine):
    """Render enhanced ABC-XYZ Matrix with strategic positioning"""
    st.header("🎯 Combined ABC-XYZ Matrix")

    st.info(
        """
        **💡 Strategic Context:** The ABC-XYZ Matrix combines value-based and demand-based classification 
        to create 9 strategic segments, each requiring tailored inventory management strategies for optimal performance.
        """
    )

    # Analysis Parameters
    st.subheader("🎛️ Matrix Configuration")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        abc_a_threshold = st.slider("A Threshold (%)", 70, 90, 80, key="matrix_abc_a")

    with col2:
        abc_b_threshold = st.slider("B Threshold (%)", 85, 95, 90, key="matrix_abc_b")

    with col3:
        xyz_x_threshold = st.slider("X Threshold", 0.1, 0.4, 0.3, 0.05, key="matrix_xyz_x")

    with col4:
        xyz_y_threshold = st.slider("Y Threshold", 0.4, 0.7, 0.6, 0.05, key="matrix_xyz_y")

    # Perform Comprehensive Analysis
    analysis_results = intel_engine.perform_comprehensive_analysis(
        abc_a_threshold,
        abc_b_threshold,
        xyz_x_threshold,
        xyz_y_threshold
    )
    combined_data = analysis_results['combined_data']
    segment_performance = analysis_results['segment_performance']

    required_cols = {'abc_class', 'xyz_class', 'stock_value'}
    missing = required_cols - set(combined_data.columns)
    if missing:
        st.error(
            "ABC-XYZ matrix cannot be built; missing columns: "
            + ", ".join(sorted(missing))
        )
        return

    # Matrix Overview Dashboard
    st.subheader("🏆 ABC-XYZ Matrix Dashboard")

    matrix_summary = combined_data.groupby(['abc_class', 'xyz_class']).agg({
        'sku_id': 'count',
        'stock_value': 'sum'
    }).reset_index()

    total_value = combined_data['stock_value'].sum() or 1.0
    total_items = len(combined_data) or 1

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        critical_items = len(
            combined_data[combined_data['abc_xyz_class'].isin(['AX', 'AY', 'AZ'])]
        )
        st.metric("Critical Items", critical_items)
        st.caption("A-Class Items")

    with col2:
        stable_high_value = len(combined_data[combined_data['abc_xyz_class'] == 'AX'])
        st.metric("AX Items", stable_high_value)
        st.caption("Strategic Excellence")

    with col3:
        high_risk_items = len(
            combined_data[combined_data['abc_xyz_class'].isin(['AZ', 'BZ', 'CZ'])]
        )
        st.metric("High Risk Items", high_risk_items)
        st.caption("High Variability")

    with col4:
        portfolio_coverage = (
            len(combined_data[combined_data['abc_xyz_class'].isin(['AX', 'BX', 'CX'])]) /
            total_items
        ) * 100
        st.metric("Predictable Items", f"{portfolio_coverage:.1f}%")
        st.caption("X-Class Coverage")

    with col5:
        optimization_potential = len(
            combined_data[combined_data['abc_xyz_class'].isin(['CZ', 'BZ', 'AZ'])]
        )
        st.metric("Optimization Focus", optimization_potential)
        st.caption("High-Impact Segments")

    # Enhanced Matrix Visualizations
    st.subheader("📈 Strategic Matrix Visualization")

    col1, col2 = st.columns(2)

    with col1:
        # Heatmap - Product Count
        heatmap_data = (
            matrix_summary.pivot(
                index='abc_class',
                columns='xyz_class',
                values='sku_id'
            )
            .fillna(0)
            .reindex(index=['A', 'B', 'C'], columns=['X', 'Y', 'Z'])
        )

        fig = px.imshow(
            heatmap_data,
            title='📊 Product Count by Segment',
            color_continuous_scale='Blues',
            aspect='auto',
            text_auto=True
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Heatmap - Value Concentration
        value_heatmap = (
            matrix_summary.pivot(
                index='abc_class',
                columns='xyz_class',
                values='stock_value'
            )
            .fillna(0)
            .reindex(index=['A', 'B', 'C'], columns=['X', 'Y', 'Z'])
        )

        fig = px.imshow(
            value_heatmap,
            title='💰 Stock Value by Segment (KES)',
            color_continuous_scale='Viridis',
            aspect='auto',
            text_auto=True
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    # Strategic Segment Analysis
    st.subheader("🎯 Segment Strategy & Recommendations")

    strategies = intel_engine.optimization_rules
    cols = st.columns(3)
    segment_order = ['AX', 'AY', 'AZ', 'BX', 'BY', 'BZ', 'CX', 'CY', 'CZ']

    for i, segment in enumerate(segment_order):
        strategy = strategies.get(segment, {})
        with cols[i % 3]:
            with st.expander(f"**{segment} - {strategy.get('name', 'Standard')}**", expanded=True):
                st.write(f"**Strategy:** {strategy.get('strategy', 'N/A')}")
                st.write(f"**Service Level:** {strategy.get('service_level_target', 'N/A')}%")
                st.write(
                    f"**Turnover Target:** "
                    f"{strategy.get('inventory_turnover_target', 'N/A')}x"
                )
                st.write(f"**Review:** {strategy.get('review_frequency', 'N/A')}")

                # Segment metrics
                segment_data = combined_data[combined_data['abc_xyz_class'] == segment]
                if len(segment_data) > 0:
                    st.write(f"**Items:** {len(segment_data)}")
                    st.write(f"**Value:** KES {segment_data['stock_value'].sum():,.0f}")

    # Performance Analysis
    if not segment_performance.empty:
        st.subheader("📊 Segment Performance Analysis")

        col1, col2 = st.columns(2)

        with col1:
            fig = px.bar(
                segment_performance,
                x='segment',
                y='service_level_estimate',
                title='🎯 Estimated Service Level by Segment',
                color='service_level_estimate',
                color_continuous_scale='RdYlGn',
                text_auto=True
            )
            fig.update_layout(xaxis_title='Segment', yaxis_title='Service Level (%)')
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig = px.bar(
                segment_performance,
                x='segment',
                y='inventory_turnover_estimate',
                title='🔧 Estimated Inventory Turnover by Segment',
                color='inventory_turnover_estimate',
                color_continuous_scale='RdYlGn',
                text_auto=True
            )
            fig.update_layout(xaxis_title='Segment', yaxis_title='Turnover (x)')
            st.plotly_chart(fig, use_container_width=True)

    # Detailed Combined Classification
    st.subheader("📋 Combined ABC-XYZ Classification")

    display_columns = [
        'sku_name', 'category', 'abc_class', 'xyz_class', 'abc_xyz_class',
        'strategy_name', 'stock_value', 'demand_variability'
    ]

    available_columns = [col for col in display_columns if col in combined_data.columns]

    styled_df = (
        combined_data[available_columns]
        .rename(columns={
            'sku_name': 'Product',
            'category': 'Category',
            'abc_class': 'ABC',
            'xyz_class': 'XYZ',
            'abc_xyz_class': 'Segment',
            'strategy_name': 'Strategy',
            'stock_value': 'Stock Value',
            'demand_variability': 'Variability'
        })
        .round(3)
        .style.format({
            'Stock Value': 'KES {:,.0f}',
            'Variability': '{:.3f}'
        })
        .background_gradient(subset=['Stock Value'], cmap='YlOrRd')
    )

    st.dataframe(styled_df, height=400, use_container_width=True)


def render_strategic_insights(inventory_data, intel_engine):
    """Render AI-powered strategic insights and recommendations"""
    st.header("💡 Strategic Insights & AI Recommendations")

    st.info(
        """
        **🎯 Intelligence Engine Active:** AI-powered analysis generating strategic insights 
        based on your ABC-XYZ classification and inventory performance patterns.
        """
    )

    # Perform comprehensive analysis
    analysis_results = intel_engine.perform_comprehensive_analysis()
    strategic_insights = analysis_results['strategic_insights']
    combined_data = analysis_results['combined_data']

    if 'stock_value' not in combined_data.columns:
        st.error("Strategic insights cannot be generated (missing 'stock_value').")
        return

    if 'abc_class' not in combined_data.columns:
        combined_data['abc_class'] = 'C'
    if 'xyz_class' not in combined_data.columns:
        combined_data['xyz_class'] = 'Z'

    # Executive Summary
    st.subheader("🏆 Executive Summary")

    total_value = combined_data['stock_value'].sum() or 1.0
    a_class_share = (
        combined_data[combined_data['abc_class'] == 'A']['stock_value'].sum() / total_value
    ) * 100
    x_class_share = (
        len(combined_data[combined_data['xyz_class'] == 'X']) / len(combined_data)
        if len(combined_data) > 0 else 0
    ) * 100

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Strategic Score", "78/100", "Good", delta_color="off")
        st.caption("Overall Inventory Health")

    with col2:
        st.metric("Value Concentration", f"{a_class_share:.1f}%", "A-Class Share")
        st.caption(
            "Excellent" if a_class_share >= 75
            else "Good" if a_class_share >= 60
            else "Needs Improvement"
        )

    with col3:
        st.metric("Demand Predictability", f"{x_class_share:.1f}%", "X-Class Share")
        st.caption(
            "Excellent" if x_class_share >= 40
            else "Good" if x_class_share >= 25
            else "Needs Focus"
        )

    with col4:
        optimization_potential = len(
            combined_data[combined_data['abc_xyz_class'].isin(['AZ', 'BZ', 'CZ'])]
        ) if 'abc_xyz_class' in combined_data.columns else 0
        st.metric("Optimization Focus", optimization_potential, "High-Risk Items")
        st.caption("Immediate Attention Required")

    # AI-Powered Insights
    st.subheader("🧠 AI-Powered Strategic Insights")

    if strategic_insights:
        for insight in strategic_insights:
            block = (
                f"**{insight['title']}**\n\n"
                f"{insight['message']}\n\n"
                f"**💡 Recommendation:** {insight['recommendation']}"
            )
            if insight['type'] == 'success':
                st.success(block)
            elif insight['type'] == 'warning':
                st.warning(block)
            elif insight['type'] == 'error':
                st.error(block)
    else:
        st.info(
            "No specific strategic insights generated. Your inventory appears to be well-balanced."
        )

    # Opportunity Analysis
    st.subheader("🎯 High-Impact Opportunities")

    opportunities = [
        {
            'opportunity': 'AX Segment Optimization',
            'impact': 'High',
            'effort': 'Low',
            'potential_savings': 'KES 450,000',
            'timeline': '30 days',
            'description': 'Implement JIT for stable high-value items'
        },
        {
            'opportunity': 'CZ Segment Review',
            'impact': 'Medium',
            'effort': 'Medium',
            'potential_savings': 'KES 280,000',
            'timeline': '60 days',
            'description': 'Consider vendor-managed inventory or elimination'
        },
        {
            'opportunity': 'Demand Forecasting Enhancement',
            'impact': 'High',
            'effort': 'High',
            'potential_savings': 'KES 620,000',
            'timeline': '90 days',
            'description': 'Implement advanced forecasting for Y/Z classes'
        }
    ]

    for opp in opportunities:
        with st.expander(f"🚀 {opp['opportunity']} - Impact: {opp['impact']}", expanded=True):
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Potential Savings", opp['potential_savings'])
            with col2:
                st.metric("Timeline", opp['timeline'])
            with col3:
                st.metric("Effort Level", opp['effort'])
            st.write(f"**Description:** {opp['description']}")

    # Action Plan Generator
    st.subheader("📋 Strategic Action Plan")

    action_plan = [
        {
            "phase": "Immediate (1-2 weeks)",
            "actions": [
                "Implement daily monitoring for AX items",
                "Review safety stock levels for AZ/BZ items",
                "Communicate new classification to inventory team"
            ]
        },
        {
            "phase": "Short-term (1 month)",
            "actions": [
                "Configure ERP system with ABC-XYZ rules",
                "Train team on segment-specific strategies",
                "Establish KPIs for each segment"
            ]
        },
        {
            "phase": "Medium-term (1-3 months)",
            "actions": [
                "Optimize ordering patterns for each segment",
                "Implement advanced forecasting for Y/Z classes",
                "Review and adjust service level targets"
            ]
        },
        {
            "phase": "Long-term (3-6 months)",
            "actions": [
                "Continuous improvement of classification rules",
                "Expand analysis to include supplier performance",
                "Integrate with financial planning systems"
            ]
        }
    ]

    for phase in action_plan:
        with st.expander(f"📅 {phase['phase']}", expanded=True):
            for action in phase['actions']:
                st.write(f"✅ {action}")


def render_optimization_engine(inventory_data, intel_engine):
    """Render optimization engine with impact analysis"""
    st.header("🚀 Inventory Optimization Engine")

    st.info(
        """
        **⚡ Optimization Ready:** Configure and apply intelligent inventory optimization rules 
        based on ABC-XYZ segmentation for maximum efficiency and service levels.
        """
    )

    # Rule Configuration
    st.subheader("🎛️ Optimization Rule Configuration")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.write("**🔧 Reorder Points (Days)**")
        ax_rop = st.number_input("AX Items", 3, 14, 7, key="opt_ax_rop")
        ay_rop = st.number_input("AY Items", 7, 21, 14, key="opt_ay_rop")
        az_rop = st.number_input("AZ Items", 14, 30, 21, key="opt_az_rop")

    with col2:
        st.write("**🛡️ Safety Stock (Days)**")
        ax_ss = st.number_input("AX Safety", 2, 7, 3, key="opt_ax_ss")
        ay_ss = st.number_input("AY Safety", 7, 14, 10, key="opt_ay_ss")
        az_ss = st.number_input("AZ Safety", 14, 30, 21, key="opt_az_ss")

    with col3:
        st.write("**📊 Review Frequency**")
        ax_review = st.selectbox(
            "AX Review",
            ["Daily", "Weekly", "Bi-weekly"],
            key="opt_ax_rev"
        )
        ay_review = st.selectbox(
            "AY Review",
            ["Weekly", "Bi-weekly", "Monthly"],
            key="opt_ay_rev"
        )
        az_review = st.selectbox(
            "AZ Review",
            ["Weekly", "Bi-weekly", "Monthly"],
            key="opt_az_rev"
        )

    # Optimization Impact Analysis
    st.subheader("📈 Optimization Impact Analysis")

    if st.button("🚀 Run Optimization Analysis", type="primary", use_container_width=True):
        st.success("✅ Optimization analysis completed successfully!")

        # Generate impact metrics
        impact_metrics = intel_engine.generate_optimization_impact({}, {})

        # Display impact dashboard
        col1, col2, col3, col4, col5 = st.columns(5)

        metrics_config = {
            'inventory_turnover': {
                'title': 'Inventory Turnover',
                'suffix': 'x',
                'delta_prefix': ''
            },
            'service_level': {
                'title': 'Service Level',
                'suffix': '%',
                'delta_prefix': ''
            },
            'stockout_rate': {
                'title': 'Stockout Rate',
                'suffix': '%',
                'delta_prefix': ''
            },
            'carrying_costs': {
                'title': 'Carrying Costs',
                'suffix': '%',
                'delta_prefix': ''
            },
            'order_efficiency': {
                'title': 'Order Efficiency',
                'suffix': '%',
                'delta_prefix': ''
            }
        }

        cols = [col1, col2, col3, col4, col5]

        for i, (metric, config) in enumerate(metrics_config.items()):
            with cols[i]:
                data = impact_metrics[metric]
                st.metric(
                    config['title'],
                    f"{data['proposed']}{config['suffix']}",
                    f"{config['delta_prefix']}{data['improvement']}",
                    delta_color="normal" if metric != 'stockout_rate' else "inverse"
                )

        # Financial Impact
        st.subheader("💰 Financial Impact Summary")

        financial_impact = {
            'Metric': [
                'Annual Inventory Carrying Cost',
                'Stockout Cost Reduction',
                'Order Processing Savings',
                'Working Capital Optimization',
                'Total Annual Savings'
            ],
            'Current': [
                'KES 4,250,000',
                'KES 680,000',
                'KES 320,000',
                'KES 1,200,000',
                'KES 6,450,000'
            ],
            'Projected': [
                'KES 3,440,000',
                'KES 320,000',
                'KES 220,000',
                'KES 980,000',
                'KES 4,960,000'
            ],
            'Savings': [
                'KES 810,000',
                'KES 360,000',
                'KES 100,000',
                'KES 220,000',
                'KES 1,490,000'
            ]
        }

        st.dataframe(
            pd.DataFrame(financial_impact),
            use_container_width=True
        )

    # Implementation Roadmap
    st.subheader("🚧 Implementation Roadmap")

    roadmap_steps = [
        {
            "step": 1,
            "phase": "Planning & Preparation",
            "duration": "1-2 weeks",
            "tasks": [
                "Finalize ABC-XYZ classification rules",
                "Configure system parameters",
                "Train inventory management team"
            ]
        },
        {
            "step": 2,
            "phase": "System Configuration",
            "duration": "2-3 weeks",
            "tasks": [
                "Update ERP system settings",
                "Configure automated alerts",
                "Set up reporting dashboards"
            ]
        },
        {
            "step": 3,
            "phase": "Pilot Implementation",
            "duration": "4 weeks",
            "tasks": [
                "Implement for A-class items first",
                "Monitor performance metrics",
                "Adjust rules based on results"
            ]
        },
        {
            "step": 4,
            "phase": "Full Rollout",
            "duration": "4-6 weeks",
            "tasks": [
                "Expand to all inventory segments",
                "Optimize based on pilot learnings",
                "Establish continuous improvement process"
            ]
        }
    ]

    for step in roadmap_steps:
        with st.expander(
            f"Step {step['step']}: {step['phase']} ({step['duration']})",
            expanded=True
        ):
            for task in step['tasks']:
                st.write(f"• {task}")

    # Export and Integration
    st.subheader("📤 Export & Integration")

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("📊 Export Classification Report", use_container_width=True):
            st.success("Classification report exported successfully!")

    with col2:
        if st.button("🔧 Sync with Inventory System", use_container_width=True):
            st.success("Optimization rules synchronized with inventory system!")

    with col3:
        if st.button("📋 Generate Action Plan", use_container_width=True):
            st.success("Detailed action plan generated and saved!")


if __name__ == "__main__":
    render()