# pages/digital_intelligence/24_Web_Analytics.py
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import sys
import os

# Make project root importable if needed
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))


def render():
    """WEB ANALYTICS - Digital Experience Intelligence & Optimization"""

    st.title("🌐 Web Analytics")

    # 🌈 Enterprise gradient hero header (aligned with 01_Dashboard.py style)
    st.markdown(
        f"""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    padding: 20px; border-radius: 12px; color: white; margin-bottom: 12px;">
            <h3 style="margin: 0; color: white;">🎯 Digital Experience Intelligence &amp; Optimization</h3>
            <p style="margin: 5px 0 0 0; opacity: 0.9; color: white;">
                <strong>📍</strong> Digital Intelligence &gt; Web Analytics |
                <strong>🏢</strong> {st.session_state.get("current_tenant", "ELORA Holding")} |
                <strong>📈</strong> Multi-Channel Attribution &amp; Performance
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 💚 Soft green marquee strip – consistent with Executive Cockpit
    st.markdown(
        """
        <div style="background: #dcfce7; padding: 10px 16px; border-radius: 10px;
                    margin-bottom: 24px; border-left: 4px solid #16a34a;">
            <marquee behavior="scroll" direction="left" scrollamount="4"
                     style="font-size: 0.9rem; font-weight: 500; color: #166534;">
                🌐 <strong>Web Intelligence:</strong> Always-on digital experience monitoring •
                👥 <strong>User Engagement:</strong> High-intent traffic, deep session insights •
                🎯 <strong>Conversion Performance:</strong> Funnel analytics &amp; CRO opportunities •
                🔍 <strong>SEO &amp; Content:</strong> Organic visibility, keyword rankings &amp; ROI •
                ⚡ <strong>Technical Health:</strong> Core Web Vitals, mobile speed &amp; uptime •
                🤖 <strong>AI-Powered Decisions:</strong> Channel efficiency &amp; attribution insights
            </marquee>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Data guards – aligned with other pages (no success banners)
    if "digital_data" not in st.session_state:
        st.error("❌ Digital data not initialized. Please visit **Digital Overview** first.")
        return

    digital_data = st.session_state.digital_data
    if not isinstance(digital_data, dict) or "web_analytics" not in digital_data:
        st.error("❌ `web_analytics` dataset missing in `digital_data`.")
        return

    web_data = digital_data["web_analytics"]
    if not isinstance(web_data, pd.DataFrame) or web_data.empty:
        st.error("❌ `web_analytics` dataset is empty or not a valid DataFrame.")
        return

    # Enterprise 4-Tab Structure
    tab1, tab2, tab3, tab4 = st.tabs(
        [
            "📊 Traffic Intelligence",
            "👥 User Behavior",
            "🎯 Conversion Analytics",
            "🔧 Technical Performance",
        ]
    )

    with tab1:
        render_traffic_intelligence(web_data)

    with tab2:
        render_user_behavior(web_data)

    with tab3:
        render_conversion_analytics(web_data)

    with tab4:
        render_technical_performance(web_data)


# ---------- TRAFFIC INTELLIGENCE ----------


def render_traffic_intelligence(data: pd.DataFrame):
    """Multi-channel traffic acquisition analysis"""

    st.header("📊 Traffic Acquisition & Channel Intelligence")

    required_cols = ["sessions", "users", "pageviews"]
    if not all(col in data.columns for col in required_cols):
        st.error(f"Missing required columns for KPIs: {required_cols}")
        return

    # Traffic KPIs
    col1, col2, col3, col4 = st.columns(4)

    sessions_sum = data["sessions"].sum()
    users_sum = data["users"].sum()
    pageviews_sum = data["pageviews"].sum()

    with col1:
        st.metric("Total Sessions", f"{sessions_sum:,.0f}", "8.3%")

    with col2:
        st.metric("Unique Users", f"{users_sum:,.0f}", "6.7%")

    with col3:
        pages_per_session = pageviews_sum / sessions_sum if sessions_sum else 0
        st.metric("Pages / Session", f"{pages_per_session:.1f}", "2.1%")

    with col4:
        # Simple illustrative new user rate calculation
        try:
            new_user_rate = (users_sum - sessions_sum * 0.3) / users_sum * 100 if users_sum else 0
        except Exception:
            new_user_rate = 0
        st.metric("New User Rate", f"{new_user_rate:.1f}%", "-1.2%")

    st.divider()

    # Channel performance analysis
    st.subheader("🧭 Channel Performance Matrix")

    if not {"channel", "sessions", "users", "pageviews", "bounce_rate", "avg_session_duration"}.issubset(
        set(data.columns)
    ):
        st.warning("Some channel metrics columns are missing; showing limited view.")
        available = list(
            set(["channel", "sessions", "users", "pageviews", "bounce_rate", "avg_session_duration"])
            & set(data.columns)
        )
        channel_metrics = data[available].groupby("channel", as_index=True).sum()
    else:
        channel_metrics = (
            data.groupby("channel")
            .agg(
                {
                    "sessions": "sum",
                    "users": "sum",
                    "pageviews": "sum",
                    "bounce_rate": "mean",
                    "avg_session_duration": "mean",
                }
            )
            .round(2)
        )

    if "sessions" in channel_metrics.columns and "pageviews" in channel_metrics.columns:
        channel_metrics["pages_per_session"] = (
            channel_metrics["pageviews"] / channel_metrics["sessions"].replace(0, np.nan)
        ).fillna(0)
    else:
        channel_metrics["pages_per_session"] = 0.0

    # Synthetic conversion rate just for demo
    channel_metrics["conversion_rate"] = np.random.uniform(1.0, 5.0, len(channel_metrics))

    col1, col2 = st.columns(2)

    with col1:
        try:
            st.dataframe(
                channel_metrics.style.format(
                    {
                        "sessions": "{:,.0f}",
                        "users": "{:,.0f}",
                        "pageviews": "{:,.0f}",
                        "bounce_rate": "{:.1f}%",
                        "avg_session_duration": "{:.0f}s",
                        "pages_per_session": "{:.1f}",
                        "conversion_rate": "{:.1f}%",
                    }
                )
            )
        except Exception:
            # fallback without styling if something goes wrong
            st.dataframe(channel_metrics)

    with col2:
        try:
            cm = channel_metrics.reset_index()
            fig = px.scatter(
                cm,
                x="conversion_rate",
                y="avg_session_duration" if "avg_session_duration" in cm.columns else "sessions",
                size="sessions",
                color="channel",
                title="Channel Efficiency: Conversion vs Engagement",
                hover_data=["bounce_rate"] if "bounce_rate" in cm.columns else None,
            )
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.info(f"Channel scatter could not be rendered: {e}")

    # Traffic trends and seasonality
    st.subheader("📈 Traffic Trends & Seasonality Analysis")

    if "date" in data.columns:
        try:
            daily_traffic = (
                data.groupby("date")
                .agg({"sessions": "sum", "users": "sum"})
                .reset_index()
                .sort_values("date")
            )

            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=daily_traffic["date"],
                    y=daily_traffic["sessions"],
                    mode="lines",
                    name="Sessions",
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=daily_traffic["date"],
                    y=daily_traffic["users"],
                    mode="lines",
                    name="Users",
                )
            )

            fig.update_layout(
                title="Daily Traffic Trends", xaxis_title="Date", yaxis_title="Count"
            )
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.info(f"Traffic trend chart could not be rendered: {e}")
    else:
        st.warning("No `date` column found; cannot render time-series trends.")

    # STRATEGIC INSIGHTS: Channel Attribution & Investment Strategy
    with st.expander("🎯 Channel Attribution & Investment Strategy Framework", expanded=False):
        st.subheader("📋 Multi-Channel Attribution Framework")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown(
                """
                **Channel Performance Assessment**

                **High-Performing Channels**
                - High conversion + Low cost → scale investment  
                - High engagement + Medium cost → optimize further  

                **Moderate Channels**
                - Medium conversion + Medium cost → test & learn  
                - High traffic + Low conversion → improve targeting  

                **Underperforming Channels**
                - Low conversion + High cost → reduce or eliminate  
                - Declining traffic + High cost → strategic review  
                """
            )

        with col2:
            st.markdown(
                """
                **Channel ROI Investment Matrix**

                **INVEST & SCALE** (High ROI)  
                - Top performing channel 1  
                - Top performing channel 2  

                **OPTIMIZE & TEST** (Medium ROI)  
                - Channel with potential 1  
                - Channel with potential 2  

                **REDUCE OR ELIMINATE** (Low ROI)  
                - Underperforming channel 1  
                - Underperforming channel 2  
                """
            )

        st.divider()
        st.subheader("🎯 Strategic Channel Recommendations")

        strategic_recommendations = {
            "Organic Search": {
                "ROI Score": "High",
                "Investment": "Increase 15–20%",
                "Focus": "Content expansion, technical SEO",
                "Expected Impact": "≈25% traffic growth",
            },
            "Paid Search": {
                "ROI Score": "Medium",
                "Investment": "Optimize current spend",
                "Focus": "Keyword refinement, landing pages",
                "Expected Impact": "≈12% conversion improvement",
            },
            "Social Media": {
                "ROI Score": "Low",
                "Investment": "Reduce 10–15%",
                "Focus": "Audience targeting, content quality",
                "Expected Impact": "Better engagement metrics",
            },
        }

        for channel, strategy in strategic_recommendations.items():
            st.markdown(f"**{channel}**")
            cols = st.columns(4)
            cols[0].metric("ROI Score", strategy["ROI Score"])
            cols[1].metric("Investment", strategy["Investment"])
            cols[2].metric("Focus", strategy["Focus"])
            cols[3].metric("Expected Impact", strategy["Expected Impact"])

        st.divider()
        st.subheader("📈 Expected Business Impact")

        st.markdown(
            """
            **Quantifiable Benefits from Channel Optimization**

            💰 **Revenue Impact**: ~18–25% increase through better channel allocation  
            💸 **Cost Efficiency**: ~12–15% reduction in customer acquisition cost  
            🎯 **Conversion Lift**: ~8–12% improvement through channel-specific optimization  
            📊 **ROI Improvement**: ~22–30% better return on marketing investment  
            """
        )


# ---------- USER BEHAVIOR ----------


def render_user_behavior(data: pd.DataFrame):
    """User engagement and behavior analysis"""

    st.header("👥 User Behavior & Engagement Analytics")

    # Behavioral KPIs
    col1, col2, col3, col4 = st.columns(4)

    avg_duration = data["avg_session_duration"].mean() if "avg_session_duration" in data.columns else 0
    bounce_rate = data["bounce_rate"].mean() if "bounce_rate" in data.columns else 0
    pages_session = (
        data["pageviews"].sum() / data["sessions"].sum()
        if {"pageviews", "sessions"}.issubset(data.columns) and data["sessions"].sum()
        else 0
    )

    with col1:
        st.metric("Avg. Session Duration", f"{avg_duration:.0f}s", "5.2%")

    with col2:
        st.metric("Avg. Bounce Rate", f"{bounce_rate:.1f}%", "-3.1%")

    with col3:
        st.metric("Avg. Pages / Session", f"{pages_session:.1f}", "1.8%")

    with col4:
        returning_visitor_rate = 42.3  # Simulated
        st.metric("Returning Visitor Rate", f"{returning_visitor_rate:.1f}%", "2.7%")

    st.divider()

    # Device and technology analysis
    st.subheader("📱 Device & Technology Stack Analysis")

    if "device" in data.columns and "sessions" in data.columns:
        device_analysis = (
            data.groupby("device")
            .agg(
                {
                    "sessions": "sum",
                    "bounce_rate": "mean" if "bounce_rate" in data.columns else "sum",
                    "avg_session_duration": "mean"
                    if "avg_session_duration" in data.columns
                    else "sum",
                }
            )
            .reset_index()
        )

        col1, col2 = st.columns(2)

        with col1:
            fig = px.pie(
                device_analysis,
                values="sessions",
                names="device",
                title="Traffic Distribution by Device",
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            try:
                fig = px.bar(
                    device_analysis,
                    x="device",
                    y=["bounce_rate", "avg_session_duration"],
                    title="Device Performance Comparison",
                    barmode="group",
                    labels={"value": "Metric", "variable": "KPI"},
                )
                st.plotly_chart(fig, use_container_width=True)
            except Exception:
                st.dataframe(device_analysis)
    else:
        st.warning("No `device` column found; device analysis skipped.")

    # User flow and navigation patterns
    st.subheader("🧭 User Navigation & Flow Analysis")

    # Simulated user flow data (labels used only for nodes)
    pages = [
        "Homepage",
        "Product List",
        "Product Detail",
        "Cart",
        "Checkout",
        "Confirmation",
    ]

    fig = go.Figure(
        data=go.Sankey(
            node=dict(
                pad=15,
                thickness=20,
                line=dict(color="black", width=0.5),
                label=pages,
            ),
            link=dict(
                source=[0, 0, 1, 1, 2, 2, 3, 3, 4, 4],
                target=[1, 2, 2, 3, 3, 4, 4, 5, 5, 0],
                value=[65, 35, 45, 20, 22, 23, 18, 4, 15, 3],
            ),
        )
    )

    fig.update_layout(title_text="User Journey Flow Analysis", font_size=10)
    st.plotly_chart(fig, use_container_width=True)

    # STRATEGIC INSIGHTS: Customer Journey Optimization
    with st.expander("🎯 Customer Journey Optimization Framework", expanded=False):
        st.subheader("📋 AIDA Model Analysis (Attention → Interest → Desire → Action)")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown(
                """
                **Journey Stage Performance**

                **ATTENTION (Homepage → Product List)**  
                - Conversion: 65%  
                - Drop-off: 35%  
                - Opportunity: Improve homepage relevance and navigation  

                **INTEREST (Product List → Product Detail)**  
                - Conversion: 45%  
                - Drop-off: 55%  
                - Opportunity: Enhance product listings and filtering  
                """
            )

        with col2:
            st.markdown(
                """
                **DESIRE (Product Detail → Cart)**  
                - Conversion: 22%  
                - Drop-off: 78%  
                - Opportunity: Better product information, social proof  

                **ACTION (Cart → Purchase)**  
                - Conversion: 67%  
                - Drop-off: 33%  
                - Opportunity: Streamline checkout process  
                """
            )

        st.divider()
        st.subheader("🎯 User Experience Optimization Matrix")

        optimization_matrix = {
            "High Impact + Easy Implementation": [
                "Add trust badges to product pages",
                "Simplify checkout form fields",
                "Improve mobile loading speed",
            ],
            "High Impact + Complex Implementation": [
                "Personalized product recommendations",
                "Advanced search functionality",
                "Customer review system integration",
            ],
            "Low Impact + Easy Implementation": [
                "Social sharing buttons",
                "Email signup popup optimization",
                "Footer navigation cleanup",
            ],
        }

        for category, actions in optimization_matrix.items():
            st.markdown(f"**{category}:**")
            for action in actions:
                st.markdown(f"- {action}")

        st.divider()
        st.subheader("📈 Expected Impact from Journey Optimization")

        st.markdown(
            """
            **Quantifiable Improvements**

            🚀 **Conversion Rate**: ~15–25% increase through journey optimization  
            💰 **Average Order Value**: ~8–12% lift with better cross-selling  
            📱 **Mobile Conversion**: ~20–30% improvement with responsive design  
            🔁 **Customer Retention**: ~10–15% increase through better experience  
            """
        )


# ---------- CONVERSION ANALYTICS ----------


def render_conversion_analytics(data: pd.DataFrame):
    """Conversion funnel and goal analysis"""

    st.header("🎯 Conversion Analytics & Funnel Optimization")

    # Conversion KPIs
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        overall_conversion = 2.8  # Simulated
        st.metric("Overall Conversion Rate", f"{overall_conversion:.1f}%", "0.4%")

    with col2:
        ecommerce_conversion = 3.2  # Simulated
        st.metric("Ecommerce Conversion", f"{ecommerce_conversion:.1f}%", "0.7%")

    with col3:
        lead_conversion = 4.5  # Simulated
        st.metric("Lead Conversion", f"{lead_conversion:.1f}%", "1.2%")

    with col4:
        cart_abandonment = 68.3  # Simulated
        st.metric("Cart Abandonment", f"{cart_abandonment:.1f}%", "-2.1%")

    st.divider()

    # Conversion funnel visualization
    st.subheader("📊 Multi-Step Conversion Funnel")

    funnel_stages = [
        "Visitors",
        "Product Views",
        "Add to Cart",
        "Initiate Checkout",
        "Purchases",
    ]
    funnel_values = [10000, 3200, 850, 420, 285]

    fig = go.Funnel(
        y=funnel_stages,
        x=funnel_values,
        textinfo="value+percent initial",
        opacity=0.8,
    )

    fig_funnel = go.Figure(fig)
    fig_funnel.update_layout(title="Ecommerce Conversion Funnel Analysis")
    st.plotly_chart(fig_funnel, use_container_width=True)

    # Channel conversion performance
    st.subheader("📊 Channel-Specific Conversion Performance")

    if "channel" in data.columns:
        channels = data["channel"].unique()
    else:
        channels = ["Organic Search", "Paid Search", "Social", "Direct"]

    channel_conversion = {channel: np.random.uniform(1.0, 6.0) for channel in channels}

    conversion_df = pd.DataFrame(
        {
            "Channel": list(channel_conversion.keys()),
            "Conversion Rate": list(channel_conversion.values()),
            "Cost per Conversion": np.random.uniform(15, 85, len(channels)),
        }
    )

    fig_scatter = px.scatter(
        conversion_df,
        x="Conversion Rate",
        y="Cost per Conversion",
        size="Conversion Rate",
        color="Channel",
        title="Channel Efficiency: Conversion Rate vs Cost",
        hover_data=["Channel"],
    )

    st.plotly_chart(fig_scatter, use_container_width=True)

    # STRATEGIC INSIGHTS: Conversion Rate Optimization Framework
    with st.expander("🎯 Conversion Rate Optimization (CRO) Framework", expanded=False):
        st.subheader("📋 CRO Opportunity Assessment")

        # Flattened – NO nested expanders here
        funnel_analysis = {
            "Visitors → Product Views": {
                "Conversion Rate": "32%",
                "Industry Benchmark": "35–45%",
                "Primary Issues": [
                    "Poor landing page relevance",
                    "Weak value proposition",
                ],
                "Optimization Opportunities": [
                    "Improve meta descriptions and titles",
                    "Enhance page loading speed",
                    "Add compelling hero sections",
                ],
            },
            "Product Views → Add to Cart": {
                "Conversion Rate": "26.6%",
                "Industry Benchmark": "30–40%",
                "Primary Issues": [
                    "Insufficient product information",
                    "Lack of social proof",
                ],
                "Optimization Opportunities": [
                    "Add customer reviews and ratings",
                    "Improve product images and videos",
                    "Implement trust badges and security seals",
                ],
            },
            "Add to Cart → Checkout": {
                "Conversion Rate": "49.4%",
                "Industry Benchmark": "55–65%",
                "Primary Issues": [
                    "Unexpected costs",
                    "Complex checkout process",
                ],
                "Optimization Opportunities": [
                    "Implement guest checkout option",
                    "Show progress indicators",
                    "Provide multiple payment options",
                ],
            },
            "Checkout → Purchase": {
                "Conversion Rate": "67.9%",
                "Industry Benchmark": "70–80%",
                "Primary Issues": [
                    "Payment failures",
                    "Security concerns",
                ],
                "Optimization Opportunities": [
                    "Optimize payment gateway integration",
                    "Add order summary and confirmation",
                    "Implement exit-intent offers",
                ],
            },
        }

        for stage, analysis in funnel_analysis.items():
            st.markdown(f"### {stage}")
            cols = st.columns(2)
            with cols[0]:
                st.metric("Current Rate", analysis["Conversion Rate"])
            with cols[1]:
                st.metric("Benchmark", analysis["Industry Benchmark"])

            st.markdown("**Primary Issues:**")
            for issue in analysis["Primary Issues"]:
                st.markdown(f"- {issue}")

            st.markdown("**Optimization Opportunities:**")
            for opportunity in analysis["Optimization Opportunities"]:
                st.markdown(f"- {opportunity}")

            st.markdown("---")

        st.subheader("🎯 CRO Testing & Implementation Roadmap")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown(
                """
                **Phase 1: Quick Wins (0–30 days)**  
                - A/B test checkout button styles  
                - Implement trust badges on product pages  
                - Optimize page loading speed  
                - Add customer review widgets  

                **Phase 2: Medium-Term (30–60 days)**  
                - Personalize product recommendations  
                - Implement exit-intent popups  
                - Streamline mobile checkout  
                - Add social proof notifications  
                """
            )

        with col2:
            st.markdown(
                """
                **Phase 3: Strategic (60–90 days)**  
                - Advanced personalization engine  
                - Omnichannel experience integration  
                - AI-powered recommendations  

                **Expected CRO Impact**  
                - Overall conversion: +25–35%  
                - Mobile conversion: +40–50%  
                - Average order value: +15–20%  
                - Customer lifetime value: +20–25%  
                """
            )

        st.divider()
        st.subheader("📈 ROI Calculation for CRO Initiatives")

        roi_data = {
            "Initiative": [
                "Checkout Optimization",
                "Mobile Experience",
                "Personalization",
                "Trust Elements",
            ],
            "Investment": ["$15K", "$25K", "$40K", "$8K"],
            "Expected Lift": ["+18%", "+25%", "+22%", "+12%"],
            "Annual Revenue Impact": ["$285K", "$420K", "$350K", "$150K"],
            "Payback Period": ["2.1 months", "2.8 months", "4.2 months", "1.9 months"],
        }

        roi_df = pd.DataFrame(roi_data)
        st.dataframe(roi_df)


# ---------- TECHNICAL PERFORMANCE ----------


def render_technical_performance(data: pd.DataFrame):
    """Technical SEO and site performance analysis"""

    st.header("🔧 Technical Performance & SEO Analytics")

    # Technical KPIs
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        avg_page_load = 2.3  # Simulated
        st.metric("Avg. Page Load Time", f"{avg_page_load:.1f}s", "-0.3s")

    with col2:
        mobile_speed = 75  # Simulated
        st.metric("Mobile Speed Score", f"{mobile_speed}/100", "+5")

    with col3:
        desktop_speed = 88  # Simulated
        st.metric("Desktop Speed Score", f"{desktop_speed}/100", "+3")

    with col4:
        uptime_percentage = 99.92  # Simulated
        st.metric("Site Uptime", f"{uptime_percentage:.2f}%", "0.01%")

    st.divider()

    # Core Web Vitals analysis
    st.subheader("⚡ Core Web Vitals Performance")

    web_vitals_data = {
        "Metric": [
            "Largest Contentful Paint",
            "First Input Delay",
            "Cumulative Layout Shift",
        ],
        "Score": [2.1, 105, 0.08],
        "Status": ["Good", "Needs Improvement", "Good"],
        "Threshold": [2.5, 100, 0.1],
    }

    web_vitals_df = pd.DataFrame(web_vitals_data)

    col1, col2 = st.columns(2)

    with col1:
        st.dataframe(web_vitals_df)

    with col2:
        fig_gauge = go.Figure(
            go.Indicator(
                mode="gauge+number+delta",
                value=75,
                domain={"x": [0, 1], "y": [0, 1]},
                title={"text": "Overall Performance Score"},
                delta={"reference": 70},
                gauge={
                    "axis": {"range": [None, 100]},
                    "bar": {"color": "darkblue"},
                    "steps": [
                        {"range": [0, 50], "color": "lightgray"},
                        {"range": [50, 80], "color": "gray"},
                    ],
                    "threshold": {
                        "line": {"color": "red", "width": 4},
                        "thickness": 0.75,
                        "value": 90,
                    },
                },
            )
        )

        fig_gauge.update_layout(height=300)
        st.plotly_chart(fig_gauge, use_container_width=True)

    # SEO performance tracking
    st.subheader("🔍 SEO Performance & Keyword Tracking")

    keywords = [
        "organic coffee",
        "healthy snacks",
        "energy supplements",
        "premium tea",
        "natural foods",
    ]
    positions = [3, 8, 12, 5, 15]
    traffic_share = [25, 18, 12, 22, 23]

    seo_df = pd.DataFrame(
        {
            "Keyword": keywords,
            "Current Position": positions,
            "Traffic Share %": traffic_share,
            "Previous Position": [4, 7, 15, 6, 18],
        }
    )

    seo_df["Position Change"] = seo_df["Previous Position"] - seo_df["Current Position"]

    try:
        st.dataframe(
            seo_df.style.applymap(
                lambda x: "color: green"
                if x > 0
                else "color: red"
                if x < 0
                else "color: black",
                subset=["Position Change"],
            )
        )
    except Exception:
        st.dataframe(seo_df)

    # STRATEGIC INSIGHTS: Technical SEO & Performance Optimization
    with st.expander(
        "🎯 Technical SEO & Performance Optimization Framework", expanded=False
    ):
        st.subheader("📋 SEO Maturity Assessment")

        seo_maturity_levels = {
            "Basic": ["Meta tags optimization", "Basic keyword research", "XML sitemap"],
            "Intermediate": [
                "Content strategy",
                "Internal linking",
                "Page speed optimization",
            ],
            "Advanced": [
                "Schema markup",
                "International SEO",
                "AI-powered content",
            ],
            "Enterprise": [
                "Predictive analytics",
                "Voice search optimization",
                "AI-driven personalization",
            ],
        }

        current_maturity = "Intermediate"
        target_maturity = "Advanced"

        col1, col2 = st.columns(2)

        with col1:
            st.metric("Current SEO Maturity", current_maturity)
            st.markdown("**Current Capabilities:**")
            for capability in seo_maturity_levels[current_maturity]:
                st.markdown(f"- {capability}")

        with col2:
            st.metric("Target SEO Maturity", target_maturity)
            st.markdown("**Target Capabilities:**")
            for capability in seo_maturity_levels[target_maturity]:
                st.markdown(f"- {capability}")

        st.divider()
        st.subheader("🎯 Technical SEO Optimization Roadmap")

        optimization_roadmap = {
            "Immediate (0–30 days)": [
                "Fix Core Web Vitals issues",
                "Optimize meta descriptions for top pages",
                "Improve internal linking structure",
                "Mobile responsiveness audit",
            ],
            "Short-term (30–60 days)": [
                "Implement schema markup for products",
                "Content gap analysis and creation",
                "Technical SEO audit and fixes",
                "Local SEO optimization",
            ],
            "Medium-term (60–90 days)": [
                "Advanced keyword clustering",
                "AI-powered content optimization",
                "Voice search optimization",
                "International SEO setup",
            ],
            "Long-term (6+ months)": [
                "Predictive SEO analytics",
                "Automated content generation",
                "Competitive intelligence integration",
                "SEO performance prediction models",
            ],
        }

        for timeframe, initiatives in optimization_roadmap.items():
            st.markdown(f"**{timeframe}:**")
            for initiative in initiatives:
                st.markdown(f"- {initiative}")

        st.divider()
        st.subheader("📈 Expected SEO Performance Impact")

        impact_forecast = {
            "Organic Traffic": "≈35–50% increase within 6 months",
            "Keyword Rankings": "≈45% improvement in top 3 positions",
            "Conversion Rate": "≈18–25% lift from qualified traffic",
            "Customer Acquisition Cost": "≈40–60% reduction vs paid channels",
            "Brand Visibility": "≈3–5× increase in search impressions",
        }

        for metric, impact in impact_forecast.items():
            st.metric(metric, impact)

        st.divider()
        st.subheader("💰 SEO ROI Calculation")

        seo_roi_data = {
            "Investment Area": [
                "Technical SEO",
                "Content Creation",
                "Link Building",
                "Tools & Technology",
            ],
            "Monthly Cost": ["$5K", "$8K", "$4K", "$2K"],
            "Expected Traffic Growth": ["+45%", "+60%", "+25%", "+15%"],
            "Estimated Revenue Impact": ["$180K", "$240K", "$100K", "$60K"],
            "ROI Multiple": ["3.6×", "3.0×", "2.5×", "3.0×"],
        }

        seo_roi_df = pd.DataFrame(seo_roi_data)
        st.dataframe(seo_roi_df)


if __name__ == "__main__":
    render()
