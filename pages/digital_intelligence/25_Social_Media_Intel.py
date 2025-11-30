# -*- coding: utf-8 -*-
# pages/digital_intelligence/25_Social_Media_Intel.py

import os
import sys
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# --------------------------------------------------------------------
# Ensure project root is on the path
# --------------------------------------------------------------------
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)


def _require_columns(df: pd.DataFrame, required: list, context: str) -> bool:
    """
    Utility: check that required columns exist in df.
    If missing, show a friendly error and return False.
    """
    missing = [c for c in required if c not in df.columns]
    if missing:
        st.error(
            f"❌ Missing columns for **{context}**: "
            f"{', '.join(missing)}. Please check your digital_data structure."
        )
        return False
    return True


def render():
    """📱 SOCIAL MEDIA INTELLIGENCE - Enterprise Social Listening & Analytics"""

    st.title("📱 Social Media Intelligence")

    # 🌈 Gradient hero header (aligned with 01_Dashboard pattern)
    st.markdown(
        f"""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    padding: 20px; border-radius: 12px; color: white; margin-bottom: 12px;">
            <h3 style="margin: 0; color: white;">🎯 Enterprise Social Listening & Multi-Platform Analytics</h3>
            <p style="margin: 5px 0 0 0; opacity: 0.9; color: white;">
                <strong>📍</strong> Digital Intelligence &gt; Social Media Intelligence |
                <strong>🏢</strong> {st.session_state.get('current_tenant', 'ELORA Holding')} |
                <strong>📅</strong> {datetime.now().strftime('%Y-%m-%d')}
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 💚 Soft green marquee strip – same UX pattern as Executive Cockpit
    st.markdown(
        """
        <div style="background: #dcfce7; padding: 10px 16px; border-radius: 10px;
                    margin-bottom: 24px; border-left: 4px solid #16a34a;">
            <marquee behavior="scroll" direction="left" scrollamount="4"
                     style="font-size: 0.9rem; font-weight: 500; color: #166534;">
                🌐 <strong>Social Intelligence Live Feed:</strong>
                • 📣 Reach 1.8M | Engagement Rate 4.3% • 💬 Brand Sentiment 82.5% Positive
                • 👥 Community Growth +14.2% MoM • 🎯 Top Platform: Instagram 5.1% ER
                • 🛒 Social-Driven Revenue +31% QoQ • 🧭 Active Campaigns: 4 | Influencer Programs: 2
            </marquee>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Data availability checks (after header + marquee, same pattern as 01_Dashboard)
    if "digital_data" not in st.session_state:
        st.error("❌ Digital data not initialized. Please visit **Digital Overview** first.")
        return

    digital_data = st.session_state.digital_data

    if not isinstance(digital_data, dict):
        st.error("❌ `digital_data` is not in the expected format (dict-like).")
        return

    social_data = digital_data.get("social_media")
    competitive_data = digital_data.get("competitive")

    if social_data is None or not isinstance(social_data, pd.DataFrame):
        st.error("❌ `digital_data['social_media']` is missing or not a DataFrame.")
        return

    # competitive_data is only required for the competitive tab
    if competitive_data is None or not isinstance(competitive_data, pd.DataFrame):
        st.warning(
            "⚠️ `digital_data['competitive']` is missing or not a DataFrame. "
            "The *Competitive Intelligence* tab will be limited."
        )
        competitive_data = pd.DataFrame()

    # Enterprise 4-Tab Structure
    tab1, tab2, tab3, tab4 = st.tabs(
        [
            "📊 Social Performance",
            "🎭 Sentiment Analysis",
            "🔍 Competitive Intelligence",
            "🚀 Campaign Analytics",
        ]
    )

    with tab1:
        render_social_performance(social_data)
    with tab2:
        render_sentiment_analysis(social_data)
    with tab3:
        render_competitive_intelligence(social_data, competitive_data)
    with tab4:
        render_campaign_analytics(social_data)


# ====================================================================
# TAB 1: SOCIAL PERFORMANCE
# ====================================================================


def render_social_performance(data: pd.DataFrame):
    """Multi-platform social media performance analytics"""

    required_cols = [
        "engagements",
        "reach",
        "engagement_rate",
        "followers",
        "sentiment_score",
        "platform",
        "date",
    ]
    if not _require_columns(data, required_cols, "Social Performance"):
        return

    st.header("📊 Social Media Performance Dashboard")

    # Social KPIs
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        total_engagement = data["engagements"].sum()
        st.metric(
            "Total Engagements",
            f"{total_engagement:,.0f}",
            "18.3%",
            help="Likes, comments, shares across all platforms",
        )

    with col2:
        total_reach = data["reach"].sum()
        st.metric(
            "Total Reach",
            f"{total_reach:,.0f}",
            "22.7%",
            help="Unique users who saw your content",
        )

    with col3:
        avg_engagement_rate = data["engagement_rate"].mean()
        st.metric(
            "Avg. Engagement Rate",
            f"{avg_engagement_rate:.2f}%",
            "1.4%",
            help="Engagements per follower across platforms",
        )

    with col4:
        follower_growth = 12.8  # Simulated
        st.metric(
            "Follower Growth Rate",
            f"{follower_growth:.1f}%",
            "2.3%",
            help="Monthly follower growth across all platforms",
        )

    st.divider()

    # Platform performance comparison
    st.subheader("🔧 Multi-Platform Performance Analysis")

    platform_metrics = (
        data.groupby("platform")
        .agg(
            {
                "followers": "last",
                "engagements": "sum",
                "reach": "sum",
                "engagement_rate": "mean",
                "sentiment_score": "mean",
            }
        )
        .round(3)
    )

    # Avoid division by zero
    platform_metrics["engagement_ratio"] = np.where(
        platform_metrics["followers"] > 0,
        (platform_metrics["engagements"] / platform_metrics["followers"] * 1000).round(2),
        0.0,
    )

    col1, col2 = st.columns(2)

    with col1:
        st.dataframe(
            platform_metrics.style.format(
                {
                    "followers": "{:,.0f}",
                    "engagements": "{:,.0f}",
                    "reach": "{:,.0f}",
                    "engagement_rate": "{:.2f}%",
                    "sentiment_score": "{:.3f}",
                    "engagement_ratio": "{:.2f}",
                }
            ),
            use_container_width=True,
        )

    with col2:
        # Platform efficiency matrix
        try:
            fig = px.scatter(
                platform_metrics.reset_index(),
                x="engagement_rate",
                y="sentiment_score",
                size="followers",
                color="platform",
                title="Platform Efficiency: Engagement vs Sentiment",
                hover_data=["engagement_ratio"],
            )
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.warning(f"⚠️ Could not render platform efficiency plot: {e}")

    # Engagement trends over time
    st.subheader("📈 Social Engagement Trends")

    try:
        platform_trends = (
            data.groupby(["date", "platform"])["engagements"].sum().reset_index()
        )

        fig = px.line(
            platform_trends,
            x="date",
            y="engagements",
            color="platform",
            title="Daily Engagement Trends by Platform",
            labels={"engagements": "Daily Engagements", "date": "Date"},
        )

        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.warning(f"⚠️ Could not render engagement trends: {e}")

    # 🎯 STRATEGIC INSIGHTS: Social Media Maturity & Investment Strategy
    with st.expander(
        "🎯 **Social Media Maturity & Investment Strategy Framework**", expanded=False
    ):
        st.subheader("📊 Social Media Maturity Assessment")

        maturity_levels = {
            "Basic": ["Organic posting only", "No content strategy", "Reactive engagement"],
            "Intermediate": ["Content calendar", "Basic analytics", "Paid social testing"],
            "Advanced": ["Multi-platform strategy", "ROI measurement", "Influencer partnerships"],
            "Enterprise": ["Predictive analytics", "AI optimization", "Social commerce integration"],
        }

        current_maturity = "Intermediate"
        target_maturity = "Advanced"

        col1, col2 = st.columns(2)

        with col1:
            st.metric("Current Maturity", current_maturity)
            st.markdown("**Current Capabilities:**")
            for capability in maturity_levels[current_maturity]:
                st.markdown(f"✅ {capability}")

        with col2:
            st.metric("Target Maturity", target_maturity)
            st.markdown("**Target Capabilities:**")
            for capability in maturity_levels[target_maturity]:
                st.markdown(f"🎯 {capability}")

        st.divider()

        st.subheader("💰 Social Media ROI Investment Framework")

        roi_framework = {
            "High ROI Platforms": {
                "Platforms": ["Instagram", "TikTok"],
                "Current ROI": "4.2x",
                "Recommended Investment": "Increase 20-25%",
                "Focus Areas": ["Video content", "Influencer collabs", "Social commerce"],
            },
            "Medium ROI Platforms": {
                "Platforms": ["Facebook", "LinkedIn"],
                "Current ROI": "2.8x",
                "Recommended Investment": "Optimize current spend",
                "Focus Areas": ["Community building", "Lead generation", "Brand storytelling"],
            },
            "Low ROI Platforms": {
                "Platforms": ["Twitter"],
                "Current ROI": "1.5x",
                "Recommended Investment": "Reduce 15-20%",
                "Focus Areas": ["Customer service", "Crisis management", "Industry news"],
            },
        }

        for category, strategy in roi_framework.items():
            st.markdown(f"**{category}**")
            cols = st.columns(4)
            cols[0].write(f"**Platforms:** {', '.join(strategy['Platforms'])}")
            cols[1].metric("Current ROI", strategy["Current ROI"])
            cols[2].metric("Investment", strategy["Recommended Investment"])
            cols[3].write(f"**Focus:** {', '.join(strategy['Focus Areas'])}")

        st.divider()

        st.subheader("📈 Expected Business Impact")

        st.markdown(
            """
            **Quantifiable Benefits from Social Media Optimization:**
            
            🚀 **Brand Awareness**: 35-50% increase in reach and impressions  
            💰 **Customer Acquisition**: 25-40% reduction in acquisition cost  
            🎯 **Conversion Rate**: 15-25% lift from social traffic  
            🔧 **Customer Loyalty**: 30-45% improvement in retention rates  
            📊 **Market Intelligence**: Real-time brand sentiment and competitive insights
            """
        )


# ====================================================================
# TAB 2: SENTIMENT ANALYSIS
# ====================================================================


def render_sentiment_analysis(data: pd.DataFrame):
    """Advanced sentiment analysis and brand perception"""

    required_cols = ["sentiment_score", "platform", "date"]
    if not _require_columns(data, required_cols, "Sentiment Analysis"):
        return

    st.header("🎭 Sentiment Analysis & Brand Perception")

    # Sentiment KPIs
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        overall_sentiment = data["sentiment_score"].mean() * 100
        st.metric(
            "Overall Sentiment Score",
            f"{overall_sentiment:.1f}%",
            "3.2%",
            help="Weighted average sentiment across all platforms",
        )

    with col2:
        positive_mentions = 68.5  # Simulated
        st.metric(
            "Positive Mentions",
            f"{positive_mentions:.1f}%",
            "4.1%",
            help="Percentage of positive brand mentions",
        )

    with col3:
        negative_mentions = 8.3  # Simulated
        st.metric(
            "Negative Mentions",
            f"{negative_mentions:.1f}%",
            "-1.7%",
            help="Percentage of negative brand mentions",
        )

    with col4:
        sentiment_volatility = 12.4  # Simulated
        st.metric(
            "Sentiment Volatility",
            f"{sentiment_volatility:.1f}%",
            "-2.3%",
            help="Standard deviation of daily sentiment scores",
        )

    st.divider()

    # Sentiment trends and analysis
    st.subheader("📊 Sentiment Trends & Pattern Analysis")

    try:
        sentiment_trends = (
            data.groupby(["date", "platform"])["sentiment_score"].mean().reset_index()
        )

        fig = px.line(
            sentiment_trends,
            x="date",
            y="sentiment_score",
            color="platform",
            title="Daily Sentiment Trends by Platform",
            labels={"sentiment_score": "Sentiment Score", "date": "Date"},
        )

        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.warning(f"⚠️ Could not render sentiment trend chart: {e}")

    # Sentiment distribution analysis
    st.subheader("📈 Sentiment Distribution & Topic Analysis")

    col1, col2 = st.columns(2)

    with col1:
        try:
            # Sentiment distribution by platform
            platform_sentiment = (
                data.groupby("platform")["sentiment_score"].mean().reset_index()
            )

            fig = px.bar(
                platform_sentiment,
                x="platform",
                y="sentiment_score",
                title="Average Sentiment by Platform",
                color="sentiment_score",
                color_continuous_scale="RdYlGn",
            )
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.warning(f"⚠️ Could not render platform sentiment chart: {e}")

    with col2:
        # Sentiment topic analysis (simulated)
        topics = [
            "Product Quality",
            "Customer Service",
            "Shipping",
            "Pricing",
            "Brand Reputation",
        ]
        topic_sentiment = np.random.uniform(0.6, 0.95, len(topics))
        topic_volume = np.random.randint(100, 1000, len(topics))

        topic_df = pd.DataFrame(
            {
                "Topic": topics,
                "Sentiment Score": topic_sentiment,
                "Mention Volume": topic_volume,
            }
        )

        try:
            fig = px.scatter(
                topic_df,
                x="Sentiment Score",
                y="Mention Volume",
                size="Mention Volume",
                color="Topic",
                title="Topic Sentiment vs Volume Analysis",
                hover_data=["Topic"],
            )

            fig.add_vline(x=0.8, line_dash="dash", line_color="green")
            fig.add_vline(x=0.6, line_dash="dash", line_color="orange")

            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.warning(f"⚠️ Could not render topic sentiment chart: {e}")

    # 🎯 STRATEGIC INSIGHTS: Brand Health & Sentiment Intelligence Framework
    with st.expander(
        "🎯 **Brand Health & Sentiment Intelligence Framework**", expanded=False
    ):
        st.subheader("📊 Brand Sentiment Maturity Model")

        sentiment_maturity = {
            "Reactive": [
                "Manual sentiment tracking",
                "Basic keyword monitoring",
                "Crisis response only",
            ],
            "Proactive": [
                "Automated sentiment analysis",
                "Topic modeling",
                "Early warning systems",
            ],
            "Predictive": [
                "Sentiment forecasting",
                "Influencer impact prediction",
                "Trend anticipation",
            ],
            "Prescriptive": [
                "AI-driven recommendations",
                "Automated response systems",
                "Sentiment optimization",
            ],
        }

        current_level = "Proactive"
        target_level = "Predictive"

        col1, col2 = st.columns(2)

        with col1:
            st.metric("Current Level", current_level)
            st.markdown("**Current Capabilities:**")
            for capability in sentiment_maturity[current_level]:
                st.markdown(f"✅ {capability}")

        with col2:
            st.metric("Target Level", target_level)
            st.markdown("**Target Capabilities:**")
            for capability in sentiment_maturity[target_level]:
                st.markdown(f"🎯 {capability}")

        st.divider()

        st.subheader("🚨 Crisis Detection & Management Framework")

        crisis_framework = {
            "Early Warning Signals": [
                "Sudden sentiment drop > 15%",
                "Negative mention volume spike > 50%",
                "Influencer negative sentiment",
                "Competitive attack campaigns",
            ],
            "Immediate Response Protocol": [
                "Activate crisis response team within 1 hour",
                "Pause scheduled social content",
                "Prepare holding statement",
                "Monitor sentiment in real time",
            ],
            "Recovery Strategy": [
                "Transparent communication",
                "Customer outreach program",
                "Positive sentiment amplification",
                "Post-crisis analysis and learning",
            ],
        }

        for phase, actions in crisis_framework.items():
            st.markdown(f"**{phase}:**")
            for action in actions:
                st.markdown(f"- {action}")

        st.divider()

        st.subheader("🎯 Sentiment-Driven Business Strategy")

        sentiment_strategy = {
            "Product Development": "Use positive sentiment to guide feature development",
            "Customer Service": "Address negative sentiment patterns systematically",
            "Marketing Strategy": "Amplify positive sentiment in campaigns",
            "Competitive Positioning": "Leverage sentiment advantages in messaging",
            "Crisis Prevention": "Build sentiment buffers through community engagement",
        }

        for area, strategy in sentiment_strategy.items():
            st.metric(area, strategy)

        st.divider()

        st.subheader("📈 Sentiment Impact on Business Metrics")

        impact_data = {
            "Metric": [
                "Customer Retention",
                "Purchase Intent",
                "Brand Loyalty",
                "Word-of-Mouth",
                "Price Premium",
            ],
            "Positive Sentiment Impact": ["+35%", "+28%", "+42%", "+55%", "+18%"],
            "Negative Sentiment Impact": ["-45%", "-52%", "-38%", "-65%", "-25%"],
        }

        impact_df = pd.DataFrame(impact_data)
        st.dataframe(impact_df, use_container_width=True)


# ====================================================================
# TAB 3: COMPETITIVE INTELLIGENCE
# ====================================================================


def render_competitive_intelligence(
    social_data: pd.DataFrame, competitive_data: pd.DataFrame
):
    """Competitive social media landscape analysis"""

    if competitive_data.empty:
        st.error(
            "❌ Competitive data is not available. Please ensure "
            "`digital_data['competitive']` is populated."
        )
        return

    required_comp_cols = [
        "competitor",
        "market_share",
        "social_mentions",
        "review_rating",
        "price_index",
    ]
    if not _require_columns(
        competitive_data, required_comp_cols, "Competitive Intelligence (competitive_data)"
    ):
        return

    if not _require_columns(
        social_data, ["engagements", "sentiment_score"], "Competitive Intelligence (social_data)"
    ):
        return

    st.header("🔍 Competitive Social Intelligence")

    # Competitive KPIs
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        market_share_rank = 2  # Simulated
        st.metric(
            "Market Share Rank",
            f"#{market_share_rank}",
            "+1",
            help="Your position in market share ranking",
        )

    with col2:
        social_share_voice = 28.4  # Simulated
        st.metric(
            "Share of Voice",
            f"{social_share_voice:.1f}%",
            "3.2%",
            help="Percentage of total industry social mentions",
        )

    with col3:
        engagement_leadership = 1  # Simulated
        st.metric(
            "Engagement Leadership",
            f"#{engagement_leadership}",
            "0",
            help="Rank in engagement rate among competitors",
        )

    with col4:
        sentiment_leadership = 2  # Simulated
        st.metric(
            "Sentiment Leadership",
            f"#{sentiment_leadership}",
            "+1",
            help="Rank in sentiment score among competitors",
        )

    st.divider()

    # Competitive landscape analysis
    st.subheader("🏆 Competitive Landscape Dashboard")

    competitive_analysis = (
        competitive_data.groupby("competitor")
        .agg(
            {
                "market_share": "mean",
                "social_mentions": "sum",
                "review_rating": "mean",
                "price_index": "mean",
            }
        )
        .reset_index()
    )

    # Add your company data (simulated)
    your_company = {
        "competitor": "Your Brand",
        "market_share": 18.5,
        "social_mentions": social_data["engagements"].sum() / 10,  # Simulated scaling
        "review_rating": social_data["sentiment_score"].mean() * 5,  # Convert to 5-star
        "price_index": 1.0,
    }

    competitive_analysis = pd.concat(
        [competitive_analysis, pd.DataFrame([your_company])], ignore_index=True
    )

    col1, col2 = st.columns(2)

    with col1:
        try:
            fig = px.bar(
                competitive_analysis,
                x="competitor",
                y="market_share",
                title="Market Share Comparison",
                color="market_share",
                color_continuous_scale="Blues",
            )
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.warning(f"⚠️ Could not render market share chart: {e}")

    with col2:
        try:
            fig = px.scatter(
                competitive_analysis,
                x="social_mentions",
                y="review_rating",
                size="market_share",
                color="competitor",
                title="Social Presence vs Customer Satisfaction",
                hover_data=["price_index"],
            )
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.warning(f"⚠️ Could not render presence vs rating chart: {e}")

    # Competitive positioning matrix
    st.subheader("🎯 Competitive Positioning Analysis")

    positioning_data = competitive_analysis.copy()
    positioning_data["social_engagement"] = np.random.uniform(2, 8, len(positioning_data))
    positioning_data["content_quality"] = np.random.uniform(3, 9, len(positioning_data))

    try:
        fig = px.scatter(
            positioning_data,
            x="social_engagement",
            y="content_quality",
            size="market_share",
            color="competitor",
            title="Competitive Positioning: Engagement vs Content Quality",
            hover_data=["review_rating"],
        )

        # Add quadrant lines
        fig.add_hline(y=6, line_dash="dash", line_color="red")
        fig.add_vline(x=5, line_dash="dash", line_color="red")

        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.warning(f"⚠️ Could not render positioning matrix: {e}")

    # 🎯 STRATEGIC INSIGHTS: Competitive Social Strategy Framework
    with st.expander(
        "🎯 **Competitive Social Strategy Framework**", expanded=False
    ):
        st.subheader("📊 Competitive Social SWOT Analysis")

        swot_analysis = {
            "Strengths": [
                "Higher engagement rates than competitors",
                "Strong positive sentiment leadership",
                "Effective influencer partnerships",
                "Superior content quality scores",
            ],
            "Weaknesses": [
                "Lower share of voice in key segments",
                "Limited platform diversity vs competitors",
                "Slower response time to negative sentiment",
                "Underutilized social commerce opportunities",
            ],
            "Opportunities": [
                "Expand into competitor weak platforms",
                "Leverage sentiment advantage in messaging",
                "Develop social commerce capabilities",
                "Build community in underserved segments",
            ],
            "Threats": [
                "Competitor aggressive social spending",
                "Emerging platforms disrupting engagement",
                "Regulatory changes impacting social ads",
                "Market saturation in core segments",
            ],
        }

        cols = st.columns(4)
        for idx, (category, items) in enumerate(swot_analysis.items()):
            with cols[idx]:
                st.markdown(f"**{category}**")
                for item in items:
                    st.markdown(f"- {item}")

        st.divider()

        st.subheader("🎯 Competitive Attack & Defense Strategy")

        competitive_strategies = {
            "Offensive Strategies": [
                "Target competitor weak platforms with superior content",
                "Amplify sentiment advantages in competitive messaging",
                "Poach competitor influencers with better partnerships",
                "Create comparative content highlighting advantages",
            ],
            "Defensive Strategies": [
                "Strengthen community engagement to build loyalty",
                "Improve response time to negative sentiment",
                "Diversify platform presence to reduce risk",
                "Build social commerce as defensive moat",
            ],
            "Flanking Strategies": [
                "Focus on emerging platforms before competitors",
                "Develop niche community segments",
                "Create unique content formats competitors can't match",
                "Build proprietary social listening capabilities",
            ],
        }

        for strategy_type, tactics in competitive_strategies.items():
            st.markdown(f"**{strategy_type}:**")
            for tactic in tactics:
                st.markdown(f"- {tactic}")

        st.divider()

        st.subheader("📈 Competitive Gap Analysis & Opportunity Sizing")

        gap_analysis = {
            "Platform Gaps": {
                "Your Coverage": "Facebook, Instagram, Twitter",
                "Competitor Coverage": "Facebook, Instagram, Twitter, TikTok, LinkedIn",
                "Opportunity Size": "$2.8M annual revenue potential",
                "Investment Required": "$450K platform expansion",
            },
            "Content Gaps": {
                "Your Coverage": "Images 60%, Video 25%, Text 15%",
                "Competitor Coverage": "Video 45%, Images 35%, Interactive 20%",
                "Opportunity Size": "35% engagement rate improvement",
                "Investment Required": "$320K content production",
            },
            "Engagement Gaps": {
                "Your Coverage": "4.2 hours average response time",
                "Competitor Coverage": "1.8 hours average response time",
                "Opportunity Size": "28% sentiment improvement",
                "Investment Required": "$180K community management",
            },
        }

        for gap_type, analysis in gap_analysis.items():
            st.markdown(f"**{gap_type}**")
            cols = st.columns(4)

            # Safe lookups with sensible fallbacks
            your_pos = analysis.get("Your Coverage") or analysis.get("Your Position") or "-"
            comp_pos = (
                analysis.get("Competitor Coverage")
                or analysis.get("Competitor Position")
                or "-"
            )
            opp = analysis.get("Opportunity Size") or analysis.get("Opportunity") or "-"
            inv = (
                analysis.get("Investment Required")
                or analysis.get("Investment")
                or "-"
            )

            cols[0].metric("Your Position", your_pos)
            cols[1].metric("Competitor Position", comp_pos)
            cols[2].metric("Opportunity", opp)
            cols[3].metric("Investment", inv)


# ====================================================================
# TAB 4: CAMPAIGN ANALYTICS
# ====================================================================


def render_campaign_analytics(data: pd.DataFrame):
    """Social media campaign performance and ROI analysis"""

    st.header("🚀 Campaign Performance & ROI Analytics")

    # Campaign KPIs
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        campaign_roi = 3.8  # Simulated
        st.metric(
            "Campaign ROI",
            f"{campaign_roi:.1f}x",
            "0.4x",
            help="Return on investment for social campaigns",
        )

    with col2:
        cpm = 8.45  # Simulated
        st.metric(
            "Avg. CPM",
            f"${cpm:.2f}",
            "-$1.20",
            help="Cost per thousand impressions",
        )

    with col3:
        engagement_cost = 0.18  # Simulated
        st.metric(
            "Cost per Engagement",
            f"${engagement_cost:.2f}",
            "-$0.03",
            help="Average cost per engagement",
        )

    with col4:
        conversion_rate = 4.2  # Simulated
        st.metric(
            "Social Conversion Rate",
            f"{conversion_rate:.1f}%",
            "0.8%",
            help="Conversions from social media campaigns",
        )

    st.divider()

    # Campaign performance analysis
    st.subheader("📊 Multi-Campaign Performance Comparison")

    # Simulated campaign data
    campaigns = [
        "Q4 Product Launch",
        "Holiday Sale",
        "Brand Awareness",
        "Influencer Partnership",
    ]
    campaign_data = []

    for campaign in campaigns:
        campaign_data.append(
            {
                "Campaign": campaign,
                "Impressions": np.random.randint(50_000, 500_000),
                "Engagements": np.random.randint(5_000, 50_000),
                "Conversions": np.random.randint(200, 2_000),
                "Spend": np.random.randint(5_000, 50_000),
                "ROI": np.random.uniform(2.0, 6.0),
            }
        )

    campaign_df = pd.DataFrame(campaign_data)
    campaign_df["CPM"] = (campaign_df["Spend"] / campaign_df["Impressions"] * 1000).round(2)
    campaign_df["Cost per Conversion"] = (
        campaign_df["Spend"] / campaign_df["Conversions"]
    ).round(2)

    col1, col2 = st.columns(2)

    with col1:
        st.dataframe(
            campaign_df.style.format(
                {
                    "Impressions": "{:,.0f}",
                    "Engagements": "{:,.0f}",
                    "Conversions": "{:,.0f}",
                    "Spend": "${:,.0f}",
                    "ROI": "{:.1f}x",
                    "CPM": "${:.2f}",
                    "Cost per Conversion": "${:.2f}",
                }
            ),
            use_container_width=True,
        )

    with col2:
        try:
            fig = px.bar(
                campaign_df,
                x="Campaign",
                y="ROI",
                title="Campaign ROI Comparison",
                color="ROI",
                color_continuous_scale="Viridis",
            )
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.warning(f"⚠️ Could not render campaign ROI chart: {e}")

    # 🎯 STRATEGIC INSIGHTS: Social Commerce & Campaign Optimization Framework
    with st.expander(
        "🎯 **Social Commerce & Campaign Optimization Framework**", expanded=False
    ):
        st.subheader("💰 Social Commerce ROI Framework")

        social_commerce_metrics = {
            "Social-Driven Revenue": "$2.8M annual",
            "Social Conversion Rate": "3.2%",
            "Average Order Value": "$85",
            "Customer Acquisition Cost": "$22",
            "Lifetime Value": "$340",
        }

        cols = st.columns(5)
        for idx, (metric, value) in enumerate(social_commerce_metrics.items()):
            cols[idx].metric(metric, value)

        st.divider()

        st.subheader("🎯 Campaign Optimization Investment Framework")

        optimization_framework = {
            "Content Strategy": {
                "Current Performance": "Medium",
                "Recommended Investment": "+25%",
                "Focus Areas": [
                    "Video production",
                    "User-generated content",
                    "Interactive formats",
                ],
                "Expected ROI": "4.2x",
            },
            "Influencer Partnerships": {
                "Current Performance": "High",
                "Recommended Investment": "+35%",
                "Focus Areas": [
                    "Micro-influencers",
                    "Category experts",
                    "Brand ambassadors",
                ],
                "Expected ROI": "5.8x",
            },
            "Paid Social": {
                "Current Performance": "Low",
                "Recommended Investment": "-15%",
                "Focus Areas": [
                    "Audience targeting",
                    "Creative optimization",
                    "Bidding strategy",
                ],
                "Expected ROI": "2.1x",
            },
            "Social Commerce": {
                "Current Performance": "Emerging",
                "Recommended Investment": "+50%",
                "Focus Areas": [
                    "Shoppable posts",
                    "Live shopping",
                    "Social storefront",
                ],
                "Expected ROI": "6.5x",
            },
        }

        for area, strategy in optimization_framework.items():
            st.markdown(f"**{area}**")
            cols = st.columns(4)
            cols[0].metric("Performance", strategy["Current Performance"])
            cols[1].metric("Investment", strategy["Recommended Investment"])
            cols[2].write(f"**Focus:** {', '.join(strategy['Focus Areas'])}")
            cols[3].metric("Expected ROI", strategy["Expected ROI"])

        st.divider()

        st.subheader("📈 Social Media Investment Portfolio Strategy")

        investment_portfolio = {
            "High Growth / High ROI": [
                "Influencer Marketing",
                "Social Commerce",
                "Video Content",
            ],
            "High Growth / Medium ROI": [
                "Community Building",
                "Content Marketing",
                "Live Streaming",
            ],
            "Medium Growth / High ROI": [
                "Paid Social",
                "Retargeting",
                "Lead Generation",
            ],
            "Maintenance / Low ROI": [
                "Organic Posting",
                "Customer Service",
                "Brand Monitoring",
            ],
        }

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**🎯 Investment Allocation Strategy:**")
            for category, areas in investment_portfolio.items():
                st.markdown(f"**{category}:**")
                for area in areas:
                    st.markdown(f"- {area}")

        with col2:
            st.markdown("**📊 Expected Portfolio Returns:**")
            st.metric("Overall ROI Target", "4.8x")
            st.metric("Revenue Growth Target", "+35%")
            st.metric("Customer Acquisition Target", "+28%")
            st.metric("Brand Equity Growth", "+42%")

        st.divider()

        st.subheader("🚀 Advanced Social Media Capabilities Roadmap")

        capability_roadmap = {
            "Phase 1 (30 days)": [
                "Implement social commerce features",
                "Launch influencer partnership program",
                "Optimize paid social bidding strategy",
            ],
            "Phase 2 (60 days)": [
                "Develop AI-powered content optimization",
                "Build advanced social listening",
                "Implement cross-platform analytics",
            ],
            "Phase 3 (90 days)": [
                "Launch predictive campaign performance",
                "Develop social CRM integration",
                "Build automated optimization engine",
            ],
            "Phase 4 (6+ months)": [
                "AI-driven content generation",
                "Predictive sentiment analysis",
                "Automated social commerce optimization",
            ],
        }

        for phase, initiatives in capability_roadmap.items():
            st.markdown(f"**{phase}:**")
            for initiative in initiatives:
                st.markdown(f"- {initiative}")


if __name__ == "__main__":
    render()
