# scripts/initialize_digital_hub.py
import os
import sys

def create_digital_intelligence_structure():
    """Create the digital intelligence hub folder structure with proper encoding"""
    
    base_path = "pages/digital_intelligence"
    
    # Create main digital intelligence directory
    os.makedirs(base_path, exist_ok=True)
    
    # Create empty module files with proper encoding
    modules = [
        "22_Digital_Overview.py",
        "23_Ecommerce_Analytics.py", 
        "24_Web_Analytics.py",
        "25_Social_Media_Intel.py",
        "26_Digital_Operations.py"
    ]
    
    for module in modules:
        file_path = os.path.join(base_path, module)
        if not os.path.exists(file_path):
            with open(file_path, 'w', encoding='utf-8') as f:  # ✅ FIX: Added encoding
                module_name = module.replace(".py", "").replace("_", " ")
                f.write(f'# {module_name} - Digital Intelligence Hub\n')
                f.write('import streamlit as st\n\n')
                f.write('def render():\n')
                f.write(f'    st.title("{module_name}")\n')
                f.write('    st.info("Module under development - Digital Intelligence Hub")\n')
                f.write('    st.write("This module is part of the consolidated digital analytics platform.")\n\n')
                f.write('if __name__ == "__main__":\n')
                f.write('    render()\n')
    
    # Create digital analytics core with proper encoding
    analytics_path = "logistics_core/analytics"
    os.makedirs(analytics_path, exist_ok=True)
    
    digital_analytics_file = os.path.join(analytics_path, "digital_analytics.py")
    if not os.path.exists(digital_analytics_file):
        with open(digital_analytics_file, 'w', encoding='utf-8') as f:  # ✅ FIX: Added encoding
            f.write('# Digital Analytics Engine - Core digital intelligence\n')
            f.write('# This module handles ecommerce, web analytics, and social media data\n')
            f.write('import pandas as pd\nimport numpy as np\n\n')
            f.write('class DigitalAnalyticsEngine:\n')
            f.write('    """Unified digital analytics engine"""\n')
            f.write('    def __init__(self):\n')
            f.write('        self.ecommerce_data = None\n')
            f.write('        self.web_analytics_data = None\n')
            f.write('        self.social_data = None\n')
    
    print("✅ Digital Intelligence Hub structure created successfully!")
    print("📁 Created folders:")
    print(f"   - {base_path}/")
    for module in modules:
        print(f"   - {base_path}/{module}")

# Also create the digital analytics engine file separately to avoid encoding issues
def create_digital_analytics_engine():
    """Create the main digital analytics engine with proper encoding"""
    
    digital_analytics_content = '''# logistics_core/analytics/digital_analytics.py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List, Optional

class DigitalAnalyticsEngine:
    """Unified digital analytics engine for ecommerce, web, and social data"""
    
    def __init__(self):
        self.ecommerce_data = None
        self.web_analytics_data = None
        self.social_data = None
        self.competitive_data = None
        
    def generate_synthetic_digital_data(self, start_date: str, end_date: str) -> Dict:
        """Generate comprehensive synthetic digital data"""
        
        # Date range
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        
        # Ecommerce data
        ecommerce_data = self._generate_ecommerce_data(dates)
        
        # Web analytics data
        web_data = self._generate_web_analytics_data(dates)
        
        # Social media data
        social_data = self._generate_social_media_data(dates)
        
        # Competitive intelligence
        competitive_data = self._generate_competitive_data(dates)
        
        return {
            'ecommerce': ecommerce_data,
            'web_analytics': web_data,
            'social_media': social_data,
            'competitive': competitive_data
        }
    
    def _generate_ecommerce_data(self, dates):
        """Generate multi-platform ecommerce data"""
        platforms = ['Amazon', 'Shopify', 'WooCommerce', 'eBay', 'Facebook Marketplace']
        products = ['Premium Coffee', 'Organic Tea', 'Energy Bars', 'Snack Mix', 'Health Supplements']
        
        data = []
        for date in dates:
            for platform in platforms:
                for product in products:
                    orders = np.random.poisson(15)
                    revenue = orders * np.random.uniform(20, 100)
                    visitors = np.random.poisson(200)
                    conversion_rate = np.random.uniform(1.5, 4.5)
                    
                    data.append({
                        'date': date,
                        'platform': platform,
                        'product': product,
                        'orders': orders,
                        'revenue': revenue,
                        'visitors': visitors,
                        'conversion_rate': conversion_rate,
                        'aov': revenue / orders if orders > 0 else 0
                    })
        
        return pd.DataFrame(data)
    
    def _generate_web_analytics_data(self, dates):
        """Generate web analytics data"""
        channels = ['Organic Search', 'Paid Search', 'Social Media', 'Email', 'Direct', 'Referral']
        devices = ['Desktop', 'Mobile', 'Tablet']
        
        data = []
        for date in dates:
            for channel in channels:
                for device in devices:
                    sessions = np.random.poisson(500)
                    users = np.random.poisson(450)
                    pageviews = np.random.poisson(1200)
                    bounce_rate = np.random.uniform(35, 65)
                    avg_session_duration = np.random.uniform(90, 300)
                    
                    data.append({
                        'date': date,
                        'channel': channel,
                        'device': device,
                        'sessions': sessions,
                        'users': users,
                        'pageviews': pageviews,
                        'bounce_rate': bounce_rate,
                        'avg_session_duration': avg_session_duration
                    })
        
        return pd.DataFrame(data)
    
    def _generate_social_media_data(self, dates):
        """Generate social media engagement data"""
        platforms = ['Facebook', 'Instagram', 'Twitter', 'LinkedIn', 'TikTok']
        
        data = []
        for date in dates:
            for platform in platforms:
                followers = np.random.poisson(15000)
                engagements = np.random.poisson(500)
                reach = np.random.poisson(25000)
                sentiment = np.random.uniform(0.6, 0.95)  # Positive sentiment score
                
                data.append({
                    'date': date,
                    'platform': platform,
                    'followers': followers,
                    'engagements': engagements,
                    'reach': reach,
                    'sentiment_score': sentiment,
                    'engagement_rate': (engagements / followers) * 100 if followers > 0 else 0
                })
        
        return pd.DataFrame(data)
    
    def _generate_competitive_data(self, dates):
        """Generate competitive intelligence data"""
        competitors = ['Competitor A', 'Competitor B', 'Competitor C', 'Market Leader']
        
        data = []
        for date in dates:
            for competitor in competitors:
                market_share = np.random.uniform(5, 25)
                price_index = np.random.uniform(0.8, 1.2)
                social_mentions = np.random.poisson(200)
                review_rating = np.random.uniform(3.5, 4.8)
                
                data.append({
                    'date': date,
                    'competitor': competitor,
                    'market_share': market_share,
                    'price_index': price_index,
                    'social_mentions': social_mentions,
                    'review_rating': review_rating
                })
        
        return pd.DataFrame(data)
'''

    analytics_path = "logistics_core/analytics"
    os.makedirs(analytics_path, exist_ok=True)
    
    digital_analytics_file = os.path.join(analytics_path, "digital_analytics.py")
    with open(digital_analytics_file, 'w', encoding='utf-8') as f:
        f.write(digital_analytics_content)
    
    print("✅ Digital Analytics Engine created successfully!")

if __name__ == "__main__":
    create_digital_intelligence_structure()
    create_digital_analytics_engine()