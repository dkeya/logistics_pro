import os
import sys
from pathlib import Path

def create_directory_structure():
    """Create the complete directory structure for Logistics Pro"""
    
    base_dir = Path(__file__).parent.parent
    
    # Main directories
    directories = [
        # Core application
        "logistics_core/analytics",
        "logistics_core/schemas",
        "logistics_core/connectors",
        "logistics_core/policy",
        
        # Pages
        "pages",
        
        # Data directories
        "data/raw",
        "data/processed",
        "data/warehouse",
        "data/exports",
        "data/tenants/elora_holding",
        "data/tenants/naivas_supermarkets",
        "data/tenants/quickmart_kenya",
        "data/tenants/chandarana_foodplus",
        
        # Configurations
        "configs",
        
        # Assets
        "assets/images",
        "assets/css",
        "assets/templates",
        
        # Scripts
        "scripts",
        
        # Tests
        "tests",
        
        # Logs
        "logs/application",
        "logs/tenant/elora_holding",
        "logs/tenant/naivas_supermarkets",
        "logs/tenant/quickmart_kenya",
        "logs/tenant/chandarana_foodplus",
        "logs/audit"
    ]
    
    # Create directories
    for directory in directories:
        dir_path = base_dir / directory
        dir_path.mkdir(parents=True, exist_ok=True)
        print(f"✅ Created: {directory}")
        
        # Create __init__.py files for Python packages
        if directory.startswith("logistics_core") and not directory.endswith("__pycache__"):
            init_file = dir_path / "__init__.py"
            init_file.touch(exist_ok=True)
    
    # Create placeholder files
    placeholder_files = [
        "data/raw/.gitkeep",
        "data/processed/.gitkeep",
        "data/warehouse/.gitkeep",
        "data/exports/.gitkeep",
        "assets/images/.gitkeep",
        "assets/css/.gitkeep",
        "assets/templates/.gitkeep",
        "logs/application/.gitkeep",
        "logs/audit/.gitkeep"
    ]
    
    for file_path in placeholder_files:
        file = base_dir / file_path
        file.parent.mkdir(parents=True, exist_ok=True)
        file.touch(exist_ok=True)
        print(f"✅ Created: {file_path}")
    
    print("\n🎉 Directory structure initialized successfully!")
    print("📁 Project structure ready for development.")

def create_sample_data():
    """Create sample data files for demonstration"""
    base_dir = Path(__file__).parent.parent
    
    # Sample configuration files
    sample_files = {
        "README.md": """# Logistics Pro - FMCG Intelligence Platform
        
## 🚀 Project Overview
Enterprise multi-tenant analytics platform for FMCG distribution operations.

## 📁 Project Structure
- `app.py` - Main application
- `pages/` - Streamlit multi-page modules
- `logistics_core/` - Core business logic
- `data/` - Data storage and management
- `configs/` - Configuration files

## 🏢 Supported Tenants
- ELORA Holding
- Naivas Supermarkets
- QuickMart Kenya
- Chandarana FoodPlus

## 🚀 Getting Started
1. Run `python scripts/initialize_system.py`
2. Install dependencies: `pip install -r requirements.txt`
3. Launch app: `streamlit run app.py`
""",
        
        ".env.example": """# Database Configuration
DB_HOST=localhost
DB_PORT=5432
DB_NAME=logistics_pro
DB_USER=username
DB_PASSWORD=password

# Application Settings
DEBUG=True
LOG_LEVEL=INFO

# API Keys
MAPS_API_KEY=your_maps_api_key
ML_API_KEY=your_ml_service_key
"""
    }
    
    for filename, content in sample_files.items():
        file_path = base_dir / filename
        file_path.write_text(content)
        print(f"✅ Created: {filename}")

if __name__ == "__main__":
    print("🚀 Initializing Logistics Pro System...")
    create_directory_structure()
    create_sample_data()
    print("\n🎉 System initialization complete!")
    print("📋 Next steps:")
    print("   1. Install dependencies: pip install -r requirements.txt")
    print("   2. Launch the application: streamlit run app.py")
    print("   3. Access the platform at http://localhost:8501")