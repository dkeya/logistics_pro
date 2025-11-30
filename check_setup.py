# check_setup.py
import os
import sys

def check_setup():
    print("🔍 Checking Logistics Pro Setup...")
    print("=" * 50)
    
    # Check current directory
    current_dir = os.getcwd()
    print(f"📁 Current directory: {current_dir}")
    
    # Check required files
    required_files = [
        "app.py",
        "pages/01_Dashboard.py", 
        "logistics_core/__init__.py",
        "logistics_core/analytics/__init__.py",
        "logistics_core/analytics/forecasting.py",
        "logistics_core/analytics/optimization.py",
        "logistics_core/connectors/__init__.py",
        "logistics_core/connectors/data_generator.py"
    ]
    
    all_files_exist = True
    for file in required_files:
        if os.path.exists(file):
            print(f"✅ {file}")
        else:
            print(f"❌ {file}")
            all_files_exist = False
    
    print("=" * 50)
    if all_files_exist:
        print("🎉 All files found! Run: streamlit run app.py")
    else:
        print("❌ Missing files. Please check your project structure.")
    
    return all_files_exist

if __name__ == "__main__":
    check_setup()