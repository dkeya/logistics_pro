# logistics_pro/scripts/setup_environment.py
import os
import subprocess
import sys

def create_project_structure():
    """Create the complete Logistics Pro project structure"""
    
    base_dir = "logistics_pro"
    
    # Define the folder structure
    structure = [
        f"{base_dir}/pages",
        f"{base_dir}/logistics_core/schemas",
        f"{base_dir}/logistics_core/analytics",
        f"{base_dir}/logistics_core/connectors", 
        f"{base_dir}/logistics_core/policy",
        f"{base_dir}/data/raw",
        f"{base_dir}/data/processed",
        f"{base_dir}/data/warehouse",
        f"{base_dir}/data/exports",
        f"{base_dir}/assets/images",
        f"{base_dir}/assets/css",
        f"{base_dir}/assets/templates",
        f"{base_dir}/configs",
        f"{base_dir}/scripts",
        f"{base_dir}/tests",
        f"{base_dir}/logs",
    ]
    
    # Create all directories
    for folder in structure:
        os.makedirs(folder, exist=True, exist_ok=True)
        print(f"Created: {folder}")
    
    # Create __init__.py files
    init_files = [
        f"{base_dir}/__init__.py",
        f"{base_dir}/logistics_core/__init__.py",
        f"{base_dir}/logistics_core/schemas/__init__.py",
        f"{base_dir}/logistics_core/analytics/__init__.py",
        f"{base_dir}/logistics_core/connectors/__init__.py",
        f"{base_dir}/logistics_core/policy/__init__.py",
    ]
    
    for init_file in init_files:
        with open(init_file, 'w') as f:
            f.write('"""Module initialization"""\n')
        print(f"Created: {init_file}")
    
    print(f"\n✅ Project structure created successfully in '{base_dir}/'")

def install_dependencies():
    """Install required Python packages"""
    print("\n📦 Installing dependencies...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Dependencies installed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        # Install manually
        packages = [
            "streamlit==1.28.0",
            "pandas==2.0.3", 
            "numpy==1.24.3",
            "plotly==5.15.0",
            "scikit-learn==1.3.0",
            "openpyxl==3.1.2"
        ]
        for package in packages:
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", package])
                print(f"✅ Installed: {package}")
            except:
                print(f"❌ Failed to install: {package}")

if __name__ == "__main__":
    create_project_structure()
    install_dependencies()
    print("\n🎉 Setup completed! You can now run:")
    print("   cd logistics_pro")
    print("   streamlit run app.py")