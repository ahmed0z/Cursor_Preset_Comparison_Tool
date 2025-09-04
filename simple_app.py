"""
Simplified Streamlit app for testing.
"""

import streamlit as st
import pandas as pd
import sys
import os

# Add current directory to path
sys.path.append(os.getcwd())

# Page configuration
st.set_page_config(
    page_title="Value Comparison Tool - Test",
    page_icon="🔍",
    layout="wide"
)

def main():
    """Simplified main function."""
    
    st.title("🔍 Value Comparison Tool - Test Version")
    
    st.write("This is a simplified version to test basic functionality.")
    
    # Test basic functionality
    if st.button("Test Data Loading"):
        try:
            from data_processor import DataProcessor
            processor = DataProcessor()
            data = processor.load_preset_data_auto()
            
            if data is not None:
                st.success(f"✅ Database loaded successfully: {len(data):,} records")
                st.write("Sample data:")
                st.dataframe(data.head(10))
            else:
                st.warning("⚠️ No database found")
                
        except Exception as e:
            st.error(f"❌ Error: {e}")
            st.code(str(e))
    
    # Test matching engine
    if st.button("Test Matching Engine"):
        try:
            from data_processor import DataProcessor
            from matching_engine import MatchingEngine
            
            processor = DataProcessor()
            data = processor.load_preset_data_auto()
            
            if data is not None:
                engine = MatchingEngine(data)
                st.success("✅ Matching engine created successfully")
                
                # Test with sample data
                test_data = pd.DataFrame({
                    'Category': ['Switches'],
                    'Sub-Category': ['Push Button'],
                    'Attribute Name': ['Voltage'],
                    'Input Value': ['24V']
                })
                
                results = engine.compare_values(test_data)
                st.write("Test results:")
                st.dataframe(results)
            else:
                st.warning("⚠️ No database available for testing")
                
        except Exception as e:
            st.error(f"❌ Error: {e}")
            st.code(str(e))
    
    # Show system info
    st.sidebar.title("System Info")
    st.sidebar.write(f"Python version: {sys.version}")
    st.sidebar.write(f"Streamlit version: {st.__version__}")
    st.sidebar.write(f"Pandas version: {pd.__version__}")
    
    # Check files
    st.sidebar.title("File Status")
    files_to_check = [
        "Preset 25.xlsx",
        "preset_database.pkl",
        "app.py",
        "data_processor.py",
        "matching_engine.py"
    ]
    
    for file in files_to_check:
        if os.path.exists(file):
            size = os.path.getsize(file) / (1024 * 1024)  # MB
            st.sidebar.success(f"✅ {file} ({size:.1f} MB)")
        else:
            st.sidebar.error(f"❌ {file} not found")

if __name__ == "__main__":
    main()