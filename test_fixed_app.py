"""
Test app to verify the fixes work correctly.
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
    """Test the fixed matching logic."""
    
    st.title("🔍 Value Comparison Tool - Fixed Version Test")
    
    st.write("Testing the fixed matching logic to ensure:")
    st.write("1. ✅ No app crashes")
    st.write("2. ✅ Matches only within the same composite key")
    st.write("3. ✅ Cross-category matching only when no same-key match")
    
    # Test basic functionality
    if st.button("Test Fixed Matching Logic"):
        try:
            from data_processor import DataProcessor
            from matching_engine import MatchingEngine
            
            # Load data
            processor = DataProcessor()
            data = processor.load_preset_data_auto()
            
            if data is not None:
                st.success(f"✅ Database loaded: {len(data):,} records")
                
                # Create matching engine
                engine = MatchingEngine(data)
                st.success("✅ Matching engine created successfully")
                
                # Test with sample data
                test_data = pd.DataFrame({
                    'Category': ['Switches'],
                    'Sub-Category': ['Push Button'],
                    'Attribute Name': ['Voltage Rating'],
                    'Input Value': ['24V']
                })
                
                st.write("**Testing with input: 24V for Switches|Push Button|Voltage Rating**")
                
                results = engine.compare_values(test_data, similarity_threshold=75.0, max_matches=3)
                
                if len(results) > 0:
                    st.write("**Results:**")
                    for i, (_, result) in enumerate(results.iterrows()):
                        actual_key = f'{result["Category"]}|{result["Sub-Category"]}|{result["Attribute Name"]}'
                        expected_key = "Switches|Push Button|Voltage Rating"
                        
                        # Check if key matches
                        key_status = "✅" if actual_key == expected_key else "❌"
                        
                        st.write(f"{i+1}. {key_status} **Match:** \"{result['Matched Preset Value']}\"")
                        st.write(f"   **Key:** {actual_key}")
                        st.write(f"   **Score:** {result['Similarity %']:.1f}%")
                        st.write(f"   **Comment:** {result['Comment']}")
                        st.write("---")
                    
                    # Check if all results are from the correct key
                    all_correct = all(
                        f'{r["Category"]}|{r["Sub-Category"]}|{r["Attribute Name"]}' == expected_key 
                        for _, r in results.iterrows()
                    )
                    
                    if all_correct:
                        st.success("🎉 **SUCCESS:** All matches are from the correct composite key!")
                    else:
                        st.error("❌ **ISSUE:** Some matches are from different keys!")
                else:
                    st.warning("No matches found")
                
            else:
                st.error("❌ Could not load database")
                
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