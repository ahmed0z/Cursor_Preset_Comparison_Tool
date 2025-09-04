"""
Minimal Streamlit app to test basic functionality.
"""

import streamlit as st

st.set_page_config(
    page_title="Value Comparison Tool",
    page_icon="🔍",
    layout="wide"
)

st.title("🔍 Value Comparison Tool")

st.write("This is a minimal version to test if Streamlit is working.")

if st.button("Test Button"):
    st.success("✅ Streamlit is working correctly!")

st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Choose a page:", ["Home", "Test"])

if page == "Home":
    st.write("Welcome to the Value Comparison Tool!")
elif page == "Test":
    st.write("This is the test page.")
    
    # Test imports
    try:
        import pandas as pd
        st.success("✅ Pandas imported successfully")
    except Exception as e:
        st.error(f"❌ Pandas import error: {e}")
    
    try:
        from data_processor import DataProcessor
        st.success("✅ DataProcessor imported successfully")
    except Exception as e:
        st.error(f"❌ DataProcessor import error: {e}")
    
    try:
        from matching_engine import MatchingEngine
        st.success("✅ MatchingEngine imported successfully")
    except Exception as e:
        st.error(f"❌ MatchingEngine import error: {e}")