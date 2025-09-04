#!/usr/bin/env python3
"""
Debug script to identify Streamlit app issues.
"""

import sys
import traceback

def test_imports():
    """Test all imports to identify issues."""
    print("🔍 Testing imports...")
    
    try:
        import streamlit as st
        print("✅ Streamlit imported successfully")
    except Exception as e:
        print(f"❌ Streamlit import error: {e}")
        return False
    
    try:
        import pandas as pd
        print("✅ Pandas imported successfully")
    except Exception as e:
        print(f"❌ Pandas import error: {e}")
        return False
    
    try:
        import numpy as np
        print("✅ NumPy imported successfully")
    except Exception as e:
        print(f"❌ NumPy import error: {e}")
        return False
    
    try:
        from rapidfuzz import fuzz
        print("✅ RapidFuzz imported successfully")
    except Exception as e:
        print(f"❌ RapidFuzz import error: {e}")
        return False
    
    try:
        import textdistance
        print("✅ TextDistance imported successfully")
    except Exception as e:
        print(f"❌ TextDistance import error: {e}")
        return False
    
    try:
        from sentence_transformers import SentenceTransformer
        print("✅ SentenceTransformers imported successfully")
    except Exception as e:
        print(f"❌ SentenceTransformers import error: {e}")
        return False
    
    return True

def test_custom_modules():
    """Test custom module imports."""
    print("\n🔍 Testing custom modules...")
    
    try:
        from data_processor import DataProcessor
        print("✅ DataProcessor imported successfully")
    except Exception as e:
        print(f"❌ DataProcessor import error: {e}")
        traceback.print_exc()
        return False
    
    try:
        from matching_engine import MatchingEngine
        print("✅ MatchingEngine imported successfully")
    except Exception as e:
        print(f"❌ MatchingEngine import error: {e}")
        traceback.print_exc()
        return False
    
    try:
        from utils import create_template_excel
        print("✅ Utils imported successfully")
    except Exception as e:
        print(f"❌ Utils import error: {e}")
        traceback.print_exc()
        return False
    
    try:
        from config import Config
        print("✅ Config imported successfully")
    except Exception as e:
        print(f"❌ Config import error: {e}")
        traceback.print_exc()
        return False
    
    return True

def test_app_creation():
    """Test creating the main app components."""
    print("\n🔍 Testing app creation...")
    
    try:
        # Test creating a simple Streamlit app
        import streamlit as st
        
        # Test basic Streamlit functionality
        st.set_page_config(page_title="Test", page_icon="🔍")
        print("✅ Streamlit page config works")
        
        return True
    except Exception as e:
        print(f"❌ Streamlit app creation error: {e}")
        traceback.print_exc()
        return False

def test_data_loading():
    """Test data loading functionality."""
    print("\n🔍 Testing data loading...")
    
    try:
        from data_processor import DataProcessor
        processor = DataProcessor()
        print("✅ DataProcessor created successfully")
        
        # Test auto-loading
        data = processor.load_preset_data_auto()
        if data is not None:
            print(f"✅ Data loaded successfully: {len(data):,} records")
        else:
            print("⚠️ No data loaded (this might be expected if no Excel file exists)")
        
        return True
    except Exception as e:
        print(f"❌ Data loading error: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all diagnostic tests."""
    print("🚀 Streamlit App Diagnostic Tool")
    print("=" * 50)
    
    # Test imports
    if not test_imports():
        print("\n❌ Import tests failed. Please install missing packages.")
        return
    
    # Test custom modules
    if not test_custom_modules():
        print("\n❌ Custom module tests failed. Check for syntax errors.")
        return
    
    # Test app creation
    if not test_app_creation():
        print("\n❌ App creation tests failed. Check Streamlit installation.")
        return
    
    # Test data loading
    if not test_data_loading():
        print("\n❌ Data loading tests failed. Check data files.")
        return
    
    print("\n🎉 All tests passed! The app should work correctly.")
    print("\n💡 If you're still getting errors, try:")
    print("   1. streamlit run app.py --server.headless true")
    print("   2. Check the Streamlit logs for specific error messages")
    print("   3. Ensure all dependencies are installed: pip install -r requirements.txt")

if __name__ == "__main__":
    main()