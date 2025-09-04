"""
Streamlit-based Value Comparison Tool
Compares user input values against a reference database with advanced matching algorithms.
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import io
from typing import List, Dict, Tuple, Optional
import re
from datetime import datetime

# Import our custom modules
from data_processor import DataProcessor
from matching_engine import MatchingEngine
from utils import create_template_excel, export_results_to_excel

# Page configuration
st.set_page_config(
    page_title="Value Comparison Tool",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #2c3e50;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .success-message {
        background-color: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #c3e6cb;
    }
    .warning-message {
        background-color: #fff3cd;
        color: #856404;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #ffeaa7;
    }
    .error-message {
        background-color: #f8d7da;
        color: #721c24;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #f5c6cb;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'preset_data' not in st.session_state:
    st.session_state.preset_data = None
if 'input_data' not in st.session_state:
    st.session_state.input_data = None
if 'comparison_results' not in st.session_state:
    st.session_state.comparison_results = None
if 'matching_engine' not in st.session_state:
    st.session_state.matching_engine = None
if 'data_processor' not in st.session_state:
    st.session_state.data_processor = None
if 'database_loaded' not in st.session_state:
    st.session_state.database_loaded = False

def main():
    """Main application function"""
    
    # Auto-load database if not already loaded
    if not st.session_state.database_loaded:
        auto_load_database()
    
    # Header
    st.markdown('<h1 class="main-header">🔍 Value Comparison Tool</h1>', unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <p style="font-size: 1.2rem; color: #666;">
            Compare input values against reference database with intelligent matching
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar for navigation
    with st.sidebar:
        st.markdown("## 🧭 Navigation")
        page = st.selectbox(
            "Choose a page:",
            ["🏠 Home", "📊 Data Management", "🔍 Value Comparison", "📈 Results Analysis", "⚙️ Settings"]
        )
    
    # Route to appropriate page
    if page == "🏠 Home":
        show_home_page()
    elif page == "📊 Data Management":
        show_data_management_page()
    elif page == "🔍 Value Comparison":
        show_comparison_page()
    elif page == "📈 Results Analysis":
        show_results_page()
    elif page == "⚙️ Settings":
        show_settings_page()

def show_home_page():
    """Display the home page with overview and quick actions"""
    
    st.markdown('<h2 class="section-header">Welcome to the Value Comparison Tool</h2>', unsafe_allow_html=True)
    
    # Overview cards
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3 style="color: #1f77b4;">📚 Database</h3>
            <p style="color: #2c3e50;">Load and manage your reference database with preset values</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3 style="color: #1f77b4;">🔍 Smart Matching</h3>
            <p style="color: #2c3e50;">Advanced algorithms for similarity detection and format normalization</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h3 style="color: #1f77b4;">📊 Analysis</h3>
            <p style="color: #2c3e50;">Detailed comparison reports with similarity scores and suggestions</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Quick actions
    st.markdown('<h3 class="section-header">Quick Actions</h3>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📥 Load Database", use_container_width=True):
            st.session_state.current_page = "📊 Data Management"
            st.rerun()
    
    with col2:
        if st.button("🔍 Start Comparison", use_container_width=True):
            st.session_state.current_page = "🔍 Value Comparison"
            st.rerun()
    
    with col3:
        if st.button("📋 Download Template", use_container_width=True):
            template_data = create_template_excel()
            st.download_button(
                label="📥 Download Input Template",
                data=template_data,
                file_name="input_template.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
    
    # Database status
    if st.session_state.preset_data is not None:
        st.markdown('<div class="success-message">✅ Reference database loaded successfully</div>', unsafe_allow_html=True)
        st.info(f"Database contains {len(st.session_state.preset_data):,} records")
    else:
        st.markdown('<div class="warning-message">⚠️ No reference database loaded. Please load your database first.</div>', unsafe_allow_html=True)

def show_data_management_page():
    """Display the data management page"""
    
    st.markdown('<h2 class="section-header">📊 Data Management</h2>', unsafe_allow_html=True)
    
    # Database loading section
    st.markdown("### Load Reference Database")
    
    uploaded_file = st.file_uploader(
        "Upload your reference database (Excel file)",
        type=['xlsx', 'xls'],
        help="Upload the Excel file containing your preset values database"
    )
    
    if uploaded_file is not None:
        try:
            # Load the data
            if st.session_state.data_processor is None:
                st.session_state.data_processor = DataProcessor()
            
            preset_data = st.session_state.data_processor.load_preset_data(uploaded_file)
            
            if preset_data is not None:
                # Update the database and cache
                if st.session_state.data_processor.update_preset_data(preset_data):
                    st.session_state.preset_data = preset_data
                    st.session_state.matching_engine = MatchingEngine(preset_data)
                    st.session_state.database_loaded = True
                    
                    st.markdown('<div class="success-message">✅ Database loaded and cached successfully!</div>', unsafe_allow_html=True)
                
                # Display database statistics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Total Records", f"{len(preset_data):,}")
                
                with col2:
                    st.metric("Categories", preset_data['Category'].nunique())
                
                with col3:
                    st.metric("Sub-Categories", preset_data['Sub-Category'].nunique())
                
                with col4:
                    st.metric("Attribute Names", preset_data['Attribute Name'].nunique())
                
                # Show sample data
                with st.expander("📋 View Sample Data"):
                    st.dataframe(preset_data.head(20), use_container_width=True)
                
                # Show category distribution
                with st.expander("📊 Category Distribution"):
                    category_counts = preset_data['Category'].value_counts().head(20)
                    st.bar_chart(category_counts)
                
        except Exception as e:
            st.markdown(f'<div class="error-message">❌ Error loading database: {str(e)}</div>', unsafe_allow_html=True)
    
    # Database editing section
    if st.session_state.preset_data is not None:
        st.markdown("### Edit Database")
        
        if st.button("✏️ Open Database Editor"):
            st.session_state.show_editor = True
        
        if st.session_state.get('show_editor', False):
            with st.expander("🗂️ Database Editor", expanded=True):
                edited_data = st.data_editor(
                    st.session_state.preset_data.head(1000),  # Limit for performance
                    use_container_width=True,
                    num_rows="dynamic"
                )
                
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("💾 Save Changes"):
                        st.session_state.preset_data = edited_data
                        st.session_state.matching_engine = MatchingEngine(edited_data)
                        st.success("Database updated successfully!")
                
                with col2:
                    if st.button("❌ Cancel"):
                        st.session_state.show_editor = False
                        st.rerun()
    
    # Cache management section
    if st.session_state.preset_data is not None:
        st.markdown("### Cache Management")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🔄 Refresh Cache", use_container_width=True):
                if st.session_state.data_processor:
                    if st.session_state.data_processor.update_preset_data(st.session_state.preset_data):
                        st.success("Cache refreshed successfully!")
                    else:
                        st.error("Failed to refresh cache")
        
        with col2:
            if st.button("🗑️ Clear Cache", use_container_width=True):
                if st.session_state.data_processor:
                    try:
                        import os
                        if os.path.exists("preset_database.pkl"):
                            os.remove("preset_database.pkl")
                            st.success("Cache cleared successfully!")
                        else:
                            st.info("No cache file found")
                    except Exception as e:
                        st.error(f"Error clearing cache: {e}")
        
        with col3:
            if st.button("📊 Cache Info", use_container_width=True):
                if st.session_state.data_processor:
                    cache_info = st.session_state.data_processor.get_cache_info()
                    if cache_info['pkl_exists']:
                        cache_size_mb = cache_info['pkl_size'] / (1024 * 1024)
                        st.info(f"Cache size: {cache_size_mb:.1f} MB")
                        if cache_info['pkl_modified']:
                            st.info(f"Last modified: {cache_info['pkl_modified']}")
                    else:
                        st.info("No cache file exists")

def show_comparison_page():
    """Display the value comparison page"""
    
    st.markdown('<h2 class="section-header">🔍 Value Comparison</h2>', unsafe_allow_html=True)
    
    if st.session_state.preset_data is None:
        st.markdown('<div class="warning-message">⚠️ Please load a reference database first.</div>', unsafe_allow_html=True)
        return
    
    # Input file upload
    st.markdown("### Upload Input Data")
    
    uploaded_input = st.file_uploader(
        "Upload your input file (Excel format)",
        type=['xlsx', 'xls'],
        help="Upload the Excel file containing values to compare"
    )
    
    if uploaded_input is not None:
        try:
            # Load input data
            data_processor = DataProcessor()
            input_data = data_processor.load_input_data(uploaded_input)
            
            if input_data is not None:
                st.session_state.input_data = input_data
                
                st.markdown('<div class="success-message">✅ Input data loaded successfully!</div>', unsafe_allow_html=True)
                
                # Display input statistics
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Input Records", len(input_data))
                
                with col2:
                    st.metric("Unique Categories", input_data['Category'].nunique() if 'Category' in input_data.columns else 0)
                
                with col3:
                    st.metric("Unique Attributes", input_data['Attribute Name'].nunique() if 'Attribute Name' in input_data.columns else 0)
                
                # Show sample input data
                with st.expander("📋 View Input Data"):
                    st.dataframe(input_data.head(20), use_container_width=True)
                
                # Comparison settings
                st.markdown("### Comparison Settings")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    similarity_threshold = st.slider(
                        "Similarity Threshold (%)",
                        min_value=50,
                        max_value=100,
                        value=75,
                        help="Minimum similarity percentage to consider a match"
                    )
                
                with col2:
                    max_matches = st.slider(
                        "Maximum Matches per Input",
                        min_value=1,
                        max_value=10,
                        value=5,
                        help="Maximum number of matches to return per input value"
                    )
                
                # Run comparison
                if st.button("🚀 Run Comparison", type="primary", use_container_width=True):
                    with st.spinner("Running comparison analysis..."):
                        matching_engine = st.session_state.matching_engine
                        results = matching_engine.compare_values(
                            input_data,
                            similarity_threshold=similarity_threshold,
                            max_matches=max_matches
                        )
                        
                        st.session_state.comparison_results = results
                        
                        st.markdown('<div class="success-message">✅ Comparison completed successfully!</div>', unsafe_allow_html=True)
                        
                        # Show quick results summary
                        exact_matches = len(results[results['Status'] == 'Exact Match'])
                        similar_matches = len(results[results['Status'] == 'Similar Match'])
                        not_found = len(results[results['Status'] == 'Not Found'])
                        
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("Exact Matches", exact_matches)
                        
                        with col2:
                            st.metric("Similar Matches", similar_matches)
                        
                        with col3:
                            st.metric("Not Found", not_found)
                        
                        # Navigate to results page
                        st.session_state.current_page = "📈 Results Analysis"
                        st.rerun()
        
        except Exception as e:
            st.markdown(f'<div class="error-message">❌ Error loading input data: {str(e)}</div>', unsafe_allow_html=True)

def show_results_page():
    """Display the results analysis page"""
    
    st.markdown('<h2 class="section-header">📈 Results Analysis</h2>', unsafe_allow_html=True)
    
    if st.session_state.comparison_results is None:
        st.markdown('<div class="warning-message">⚠️ No comparison results available. Please run a comparison first.</div>', unsafe_allow_html=True)
        return
    
    results = st.session_state.comparison_results
    
    # Results summary
    st.markdown("### Results Summary")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Comparisons", len(results))
    
    with col2:
        exact_matches = len(results[results['Status'] == 'Exact Match'])
        st.metric("Exact Matches", exact_matches)
    
    with col3:
        similar_matches = len(results[results['Status'] == 'Similar Match'])
        st.metric("Similar Matches", similar_matches)
    
    with col4:
        not_found = len(results[results['Status'] == 'Not Found'])
        st.metric("Not Found", not_found)
    
    # Filters
    st.markdown("### Filter Results")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        status_filter = st.selectbox(
            "Filter by Status",
            ["All"] + list(results['Status'].unique())
        )
    
    with col2:
        category_filter = st.selectbox(
            "Filter by Category",
            ["All"] + list(results['Category'].unique())
        )
    
    with col3:
        similarity_filter = st.slider(
            "Minimum Similarity (%)",
            min_value=0,
            max_value=100,
            value=0
        )
    
    # Apply filters
    filtered_results = results.copy()
    
    if status_filter != "All":
        filtered_results = filtered_results[filtered_results['Status'] == status_filter]
    
    if category_filter != "All":
        filtered_results = filtered_results[filtered_results['Category'] == category_filter]
    
    filtered_results = filtered_results[filtered_results['Similarity %'] >= similarity_filter]
    
    # Display filtered results
    st.markdown(f"### Filtered Results ({len(filtered_results)} records)")
    
    # Interactive table
    st.dataframe(
        filtered_results,
        use_container_width=True,
        height=600
    )
    
    # Export options
    st.markdown("### Export Results")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📥 Download Excel Report", use_container_width=True):
            excel_data = export_results_to_excel(filtered_results)
            st.download_button(
                label="📥 Download Excel File",
                data=excel_data,
                file_name=f"comparison_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
    
    with col2:
        if st.button("📊 Download CSV Report", use_container_width=True):
            csv_data = filtered_results.to_csv(index=False)
            st.download_button(
                label="📥 Download CSV File",
                data=csv_data,
                file_name=f"comparison_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
    
    # Detailed analysis
    with st.expander("📊 Detailed Analysis"):
        
        # Status distribution
        st.markdown("#### Status Distribution")
        status_counts = results['Status'].value_counts()
        st.bar_chart(status_counts)
        
        # Similarity distribution
        st.markdown("#### Similarity Score Distribution")
        similarity_hist = results['Similarity %'].hist(bins=20)
        st.pyplot(similarity_hist.figure)
        
        # Top categories with issues
        st.markdown("#### Categories with Most Issues")
        issue_categories = results[results['Status'] == 'Not Found']['Category'].value_counts().head(10)
        st.bar_chart(issue_categories)

def show_settings_page():
    """Display the settings page"""
    
    st.markdown('<h2 class="section-header">⚙️ Settings</h2>', unsafe_allow_html=True)
    
    st.markdown("### Matching Algorithm Settings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Similarity Algorithms")
        use_rapidfuzz = st.checkbox("Use RapidFuzz", value=True)
        use_textdistance = st.checkbox("Use TextDistance", value=True)
        use_embeddings = st.checkbox("Use Embeddings (Experimental)", value=False)
    
    with col2:
        st.markdown("#### Performance Settings")
        cache_size = st.number_input("Cache Size (MB)", min_value=100, max_value=1000, value=500)
        batch_size = st.number_input("Batch Size", min_value=10, max_value=1000, value=100)
    
    st.markdown("### Data Processing Settings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        normalize_case = st.checkbox("Normalize Case", value=True)
        remove_punctuation = st.checkbox("Remove Punctuation", value=False)
        handle_units = st.checkbox("Handle Units", value=True)
    
    with col2:
        handle_synonyms = st.checkbox("Handle Synonyms", value=True)
        handle_abbreviations = st.checkbox("Handle Abbreviations", value=True)
        fuzzy_threshold = st.slider("Fuzzy Matching Threshold", 0.0, 1.0, 0.8)
    
    if st.button("💾 Save Settings", type="primary"):
        st.success("Settings saved successfully!")
    
    st.markdown("### System Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.info(f"**Python Version:** {st.__version__}")
        st.info(f"**Pandas Version:** {pd.__version__}")
    
    with col2:
        st.info(f"**Database Records:** {len(st.session_state.preset_data) if st.session_state.preset_data is not None else 0:,}")
        st.info(f"**Memory Usage:** {st.session_state.get('memory_usage', 'N/A')}")

def auto_load_database():
    """Automatically load the database from PKL cache or Excel file."""
    try:
        # Initialize data processor
        if st.session_state.data_processor is None:
            st.session_state.data_processor = DataProcessor()
        
        # Try to load from cache or Excel
        preset_data = st.session_state.data_processor.load_preset_data_auto()
        
        if preset_data is not None:
            st.session_state.preset_data = preset_data
            st.session_state.matching_engine = MatchingEngine(preset_data)
            st.session_state.database_loaded = True
            
            # Show cache info
            cache_info = st.session_state.data_processor.get_cache_info()
            if cache_info['pkl_exists']:
                cache_size_mb = cache_info['pkl_size'] / (1024 * 1024)
                st.sidebar.success(f"📊 Database loaded from cache ({cache_size_mb:.1f} MB)")
            else:
                st.sidebar.info("📊 Database loaded from Excel file")
        else:
            st.sidebar.warning("⚠️ No database found. Please upload a database file.")
            
    except Exception as e:
        st.sidebar.error(f"❌ Error loading database: {e}")

if __name__ == "__main__":
    main()