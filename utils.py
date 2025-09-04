"""
Utility functions for the Value Comparison Tool.
"""

import pandas as pd
import io
from typing import Dict, Any, Optional
import streamlit as st
from datetime import datetime

def create_template_excel() -> bytes:
    """Create a template Excel file for input data."""
    
    # Create template data
    template_data = {
        'Category': [
            'Switches',
            'Connectors',
            'Sensors',
            'Example Category'
        ],
        'Sub-Category': [
            'Push Button',
            'Terminal Block',
            'Temperature Sensor',
            'Example Sub-Category'
        ],
        'Attribute Name': [
            'Brand',
            'Voltage Rating',
            'Temperature Range',
            'Example Attribute'
        ],
        'Input Value': [
            'EAO, 02',
            '24V',
            '-40°C to +85°C',
            'Your value here'
        ]
    }
    
    # Create DataFrame
    template_df = pd.DataFrame(template_data)
    
    # Convert to Excel bytes
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        template_df.to_excel(writer, sheet_name='Input Data', index=False)
        
        # Add instructions sheet
        instructions_data = {
            'Column': ['Category', 'Sub-Category', 'Attribute Name', 'Input Value'],
            'Description': [
                'The main category of the item',
                'The sub-category within the main category',
                'The specific attribute being measured',
                'The value you want to compare against the database'
            ],
            'Example': [
                'Switches',
                'Push Button',
                'Brand',
                'EAO, 02'
            ]
        }
        
        instructions_df = pd.DataFrame(instructions_data)
        instructions_df.to_excel(writer, sheet_name='Instructions', index=False)
    
    output.seek(0)
    return output.getvalue()

def export_results_to_excel(results_df: pd.DataFrame) -> bytes:
    """Export comparison results to Excel format."""
    
    output = io.BytesIO()
    
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        # Main results sheet
        results_df.to_excel(writer, sheet_name='Comparison Results', index=False)
        
        # Summary sheet
        summary_data = create_summary_data(results_df)
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_excel(writer, sheet_name='Summary', index=False)
        
        # Status breakdown sheet
        status_breakdown = results_df['Status'].value_counts().reset_index()
        status_breakdown.columns = ['Status', 'Count']
        status_breakdown.to_excel(writer, sheet_name='Status Breakdown', index=False)
        
        # Category analysis sheet
        if 'Category' in results_df.columns:
            category_analysis = results_df.groupby('Category')['Status'].value_counts().unstack(fill_value=0)
            category_analysis.to_excel(writer, sheet_name='Category Analysis')
    
    output.seek(0)
    return output.getvalue()

def create_summary_data(results_df: pd.DataFrame) -> Dict[str, Any]:
    """Create summary data for export."""
    
    summary = {
        'Metric': [
            'Total Comparisons',
            'Exact Matches',
            'Similar Matches',
            'Not Found',
            'Average Similarity Score',
            'Highest Similarity Score',
            'Lowest Similarity Score',
            'Export Date'
        ],
        'Value': [
            len(results_df),
            len(results_df[results_df['Status'] == 'Exact Match']),
            len(results_df[results_df['Status'] == 'Similar Match']),
            len(results_df[results_df['Status'] == 'Not Found']),
            f"{results_df['Similarity %'].mean():.2f}%" if 'Similarity %' in results_df.columns else 'N/A',
            f"{results_df['Similarity %'].max():.2f}%" if 'Similarity %' in results_df.columns else 'N/A',
            f"{results_df['Similarity %'].min():.2f}%" if 'Similarity %' in results_df.columns else 'N/A',
            datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        ]
    }
    
    return summary

def format_similarity_score(score: float) -> str:
    """Format similarity score for display."""
    return f"{score:.1f}%"

def get_status_color(status: str) -> str:
    """Get color for status display."""
    colors = {
        'Exact Match': '#28a745',
        'Similar Match': '#ffc107',
        'Not Found': '#dc3545'
    }
    return colors.get(status, '#6c757d')

def validate_input_data(df: pd.DataFrame) -> tuple[bool, str]:
    """Validate input data structure."""
    
    required_columns = ['Category', 'Sub-Category', 'Attribute Name', 'Input Value']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        return False, f"Missing required columns: {missing_columns}"
    
    # Check for empty values in key columns
    key_columns = ['Category', 'Sub-Category', 'Attribute Name']
    empty_rows = df[key_columns].isnull().any(axis=1).sum()
    
    if empty_rows > 0:
        return False, f"Found {empty_rows} rows with empty values in key columns"
    
    return True, "Data validation passed"

def clean_data_for_display(df: pd.DataFrame, max_length: int = 100) -> pd.DataFrame:
    """Clean data for better display in Streamlit."""
    
    df_clean = df.copy()
    
    # Truncate long text values
    text_columns = df_clean.select_dtypes(include=['object']).columns
    for col in text_columns:
        df_clean[col] = df_clean[col].astype(str).apply(
            lambda x: x[:max_length] + '...' if len(x) > max_length else x
        )
    
    return df_clean

def create_download_link(data: bytes, filename: str, mime_type: str) -> str:
    """Create a download link for data."""
    
    import base64
    
    b64 = base64.b64encode(data).decode()
    href = f'<a href="data:{mime_type};base64,{b64}" download="{filename}">Download {filename}</a>'
    return href

def get_file_size_mb(data: bytes) -> float:
    """Get file size in MB."""
    return len(data) / (1024 * 1024)

def format_file_size(size_bytes: int) -> str:
    """Format file size in human readable format."""
    
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    elif size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.1f} MB"
    else:
        return f"{size_bytes / (1024 * 1024 * 1024):.1f} GB"

def create_progress_bar(current: int, total: int, label: str = "Progress") -> None:
    """Create a progress bar in Streamlit."""
    
    progress = current / total if total > 0 else 0
    st.progress(progress)
    st.caption(f"{label}: {current}/{total} ({progress:.1%})")

def display_metric_card(title: str, value: str, delta: Optional[str] = None, 
                       delta_color: str = "normal") -> None:
    """Display a metric card in Streamlit."""
    
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        st.metric(title, value, delta)
    
    with col2:
        if delta_color == "normal":
            st.caption("")
        elif delta_color == "inverse":
            st.caption("📈")
        else:
            st.caption("📊")

def create_data_quality_report(df: pd.DataFrame) -> Dict[str, Any]:
    """Create a data quality report."""
    
    report = {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'memory_usage_mb': df.memory_usage(deep=True).sum() / (1024 * 1024),
        'duplicate_rows': df.duplicated().sum(),
        'null_values': df.isnull().sum().to_dict(),
        'data_types': df.dtypes.to_dict(),
        'column_stats': {}
    }
    
    # Column-specific statistics
    for col in df.columns:
        if df[col].dtype == 'object':
            report['column_stats'][col] = {
                'unique_values': df[col].nunique(),
                'null_count': df[col].isnull().sum(),
                'null_percentage': (df[col].isnull().sum() / len(df)) * 100,
                'most_common': df[col].value_counts().head(5).to_dict()
            }
    
    return report

def display_data_quality_report(report: Dict[str, Any]) -> None:
    """Display data quality report in Streamlit."""
    
    st.markdown("### 📊 Data Quality Report")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Rows", f"{report['total_rows']:,}")
    
    with col2:
        st.metric("Total Columns", report['total_columns'])
    
    with col3:
        st.metric("Memory Usage", f"{report['memory_usage_mb']:.1f} MB")
    
    with col4:
        st.metric("Duplicate Rows", report['duplicate_rows'])
    
    # Column statistics
    with st.expander("📋 Column Statistics"):
        for col, stats in report['column_stats'].items():
            st.markdown(f"**{col}**")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.caption(f"Unique: {stats['unique_values']}")
            
            with col2:
                st.caption(f"Null: {stats['null_count']} ({stats['null_percentage']:.1f}%)")
            
            with col3:
                if stats['most_common']:
                    most_common = max(stats['most_common'], key=stats['most_common'].get)
                    st.caption(f"Most common: {most_common}")

def create_sample_data(n_samples: int = 10) -> pd.DataFrame:
    """Create sample data for testing."""
    
    import random
    
    categories = ['Switches', 'Connectors', 'Sensors', 'Resistors', 'Capacitors']
    sub_categories = ['Push Button', 'Terminal Block', 'Temperature', 'Carbon Film', 'Ceramic']
    attributes = ['Brand', 'Voltage', 'Current', 'Resistance', 'Capacitance']
    values = ['EAO, 02', '24V', '1A', '10kΩ', '100µF', 'IDEC, AL6', '12V', '2A', '1kΩ', '47µF']
    
    data = []
    for i in range(n_samples):
        data.append({
            'Category': random.choice(categories),
            'Sub-Category': random.choice(sub_categories),
            'Attribute Name': random.choice(attributes),
            'Input Value': random.choice(values)
        })
    
    return pd.DataFrame(data)