"""
Data processing module for loading and preprocessing Excel files.
"""

import pandas as pd
import numpy as np
import streamlit as st
from typing import Optional, Dict, Any
import re
from pathlib import Path

class DataProcessor:
    """Handles data loading and preprocessing operations."""
    
    def __init__(self):
        self.preset_data = None
        self.input_data = None
    
    @st.cache_data
    def load_preset_data(_self, uploaded_file) -> Optional[pd.DataFrame]:
        """
        Load and preprocess the preset database from uploaded Excel file.
        
        Args:
            uploaded_file: Uploaded file object from Streamlit
            
        Returns:
            Preprocessed DataFrame or None if error
        """
        try:
            # Read the Excel file
            df = pd.read_excel(uploaded_file)
            
            # Validate required columns
            required_columns = ['Category', 'Sub-Category', 'Attribute Name', 'Preset values']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                st.error(f"Missing required columns: {missing_columns}")
                return None
            
            # Clean the data
            df = _self._clean_preset_data(df)
            
            # Add computed columns for better matching
            df = _self._add_computed_columns(df)
            
            return df
            
        except Exception as e:
            st.error(f"Error loading preset data: {str(e)}")
            return None
    
    @st.cache_data
    def load_input_data(_self, uploaded_file) -> Optional[pd.DataFrame]:
        """
        Load and preprocess input data from uploaded Excel file.
        
        Args:
            uploaded_file: Uploaded file object from Streamlit
            
        Returns:
            Preprocessed DataFrame or None if error
        """
        try:
            # Read the Excel file
            df = pd.read_excel(uploaded_file)
            
            # Handle different column naming conventions
            column_mapping = {
                'Structure': 'Category',
                'Full Structure': 'Sub-Category', 
                'Attribute Name': 'Attribute Name',
                'Attribute value': 'Input Value'
            }
            
            # Rename columns if they exist
            for old_name, new_name in column_mapping.items():
                if old_name in df.columns and new_name not in df.columns:
                    df = df.rename(columns={old_name: new_name})
            
            # If we don't have the expected structure, try to infer it
            if 'Category' not in df.columns:
                # Try to create category from first column
                if len(df.columns) >= 3:
                    df['Category'] = df.iloc[:, 0]
                    df['Sub-Category'] = df.iloc[:, 1] if len(df.columns) > 1 else ''
                    df['Attribute Name'] = df.iloc[:, 2] if len(df.columns) > 2 else ''
                    df['Input Value'] = df.iloc[:, 3] if len(df.columns) > 3 else ''
            
            # Clean the data
            df = _self._clean_input_data(df)
            
            return df
            
        except Exception as e:
            st.error(f"Error loading input data: {str(e)}")
            return None
    
    def _clean_preset_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and normalize preset data."""
        
        # Remove completely empty rows
        df = df.dropna(how='all')
        
        # Fill NaN values with empty strings for string columns
        string_columns = ['Category', 'Sub-Category', 'Attribute Name', 'Preset values']
        for col in string_columns:
            if col in df.columns:
                df[col] = df[col].fillna('').astype(str)
        
        # Remove rows where all key columns are empty
        key_columns = ['Category', 'Sub-Category', 'Attribute Name']
        df = df[~(df[key_columns].eq('').all(axis=1))]
        
        # Normalize text data
        for col in string_columns:
            if col in df.columns:
                df[col] = df[col].str.strip()
        
        return df
    
    def _clean_input_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and normalize input data."""
        
        # Remove completely empty rows
        df = df.dropna(how='all')
        
        # Fill NaN values with empty strings for string columns
        string_columns = ['Category', 'Sub-Category', 'Attribute Name', 'Input Value']
        for col in string_columns:
            if col in df.columns:
                df[col] = df[col].fillna('').astype(str)
        
        # Remove rows where all key columns are empty
        key_columns = ['Category', 'Sub-Category', 'Attribute Name']
        df = df[~(df[key_columns].eq('').all(axis=1))]
        
        # Normalize text data
        for col in string_columns:
            if col in df.columns:
                df[col] = df[col].str.strip()
        
        return df
    
    def _add_computed_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add computed columns for better matching."""
        
        # Create a unique key for each row
        df['Row_ID'] = df.index
        
        # Create normalized versions of preset values for matching
        df['Preset_Normalized'] = df['Preset values'].apply(self._normalize_value)
        
        # Create a composite key for grouping
        df['Composite_Key'] = (
            df['Category'].astype(str) + '|' + 
            df['Sub-Category'].astype(str) + '|' + 
            df['Attribute Name'].astype(str)
        )
        
        # Extract units and numbers for better matching
        df['Has_Units'] = df['Preset values'].apply(self._has_units)
        df['Has_Numbers'] = df['Preset values'].apply(self._has_numbers)
        df['Has_Commas'] = df['Preset values'].apply(self._has_commas)
        
        return df
    
    def _normalize_value(self, value: str) -> str:
        """Normalize a value for comparison."""
        if pd.isna(value) or value == '':
            return ''
        
        # Convert to string and strip whitespace
        normalized = str(value).strip()
        
        # Convert to lowercase
        normalized = normalized.lower()
        
        # Remove extra whitespace
        normalized = re.sub(r'\s+', ' ', normalized)
        
        # Remove common punctuation that doesn't affect meaning
        normalized = re.sub(r'[^\w\s\-.,()]', '', normalized)
        
        return normalized
    
    def _has_units(self, value: str) -> bool:
        """Check if value contains units."""
        if pd.isna(value) or value == '':
            return False
        
        # Common unit patterns
        unit_patterns = [
            r'\d+\s*(mm|cm|m|in|inch|ft|kg|g|lb|oz|v|a|w|hz|°c|°f)',
            r'\d+\s*[a-z]{1,3}(?=\s|$|,|\))',
        ]
        
        value_str = str(value).lower()
        return any(re.search(pattern, value_str) for pattern in unit_patterns)
    
    def _has_numbers(self, value: str) -> bool:
        """Check if value contains numbers."""
        if pd.isna(value) or value == '':
            return False
        
        return bool(re.search(r'\d', str(value)))
    
    def _has_commas(self, value: str) -> bool:
        """Check if value contains commas."""
        if pd.isna(value) or value == '':
            return False
        
        return ',' in str(value)
    
    def get_data_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get summary statistics for a DataFrame."""
        
        if df is None or df.empty:
            return {}
        
        summary = {
            'total_records': len(df),
            'columns': list(df.columns),
            'memory_usage': df.memory_usage(deep=True).sum() / 1024 / 1024,  # MB
        }
        
        # Add column-specific statistics
        for col in df.columns:
            if df[col].dtype == 'object':
                summary[f'{col}_unique'] = df[col].nunique()
                summary[f'{col}_null'] = df[col].isnull().sum()
        
        return summary
    
    def validate_data_structure(self, df: pd.DataFrame, data_type: str = 'preset') -> bool:
        """Validate that DataFrame has the expected structure."""
        
        if data_type == 'preset':
            required_columns = ['Category', 'Sub-Category', 'Attribute Name', 'Preset values']
        else:  # input
            required_columns = ['Category', 'Sub-Category', 'Attribute Name', 'Input Value']
        
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            st.error(f"Missing required columns for {data_type} data: {missing_columns}")
            return False
        
        return True