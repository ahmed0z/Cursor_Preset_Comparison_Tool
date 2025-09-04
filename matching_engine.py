"""
Advanced matching engine for comparing input values against preset database.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
import re
from rapidfuzz import fuzz, process
import textdistance
from sentence_transformers import SentenceTransformer
import streamlit as st
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

@dataclass
class MatchResult:
    """Data class for storing match results."""
    input_value: str
    matched_preset: str
    similarity_score: float
    match_type: str
    comment: str
    suggested_value: str
    status: str
    category: str
    sub_category: str
    attribute_name: str
    preset_row_id: int

class MatchingEngine:
    """Advanced matching engine for value comparison."""
    
    def __init__(self, preset_data: pd.DataFrame):
        self.preset_data = preset_data
        self.sentence_model = None
        self._initialize_models()
        self._prepare_data()
    
    def _initialize_models(self):
        """Initialize the sentence transformer model."""
        try:
            # Use a lightweight model for better performance
            self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        except Exception as e:
            st.warning(f"Could not load sentence transformer model: {e}")
            self.sentence_model = None
    
    def _prepare_data(self):
        """Prepare data for efficient matching."""
        
        # Create lookup dictionaries for faster access
        self.category_lookup = {}
        self.attribute_lookup = {}
        self.composite_lookup = {}
        
        # Add computed columns if they don't exist
        if 'Composite_Key' not in self.preset_data.columns:
            self.preset_data['Composite_Key'] = (
                self.preset_data['Category'].astype(str) + '|' + 
                self.preset_data['Sub-Category'].astype(str) + '|' + 
                self.preset_data['Attribute Name'].astype(str)
            )
        
        if 'Preset_Normalized' not in self.preset_data.columns:
            self.preset_data['Preset_Normalized'] = self.preset_data['Preset values'].apply(self._normalize_value)
        
        if 'Has_Units' not in self.preset_data.columns:
            self.preset_data['Has_Units'] = self.preset_data['Preset values'].apply(self._has_units)
        
        if 'Has_Numbers' not in self.preset_data.columns:
            self.preset_data['Has_Numbers'] = self.preset_data['Preset values'].apply(self._has_numbers)
        
        if 'Has_Commas' not in self.preset_data.columns:
            self.preset_data['Has_Commas'] = self.preset_data['Preset values'].apply(self._has_commas)
        
        # Analyze formatting patterns for each composite key
        self._analyze_formatting_patterns()
        
        # Group by composite key for faster lookup
        for idx, row in self.preset_data.iterrows():
            composite_key = row['Composite_Key']
            if composite_key not in self.composite_lookup:
                self.composite_lookup[composite_key] = []
            
            self.composite_lookup[composite_key].append({
                'row_id': idx,
                'preset_value': row['Preset values'],
                'normalized': row['Preset_Normalized'],
                'has_units': row['Has_Units'],
                'has_numbers': row['Has_Numbers'],
                'has_commas': row['Has_Commas']
            })
        
        # Create embeddings for preset values if model is available
        if self.sentence_model is not None:
            self._create_embeddings()
    
    def _create_embeddings(self):
        """Create embeddings for preset values."""
        try:
            # Sample a subset for performance
            sample_size = min(10000, len(self.preset_data))
            sample_data = self.preset_data.sample(n=sample_size, random_state=42)
            
            preset_values = sample_data['Preset values'].fillna('').astype(str).tolist()
            self.preset_embeddings = self.sentence_model.encode(preset_values)
            self.embedding_indices = sample_data.index.tolist()
            
        except Exception as e:
            st.warning(f"Could not create embeddings: {e}")
            self.preset_embeddings = None
    
    def compare_values(self, input_data: pd.DataFrame, 
                      similarity_threshold: float = 75.0,
                      max_matches: int = 5) -> pd.DataFrame:
        """
        Compare input values against preset database.
        
        Args:
            input_data: DataFrame with input values
            similarity_threshold: Minimum similarity percentage
            max_matches: Maximum matches per input value
            
        Returns:
            DataFrame with comparison results
        """
        
        results = []
        
        # Process each input row
        for idx, input_row in input_data.iterrows():
            input_value = str(input_row.get('Input Value', ''))
            category = str(input_row.get('Category', ''))
            sub_category = str(input_row.get('Sub-Category', ''))
            attribute_name = str(input_row.get('Attribute Name', ''))
            
            if not input_value.strip():
                continue
            
            # Find matches for this input
            matches = self._find_matches(
                input_value, category, sub_category, attribute_name,
                similarity_threshold, max_matches
            )
            
            # Add results
            for match in matches:
                results.append(match)
        
        # Convert to DataFrame
        if results:
            # Convert MatchResult objects to dictionaries
            results_dicts = []
            for result in results:
                results_dicts.append({
                    'Original Input': result.input_value,
                    'Matched Preset Value': result.matched_preset,
                    'Similarity %': result.similarity_score,
                    'Comment': result.comment,
                    'Suggested Value': result.suggested_value,
                    'Status': result.status,
                    'Category': result.category,
                    'Sub-Category': result.sub_category,
                    'Attribute Name': result.attribute_name,
                    'Preset Row ID': result.preset_row_id
                })
            
            results_df = pd.DataFrame(results_dicts)
            return results_df
        else:
            # Return empty DataFrame with expected columns
            return pd.DataFrame(columns=[
                'Original Input', 'Matched Preset Value', 'Similarity %',
                'Comment', 'Suggested Value', 'Status', 'Category',
                'Sub-Category', 'Attribute Name', 'Preset Row ID'
            ])
    
    def _find_matches(self, input_value: str, category: str, sub_category: str, 
                     attribute_name: str, similarity_threshold: float, 
                     max_matches: int) -> List[MatchResult]:
        """Find matches for a single input value."""
        
        matches = []
        
        # Strategy 1: Exact match within same composite key
        exact_match = self._find_exact_match(input_value, category, sub_category, attribute_name)
        if exact_match:
            matches.append(exact_match)
            return matches  # Return immediately for exact matches
        
        # Strategy 2: Fuzzy matching within same composite key
        fuzzy_matches = self._find_fuzzy_matches(
            input_value, category, sub_category, attribute_name,
            similarity_threshold, max_matches
        )
        matches.extend(fuzzy_matches)
        
        # Strategy 3: Cross-category matching if no good matches found
        if not matches or max(m.similarity_score for m in matches) < similarity_threshold:
            cross_matches = self._find_cross_category_matches(
                input_value, similarity_threshold, max_matches
            )
            matches.extend(cross_matches)
        
        # Strategy 4: Semantic matching using embeddings
        if self.sentence_model is not None and len(matches) < max_matches:
            semantic_matches = self._find_semantic_matches(
                input_value, similarity_threshold, max_matches - len(matches)
            )
            matches.extend(semantic_matches)
        
        # Sort by similarity score and limit results
        matches.sort(key=lambda x: x.similarity_score, reverse=True)
        return matches[:max_matches]
    
    def _find_exact_match(self, input_value: str, category: str, 
                         sub_category: str, attribute_name: str) -> Optional[MatchResult]:
        """Find exact match within the same composite key."""
        
        composite_key = f"{category}|{sub_category}|{attribute_name}"
        
        if composite_key not in self.composite_lookup:
            return None
        
        # Try original input first
        normalized_input = self._normalize_value(input_value)
        
        for preset_info in self.composite_lookup[composite_key]:
            if preset_info['normalized'] == normalized_input:
                return MatchResult(
                    input_value=input_value,
                    matched_preset=preset_info['preset_value'],
                    similarity_score=100.0,
                    match_type='exact',
                    comment='Exact match found',
                    suggested_value=preset_info['preset_value'],
                    status='Exact Match',
                    category=category,
                    sub_category=sub_category,
                    attribute_name=attribute_name,
                    preset_row_id=preset_info['row_id']
                )
        
        # Try formatted input to match pattern
        formatted_input = self._format_input_to_match_pattern(input_value, composite_key)
        if formatted_input != input_value:
            normalized_formatted = self._normalize_value(formatted_input)
            
            for preset_info in self.composite_lookup[composite_key]:
                if preset_info['normalized'] == normalized_formatted:
                    return MatchResult(
                        input_value=input_value,
                        matched_preset=preset_info['preset_value'],
                        similarity_score=100.0,
                        match_type='exact',
                        comment=f'Exact match found (formatted from "{input_value}" to "{formatted_input}")',
                        suggested_value=preset_info['preset_value'],
                        status='Exact Match',
                        category=category,
                        sub_category=sub_category,
                        attribute_name=attribute_name,
                        preset_row_id=preset_info['row_id']
                    )
        
        return None
    
    def _find_fuzzy_matches(self, input_value: str, category: str, sub_category: str,
                           attribute_name: str, similarity_threshold: float,
                           max_matches: int) -> List[MatchResult]:
        """Find fuzzy matches within the same composite key."""
        
        matches = []
        composite_key = f"{category}|{sub_category}|{attribute_name}"
        
        if composite_key not in self.composite_lookup:
            return matches
        
        normalized_input = self._normalize_value(input_value)
        
        for preset_info in self.composite_lookup[composite_key]:
            # Calculate multiple similarity scores
            scores = self._calculate_similarity_scores(normalized_input, preset_info['normalized'])
            max_score = max(scores.values())
            
            if max_score >= similarity_threshold:
                match_type = max(scores, key=scores.get)
                comment = self._generate_comment(input_value, preset_info['preset_value'], match_type)
                
                matches.append(MatchResult(
                    input_value=input_value,
                    matched_preset=preset_info['preset_value'],
                    similarity_score=max_score,
                    match_type=match_type,
                    comment=comment,
                    suggested_value=preset_info['preset_value'],
                    status='Similar Match',
                    category=category,
                    sub_category=sub_category,
                    attribute_name=attribute_name,
                    preset_row_id=preset_info['row_id']
                ))
        
        return matches
    
    def _find_cross_category_matches(self, input_value: str, similarity_threshold: float,
                                   max_matches: int) -> List[MatchResult]:
        """Find matches across different categories."""
        
        matches = []
        normalized_input = self._normalize_value(input_value)
        
        # Sample from all preset values for performance
        sample_size = min(5000, len(self.preset_data))
        sample_data = self.preset_data.sample(n=sample_size, random_state=42)
        
        for idx, row in sample_data.iterrows():
            scores = self._calculate_similarity_scores(normalized_input, row['Preset_Normalized'])
            max_score = max(scores.values())
            
            if max_score >= similarity_threshold:
                match_type = max(scores, key=scores.get)
                comment = f"Cross-category match: {match_type}"
                
                matches.append(MatchResult(
                    input_value=input_value,
                    matched_preset=row['Preset values'],
                    similarity_score=max_score,
                    match_type=match_type,
                    comment=comment,
                    suggested_value=row['Preset values'],
                    status='Similar Match',
                    category=row['Category'],
                    sub_category=row['Sub-Category'],
                    attribute_name=row['Attribute Name'],
                    preset_row_id=idx
                ))
        
        return matches
    
    def _find_semantic_matches(self, input_value: str, similarity_threshold: float,
                             max_matches: int) -> List[MatchResult]:
        """Find semantic matches using embeddings."""
        
        if self.preset_embeddings is None:
            return []
        
        matches = []
        
        try:
            # Encode input value
            input_embedding = self.sentence_model.encode([input_value])
            
            # Calculate cosine similarities
            similarities = np.dot(self.preset_embeddings, input_embedding.T).flatten()
            
            # Get top matches
            top_indices = np.argsort(similarities)[::-1][:max_matches]
            
            for idx in top_indices:
                similarity = similarities[idx] * 100  # Convert to percentage
                
                if similarity >= similarity_threshold:
                    preset_idx = self.embedding_indices[idx]
                    preset_row = self.preset_data.iloc[preset_idx]
                    
                    matches.append(MatchResult(
                        input_value=input_value,
                        matched_preset=preset_row['Preset values'],
                        similarity_score=similarity,
                        match_type='semantic',
                        comment='Semantic similarity match',
                        suggested_value=preset_row['Preset values'],
                        status='Similar Match',
                        category=preset_row['Category'],
                        sub_category=preset_row['Sub-Category'],
                        attribute_name=preset_row['Attribute Name'],
                        preset_row_id=preset_idx
                    ))
        
        except Exception as e:
            st.warning(f"Error in semantic matching: {e}")
        
        return matches
    
    def _calculate_similarity_scores(self, input_value: str, preset_value: str) -> Dict[str, float]:
        """Calculate multiple similarity scores between two values."""
        
        scores = {}
        
        # RapidFuzz scores
        scores['ratio'] = fuzz.ratio(input_value, preset_value)
        scores['partial_ratio'] = fuzz.partial_ratio(input_value, preset_value)
        scores['token_sort_ratio'] = fuzz.token_sort_ratio(input_value, preset_value)
        scores['token_set_ratio'] = fuzz.token_set_ratio(input_value, preset_value)
        
        # TextDistance scores
        try:
            scores['jaro_winkler'] = textdistance.jaro_winkler(input_value, preset_value) * 100
            scores['levenshtein'] = (1 - textdistance.levenshtein.normalized(input_value, preset_value)) * 100
        except:
            scores['jaro_winkler'] = 0
            scores['levenshtein'] = 0
        
        # Custom pattern-based scoring
        scores['pattern'] = self._calculate_pattern_score(input_value, preset_value)
        
        return scores
    
    def _calculate_pattern_score(self, input_value: str, preset_value: str) -> float:
        """Calculate pattern-based similarity score."""
        
        score = 0
        
        # Check for unit conversions
        if self._is_unit_conversion(input_value, preset_value):
            score += 30
        
        # Check for case differences
        if input_value.lower() == preset_value.lower():
            score += 20
        
        # Check for punctuation differences
        clean_input = re.sub(r'[^\w\s]', '', input_value)
        clean_preset = re.sub(r'[^\w\s]', '', preset_value)
        if clean_input.lower() == clean_preset.lower():
            score += 15
        
        # Check for number format differences
        if self._is_number_format_difference(input_value, preset_value):
            score += 25
        
        return min(score, 100)
    
    def _is_unit_conversion(self, input_value: str, preset_value: str) -> bool:
        """Check if values are unit conversions of each other."""
        
        # Extract numbers and units
        input_numbers = re.findall(r'(\d+\.?\d*)', input_value)
        preset_numbers = re.findall(r'(\d+\.?\d*)', preset_value)
        
        if len(input_numbers) != len(preset_numbers):
            return False
        
        # Check if numbers are approximately equal (allowing for unit conversion)
        for inp_num, preset_num in zip(input_numbers, preset_numbers):
            try:
                inp_val = float(inp_num)
                preset_val = float(preset_num)
                
                # Allow for reasonable conversion ratios
                ratio = inp_val / preset_val if preset_val != 0 else 0
                if not (0.1 <= ratio <= 10):  # Reasonable conversion range
                    return False
            except:
                return False
        
        return True
    
    def _is_number_format_difference(self, input_value: str, preset_value: str) -> bool:
        """Check if values differ only in number formatting."""
        
        # Remove all non-numeric characters except decimal points
        input_clean = re.sub(r'[^\d.]', '', input_value)
        preset_clean = re.sub(r'[^\d.]', '', preset_value)
        
        return input_clean == preset_clean
    
    def _generate_comment(self, input_value: str, preset_value: str, match_type: str) -> str:
        """Generate a comment explaining the match."""
        
        comments = {
            'ratio': 'High overall similarity',
            'partial_ratio': 'Contains similar substring',
            'token_sort_ratio': 'Similar words in different order',
            'token_set_ratio': 'Similar word set',
            'jaro_winkler': 'Similar character sequence',
            'levenshtein': 'Small edit distance',
            'pattern': 'Pattern-based match (format/unit differences)'
        }
        
        base_comment = comments.get(match_type, 'Similar match found')
        
        # Add specific details
        if input_value.lower() == preset_value.lower():
            base_comment += ' (case difference)'
        elif re.sub(r'[^\w\s]', '', input_value).lower() == re.sub(r'[^\w\s]', '', preset_value).lower():
            base_comment += ' (punctuation difference)'
        elif self._is_unit_conversion(input_value, preset_value):
            base_comment += ' (unit conversion)'
        elif self._is_number_format_difference(input_value, preset_value):
            base_comment += ' (number format difference)'
        
        return base_comment
    
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
    
    def _analyze_formatting_patterns(self):
        """Analyze formatting patterns for each composite key."""
        self.formatting_patterns = {}
        
        for composite_key, group_data in self.composite_lookup.items():
            if not group_data:
                continue
            
            # Get all preset values for this key
            preset_values = [item['preset_value'] for item in group_data]
            
            # Analyze patterns
            patterns = {
                'common_units': self._extract_common_units(preset_values),
                'number_formats': self._extract_number_formats(preset_values),
                'separators': self._extract_separators(preset_values),
                'case_patterns': self._extract_case_patterns(preset_values),
                'typical_format': self._determine_typical_format(preset_values)
            }
            
            self.formatting_patterns[composite_key] = patterns
    
    def _extract_common_units(self, values: List[str]) -> List[str]:
        """Extract common units from a list of values."""
        units = set()
        for value in values:
            # Extract units using regex
            unit_matches = re.findall(r'\d+\s*([a-zA-Z°]+)', str(value))
            units.update(unit_matches)
        return list(units)
    
    def _extract_number_formats(self, values: List[str]) -> Dict[str, Any]:
        """Extract number formatting patterns."""
        formats = {
            'decimal_places': set(),
            'thousand_separators': set(),
            'number_ranges': []
        }
        
        for value in values:
            # Find numbers
            numbers = re.findall(r'\d+\.?\d*', str(value))
            for num in numbers:
                if '.' in num:
                    decimal_places = len(num.split('.')[1])
                    formats['decimal_places'].add(decimal_places)
        
        formats['decimal_places'] = list(formats['decimal_places'])
        return formats
    
    def _extract_separators(self, values: List[str]) -> List[str]:
        """Extract common separators used in values."""
        separators = set()
        for value in values:
            # Look for common separators
            if ',' in str(value):
                separators.add(',')
            if ';' in str(value):
                separators.add(';')
            if '|' in str(value):
                separators.add('|')
            if ' - ' in str(value):
                separators.add(' - ')
            if ' to ' in str(value):
                separators.add(' to ')
        return list(separators)
    
    def _extract_case_patterns(self, values: List[str]) -> Dict[str, Any]:
        """Extract case patterns from values."""
        patterns = {
            'all_uppercase': 0,
            'all_lowercase': 0,
            'title_case': 0,
            'mixed_case': 0
        }
        
        for value in values:
            value_str = str(value)
            if value_str.isupper():
                patterns['all_uppercase'] += 1
            elif value_str.islower():
                patterns['all_lowercase'] += 1
            elif value_str.istitle():
                patterns['title_case'] += 1
            else:
                patterns['mixed_case'] += 1
        
        return patterns
    
    def _determine_typical_format(self, values: List[str]) -> str:
        """Determine the most typical format for a set of values."""
        if not values:
            return "unknown"
        
        # Analyze the most common patterns
        sample_values = values[:10]  # Use first 10 values for analysis
        
        # Check for common patterns
        if all(re.search(r'\d+[a-zA-Z]+', str(v)) for v in sample_values):
            return "number_unit"
        elif all(re.search(r'[a-zA-Z]+,\s*\d+', str(v)) for v in sample_values):
            return "text_number"
        elif all(re.search(r'\d+', str(v)) and not re.search(r'[a-zA-Z]', str(v)) for v in sample_values):
            return "number_only"
        elif all(re.search(r'[a-zA-Z]', str(v)) and not re.search(r'\d', str(v)) for v in sample_values):
            return "text_only"
        else:
            return "mixed"
    
    def _format_input_to_match_pattern(self, input_value: str, composite_key: str) -> str:
        """Format input value to match the typical pattern for the composite key."""
        if composite_key not in self.formatting_patterns:
            return input_value
        
        patterns = self.formatting_patterns[composite_key]
        typical_format = patterns.get('typical_format', 'unknown')
        
        # Extract number and unit from input
        number_match = re.search(r'(\d+\.?\d*)', input_value)
        unit_match = re.search(r'([a-zA-Z°]+)', input_value)
        
        if not number_match:
            return input_value
        
        number = number_match.group(1)
        unit = unit_match.group(1) if unit_match else ""
        
        # Format based on typical pattern
        if typical_format == "number_unit":
            # Format like "2V", "3V", etc.
            if unit:
                return f"{number}{unit.upper()}"
            else:
                return number
        elif typical_format == "text_number":
            # Format like "Brand, 02"
            text_part = re.sub(r'\d+\.?\d*\s*[a-zA-Z°]*', '', input_value).strip()
            if text_part:
                return f"{text_part}, {number}"
            else:
                return number
        elif typical_format == "number_only":
            return number
        elif typical_format == "text_only":
            return input_value
        else:
            return input_value