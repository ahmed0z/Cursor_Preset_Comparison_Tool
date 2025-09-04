#!/usr/bin/env python3
"""
Demonstration script for the Value Comparison Tool.
Shows key features and capabilities.
"""

import sys
sys.path.append('/home/ubuntu/.local/lib/python3.13/site-packages')

from data_processor import DataProcessor
from matching_engine import MatchingEngine
from utils import create_template_excel, export_results_to_excel
import pandas as pd

def main():
    """Run demonstration of the Value Comparison Tool."""
    
    print("🔍 Value Comparison Tool - Demonstration")
    print("=" * 50)
    
    # Load sample preset data
    print("\n📚 Loading sample preset data...")
    sample_preset_data = pd.DataFrame({
        'Category': [
            'Switches', 'Switches', 'Switches', 'Switches',
            'Connectors', 'Connectors', 'Connectors', 'Connectors',
            'Sensors', 'Sensors', 'Sensors', 'Sensors',
            'Resistors', 'Resistors', 'Resistors', 'Resistors',
            'Switches', 'Switches', 'Switches', 'Switches',
            'Connectors', 'Connectors', 'Connectors', 'Connectors',
            'Sensors', 'Sensors', 'Sensors', 'Sensors',
            'Resistors', 'Resistors', 'Resistors', 'Resistors'
        ],
        'Sub-Category': [
            'Push Button', 'Push Button', 'Push Button', 'Push Button',
            'Terminal Block', 'Terminal Block', 'Terminal Block', 'Terminal Block',
            'Temperature', 'Temperature', 'Temperature', 'Temperature',
            'Carbon Film', 'Carbon Film', 'Carbon Film', 'Carbon Film',
            'Push Button', 'Push Button', 'Push Button', 'Push Button',
            'Terminal Block', 'Terminal Block', 'Terminal Block', 'Terminal Block',
            'Temperature', 'Temperature', 'Temperature', 'Temperature',
            'Carbon Film', 'Carbon Film', 'Carbon Film', 'Carbon Film'
        ],
        'Attribute Name': [
            'Brand', 'Voltage', 'Current', 'Material',
            'Brand', 'Voltage Rating', 'Current Rating', 'Material',
            'Brand', 'Temperature Range', 'Accuracy', 'Output Type',
            'Brand', 'Resistance', 'Tolerance', 'Power Rating',
            'Brand', 'Voltage', 'Current', 'Material',
            'Brand', 'Voltage Rating', 'Current Rating', 'Material',
            'Brand', 'Temperature Range', 'Accuracy', 'Output Type',
            'Brand', 'Resistance', 'Tolerance', 'Power Rating'
        ],
        'Preset values': [
            'EAO, 02', '24V', '1A', 'Plastic',
            'Phoenix Contact', '24V', '10A', 'Metal',
            'Honeywell', '-40°C to +85°C', '±0.5°C', 'Analog',
            'Vishay', '10kΩ', '±5%', '0.25W',
            'IDEC, AL6', '12V', '2A', 'Metal',
            'WAGO', '12V', '8A', 'Plastic',
            'Sensirion', '-20°C to +60°C', '±1°C', 'Digital',
            'Yageo', '1kΩ', '±1%', '0.125W'
        ]
    })
    
    print(f"✅ Loaded {len(sample_preset_data)} preset records")
    
    # Create matching engine
    print("\n🔧 Initializing matching engine...")
    engine = MatchingEngine(sample_preset_data)
    print("✅ Matching engine ready")
    
    # Test input data with various scenarios
    print("\n📝 Preparing test input data...")
    test_input_data = pd.DataFrame({
        'Category': [
            'Switches', 'Switches', 'Switches', 'Switches',
            'Connectors', 'Connectors', 'Connectors',
            'Sensors', 'Sensors', 'Sensors',
            'Resistors', 'Resistors', 'Resistors'
        ],
        'Sub-Category': [
            'Push Button', 'Push Button', 'Push Button', 'Push Button',
            'Terminal Block', 'Terminal Block', 'Terminal Block',
            'Temperature', 'Temperature', 'Temperature',
            'Carbon Film', 'Carbon Film', 'Carbon Film'
        ],
        'Attribute Name': [
            'Brand', 'Voltage', 'Current', 'Material',
            'Brand', 'Voltage Rating', 'Current Rating',
            'Brand', 'Temperature Range', 'Accuracy',
            'Brand', 'Resistance', 'Tolerance'
        ],
        'Input Value': [
            'EAO, 02',      # Exact match
            '24v',          # Case difference
            '1 a',          # Space difference
            'plastic',      # Case difference
            'Phoenix Contact', # Exact match
            '24 V',         # Space difference
            '10 A',         # Space difference
            'Honeywell',    # Exact match
            '-40°C to +85°C', # Exact match
            '±0.5°C',       # Exact match
            'Vishay',       # Exact match
            '10k ohm',      # Unit difference
            '±5%'           # Exact match
        ]
    })
    
    print(f"✅ Prepared {len(test_input_data)} test input values")
    
    # Run comparison
    print("\n🚀 Running comparison analysis...")
    results = engine.compare_values(
        test_input_data, 
        similarity_threshold=75.0, 
        max_matches=3
    )
    
    print(f"✅ Comparison completed: {len(results)} results")
    
    # Display results
    print("\n📊 Comparison Results:")
    print("-" * 80)
    
    exact_matches = 0
    similar_matches = 0
    not_found = 0
    
    for i, (_, row) in enumerate(results.iterrows()):
        status = row['Status']
        if status == 'Exact Match':
            exact_matches += 1
        elif status == 'Similar Match':
            similar_matches += 1
        else:
            not_found += 1
        
        print(f"{i+1:2d}. Input: \"{row['Original Input']}\"")
        print(f"    Match: \"{row['Matched Preset Value']}\"")
        print(f"    Score: {row['Similarity %']:.1f}% | Status: {status}")
        print(f"    Comment: {row['Comment']}")
        print()
    
    # Summary statistics
    print("📈 Summary Statistics:")
    print(f"   • Exact Matches: {exact_matches}")
    print(f"   • Similar Matches: {similar_matches}")
    print(f"   • Not Found: {not_found}")
    print(f"   • Total Comparisons: {len(results)}")
    
    if len(results) > 0:
        avg_similarity = results['Similarity %'].mean()
        print(f"   • Average Similarity: {avg_similarity:.1f}%")
    
    # Demonstrate export functionality
    print("\n💾 Testing export functionality...")
    try:
        excel_data = export_results_to_excel(results)
        print(f"✅ Excel export successful ({len(excel_data):,} bytes)")
        
        csv_data = results.to_csv(index=False)
        print(f"✅ CSV export successful ({len(csv_data):,} bytes)")
    except Exception as e:
        print(f"❌ Export error: {e}")
    
    # Demonstrate template creation
    print("\n📋 Testing template creation...")
    try:
        template_data = create_template_excel()
        print(f"✅ Template creation successful ({len(template_data):,} bytes)")
    except Exception as e:
        print(f"❌ Template creation error: {e}")
    
    print("\n🎉 Demonstration completed successfully!")
    print("\nKey Features Demonstrated:")
    print("✅ Advanced fuzzy matching algorithms")
    print("✅ Case-insensitive matching")
    print("✅ Unit and format normalization")
    print("✅ Similarity scoring with configurable thresholds")
    print("✅ Excel import/export functionality")
    print("✅ Template generation")
    print("✅ Comprehensive result analysis")
    
    print("\n🚀 To run the full Streamlit application:")
    print("   python3 run_app.py")
    print("   or")
    print("   streamlit run app.py")

if __name__ == "__main__":
    main()