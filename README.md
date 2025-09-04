# Value Comparison Tool

A sophisticated Streamlit-based application for comparing user input values against a reference database with advanced matching algorithms.

## Features

### 🔍 Advanced Matching Algorithms
- **Exact Matching**: Direct string comparison for perfect matches
- **Fuzzy Matching**: Multiple algorithms including RapidFuzz and TextDistance
- **Semantic Matching**: AI-powered similarity detection using sentence transformers
- **Pattern Recognition**: Detects unit conversions, case differences, and format variations

### 📊 Smart Data Processing
- **Format Normalization**: Handles various input formats and units
- **Unit Conversion**: Recognizes and converts between different unit systems
- **Synonym Detection**: Identifies equivalent terms and abbreviations
- **Duplicate Prevention**: Avoids duplicate outputs for the same input

### 🎯 Intelligent Comparison Rules
- **Similarity Scoring**: Configurable threshold (≥75% default)
- **Context-Aware Matching**: Considers category, sub-category, and attribute context
- **Cross-Category Analysis**: Finds matches across different categories when needed
- **Detailed Comments**: Explains why matches were found

### 📈 Comprehensive Results
- **Multiple Match Types**: Exact Match, Similar Match, Not Found
- **Similarity Percentages**: Precise scoring for each comparison
- **Suggested Values**: Recommendations from the preset database
- **Interactive Filtering**: Filter by status, category, and similarity score

### 💾 Data Management
- **Excel Integration**: Full support for .xlsx and .xls files
- **Template Generation**: Downloadable input templates
- **Database Editing**: In-app database modification capabilities
- **Export Options**: Excel and CSV export with detailed reports

## Installation

1. **Clone or download the project files**
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**:
   ```bash
   streamlit run app.py
   ```

## Usage

### 1. Load Reference Database
- Navigate to "📊 Data Management"
- Upload your `Preset 25.xlsx` file
- The system will validate and load your reference database

### 2. Prepare Input Data
- Download the input template from the home page
- Fill in your data with the same structure as the template:
  - **Category**: Main category of the item
  - **Sub-Category**: Sub-category within the main category  
  - **Attribute Name**: Specific attribute being measured
  - **Input Value**: Value to compare against the database

### 3. Run Comparison
- Navigate to "🔍 Value Comparison"
- Upload your input file
- Adjust similarity threshold and maximum matches if needed
- Click "🚀 Run Comparison"

### 4. Analyze Results
- Navigate to "📈 Results Analysis"
- Filter and explore results
- Export detailed reports in Excel or CSV format

## Data Structure

### Reference Database (Preset 25.xlsx)
```
Category | Sub-Category | Attribute Name | Preset values
---------|--------------|----------------|---------------
Switches | Push Button  | Brand         | EAO, 02
Connectors| Terminal    | Voltage       | 24V
Sensors  | Temperature  | Range         | -40°C to +85°C
```

### Input Data Template
```
Category | Sub-Category | Attribute Name | Input Value
---------|--------------|----------------|------------
Switches | Push Button  | Brand         | EAO, 02
Connectors| Terminal    | Voltage       | 24V
Sensors  | Temperature  | Range         | -40°C to +85°C
```

## Matching Examples

### Exact Matches
- `"EAO, 02"` → `"EAO, 02"` (100% similarity)
- `"24V"` → `"24V"` (100% similarity)

### Similar Matches
- `"1v"` → `"1V"` (Case difference, 95% similarity)
- `"172.72mm"` → `"6.80" (172.72mm)` (Unit conversion, 90% similarity)
- `"Cover"` → `"cover"` (Case difference, 100% similarity)
- `"Orange, Red"` → `"Red, Orange"` (Order difference, 85% similarity)

### Pattern Recognition
- **Unit Conversions**: `"20 kg"` ↔ `"44.09 lb"`
- **Format Differences**: `"1.5GHz"` ↔ `"1500 MHz"`
- **Punctuation**: `"EAO 02"` ↔ `"EAO, 02"`
- **Synonyms**: `"Voltage"` ↔ `"V"`

## Configuration

### Similarity Thresholds
- **Default**: 75% minimum similarity
- **Range**: 50% - 100%
- **Recommendation**: Start with 75%, adjust based on results

### Performance Settings
- **Cache Size**: 500MB default
- **Batch Size**: 100 records per batch
- **Max Matches**: 5 per input value (configurable)

### Algorithm Weights
- **Exact Match**: 100% weight
- **Fuzzy Ratio**: 90% weight
- **Token Sort**: 85% weight
- **Semantic Match**: 65% weight

## Technical Details

### Dependencies
- **Streamlit**: Web application framework
- **Pandas**: Data manipulation and analysis
- **RapidFuzz**: Fast fuzzy string matching
- **TextDistance**: Additional string similarity algorithms
- **Sentence Transformers**: AI-powered semantic matching
- **OpenPyXL**: Excel file handling

### Performance Optimization
- **Caching**: Streamlit caching for data loading and processing
- **Batch Processing**: Handles large datasets efficiently
- **Memory Management**: Optimized for Streamlit free plan limitations
- **Lazy Loading**: Models loaded only when needed

### Error Handling
- **File Validation**: Checks file format and structure
- **Data Validation**: Ensures required columns are present
- **Graceful Degradation**: Continues operation even if some features fail
- **User Feedback**: Clear error messages and suggestions

## Troubleshooting

### Common Issues

1. **"Missing required columns"**
   - Ensure your input file has: Category, Sub-Category, Attribute Name, Input Value
   - Use the downloadable template as a reference

2. **"File too large"**
   - Maximum file size is 100MB
   - Consider splitting large files into smaller chunks

3. **"Low similarity scores"**
   - Adjust the similarity threshold
   - Check for typos or formatting differences
   - Consider adding synonyms to your database

4. **"Slow processing"**
   - Large files may take time to process
   - Consider reducing batch size in settings
   - Ensure you have sufficient memory available

### Performance Tips

1. **Optimize Input Data**
   - Remove unnecessary columns
   - Clean and normalize data before upload
   - Use consistent formatting

2. **Adjust Settings**
   - Increase similarity threshold for faster processing
   - Reduce maximum matches per input
   - Use smaller batch sizes for large datasets

3. **Database Management**
   - Keep reference database clean and organized
   - Remove duplicate entries
   - Use consistent naming conventions

## Support

For issues, questions, or feature requests, please refer to the application's built-in help system or contact the development team.

## Version History

- **v1.0.0**: Initial release with core matching functionality
  - Advanced fuzzy matching algorithms
  - Semantic similarity detection
  - Excel integration and export
  - Interactive web interface