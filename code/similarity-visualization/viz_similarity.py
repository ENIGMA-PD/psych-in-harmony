import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import glob
import random
from itertools import combinations
from pathlib import Path

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)

# Base path to output harmonization folder
base_path = '/Users/julia/Desktop/origami/ENIGMA-PD/code/similarity-analysis/output_similarity-analysis'
OUTPUT_DIR = Path('/Users/julia/Desktop/origami/ENIGMA-PD/code/similarity-visualization/output_similarity-visualization')

# Path to question text mapping file
QUESTION_TEXT_FILE = '/Users/julia/Desktop/origami/ENIGMA-PD/code/similarity-analysis/similarity-analysis-input_without-sleep.csv'

# Path to question pairs file (NEW)
QUESTION_PAIRS_FILE = '/Users/julia/Desktop/origami/ENIGMA-PD/code/similarity-visualization/question_pairs.csv'

# Model names to look for (without file extensions)
model_names = ['Bio_ClinicalBERT', 'BioBERT', 'ClinicalBert', 'ClinicalSentenceTransformer', 
               'Harmony', 'MiniLM', 'MPNet', 'SimCSE']

# Custom color palette - ColorBrewer Yellow-Green-Blue scale
CUSTOM_COLORS = [
    '#FFFFCC',  # Light yellow
    '#C7E9B4',  # Light green
    '#7FCDBB',  # Light blue-green
    '#41B6C4',  # Medium blue-green
    '#2C7FB8',  # Medium blue
    '#253494',  # Dark blue
    '#A1DAB4',  # Soft green
    '#6BAED6'   # Light blue
]

def load_question_text_mapping(file_path):
    """
    Load the question text mapping file.
    
    Expected format: CSV with columns including 'question_number', 'question_text', 'questionnaire_name', and 'construct'
    
    Returns: Tuple of (question_mapping, construct_mapping)
    - question_mapping: Dictionary mapping item_id -> (question_text, questionnaire_name)
    - construct_mapping: Dictionary mapping item_id -> construct
    """
    try:
        df = pd.read_csv(file_path)
        print(f"Loaded question text mapping with columns: {df.columns.tolist()}")
        
        # Check if we have the expected columns
        required_cols = ['question_number', 'question_text', 'questionnaire_name', 'construct']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            print(f"Warning: Missing columns: {missing_cols}")
            print("Available columns:", df.columns.tolist())
            return {}, {}
        
        # Create mapping dictionaries
        question_mapping = {}
        construct_mapping = {}
        
        for _, row in df.iterrows():
            question_number = str(row['question_number']).strip()
            question_text = str(row['question_text']).strip()
            questionnaire_name = str(row['questionnaire_name']).strip()
            construct = str(row['construct']).strip()
            
            # Create mapping entries
            question_mapping[question_number] = (question_text, questionnaire_name)
            construct_mapping[question_number] = construct
        
        print(f"Created question mapping for {len(set([v[0] for v in question_mapping.values()]))} unique questions")
        print(f"Created construct mapping for {len(set(construct_mapping.values()))} constructs")
        print(f"Total mapping entries: {len(question_mapping)}")
        
        # Show construct distribution
        construct_counts = {}
        for construct in construct_mapping.values():
            construct_counts[construct] = construct_counts.get(construct, 0) + 1
        
        print("Construct distribution:")
        for construct, count in construct_counts.items():
            print(f"  {construct}: {count} items")
        
        # Show a few examples of the mapping
        example_keys = list(question_mapping.keys())[:5]
        print("Example mappings:")
        for key in example_keys:
            text, quest_name = question_mapping[key]
            construct = construct_mapping[key]
            print(f"  '{key}' -> '{text[:30]}...' ({quest_name}) [{construct}]")
        
        return question_mapping, construct_mapping
        
    except Exception as e:
        print(f"Error loading question text mapping: {e}")
        print("Proceeding without question text mapping...")
        return {}, {}

def load_question_pairs(file_path):
    """
    Load the question pairs file.
    
    Expected format: CSV with two columns: 
    - Column 1: questionnaire 1 item number
    - Column 2: questionnaire 2 item number
    
    Returns: List of tuples (item1, item2)
    """
    try:
        if not os.path.exists(file_path):
            print(f"Question pairs file not found: {file_path}")
            print("Please create a CSV file with two columns: item1, item2")
            return []
        
        df = pd.read_csv(file_path)
        print(f"Loaded question pairs file with columns: {df.columns.tolist()}")
        
        if df.shape[1] < 2:
            print("Error: Question pairs file must have at least 2 columns")
            return []
        
        # Use first two columns regardless of their names
        pairs = []
        for _, row in df.iterrows():
            item1 = str(row.iloc[0]).strip()
            item2 = str(row.iloc[1]).strip()
            pairs.append((item1, item2))
        
        print(f"Loaded {len(pairs)} question pairs")
        
        # Show a few examples
        for i, (item1, item2) in enumerate(pairs[:5]):
            print(f"  Pair {i+1}: '{item1}' <-> '{item2}'")
        
        return pairs
        
    except Exception as e:
        print(f"Error loading question pairs file: {e}")
        return []

def get_question_info(item_id, question_mapping):
    """Get question text and questionnaire name for an item ID, with fallback to item ID if not found"""
    if not question_mapping:
        return item_id, "Unknown"
    
    # Try exact match first
    if item_id in question_mapping:
        return question_mapping[item_id]
    
    # Try case-insensitive match
    for key, value in question_mapping.items():
        if key.lower() == item_id.lower():
            return value
    
    # For similarity matrix format (e.g., "BDI_BDI_01"), try removing the first prefix
    if '_' in item_id:
        parts = item_id.split('_')
        if len(parts) >= 3 and parts[0] == parts[1]:  # e.g., "BDI_BDI_01"
            # Try with single prefix: "BDI_01"
            single_prefix = '_'.join(parts[1:])
            if single_prefix in question_mapping:
                return question_mapping[single_prefix]
            
            # Try without any prefix: "01" (less likely but possible)
            no_prefix = '_'.join(parts[2:])
            if no_prefix in question_mapping:
                return question_mapping[no_prefix]
    
    # If no match found, return original item ID with a note
    print(f"Warning: No question text found for item '{item_id}'")
    return f"[Item {item_id}]", "Unknown"

def determine_construct_for_pair(item1, item2, construct_mapping):
    """
    Determine which construct a question pair belongs to based on the construct mapping from the data file.
    
    Args:
        item1, item2: Item IDs
        construct_mapping: Dictionary mapping item_id -> construct (from your CSV file)
    
    Returns:
        Construct name or None if not found
    """
    
    def find_construct_for_item(item_id):
        """Find construct for a single item, trying various formats"""
        # Try exact match first
        if item_id in construct_mapping:
            return construct_mapping[item_id]
        
        # Try case-insensitive match
        for key, value in construct_mapping.items():
            if key.lower() == item_id.lower():
                return value
        
        # For similarity matrix format (e.g., "BDI_BDI_01"), try removing the first prefix
        if '_' in item_id:
            parts = item_id.split('_')
            if len(parts) >= 3 and parts[0] == parts[1]:  # e.g., "BDI_BDI_01"
                # Try with single prefix: "BDI_01"
                single_prefix = '_'.join(parts[1:])
                if single_prefix in construct_mapping:
                    return construct_mapping[single_prefix]
                
                # Try without any prefix: "01" (less likely but possible)
                no_prefix = '_'.join(parts[2:])
                if no_prefix in construct_mapping:
                    return construct_mapping[no_prefix]
        
        return None
    
    # Get constructs for both items using data from your CSV file
    construct1 = find_construct_for_item(item1)
    construct2 = find_construct_for_item(item2)
    
    # If both items belong to the same construct, use that
    if construct1 and construct1 == construct2:
        return construct1
    
    # If they belong to different constructs, use the first item's construct
    # (or you could create a mixed construct approach)
    if construct1:
        return construct1
    elif construct2:
        return construct2
    
    print(f"Warning: Could not determine construct for pair {item1} <-> {item2}")
    print(f"  Construct for {item1}: {construct1}")
    print(f"  Construct for {item2}: {construct2}")
    return None

def get_constructs(base_path):
    """Get all construct folders from the base path"""
    construct_folders = []
    for item in os.listdir(base_path):
        item_path = os.path.join(base_path, item)
        if os.path.isdir(item_path) and item.endswith('_similarity-analysis_output'):
            construct = item.replace('_similarity-analysis', '')
            construct_folders.append(construct)
    return construct_folders

def get_similarity_files(base_path, construct):
    """Get all similarity CSV files for a given construct"""
    construct_folder = os.path.join(base_path, f"{construct}_similarity-analysis_output")
    files = {}
    
    for model in model_names:
        # Look for files matching the pattern
        pattern = os.path.join(construct_folder, f"{model}_{construct}_similarity*.csv")
        matching_files = glob.glob(pattern)
        
        if matching_files:
            files[model] = matching_files[0]  # Take the first match
    
    return files

def truncate_text(text, max_length=60):
    """Truncate text for better visualization"""
    if len(text) <= max_length:
        return text
    return text[:max_length] + "..."

def wrap_text(text, max_length=80):
    """Wrap text for better readability in figures"""
    words = text.split()
    lines = []
    current_line = []
    current_length = 0
    
    for word in words:
        if current_length + len(word) + 1 <= max_length:
            current_line.append(word)
            current_length += len(word) + 1
        else:
            if current_line:
                lines.append(' '.join(current_line))
            current_line = [word]
            current_length = len(word)
    
    if current_line:
        lines.append(' '.join(current_line))
    
    return '\n'.join(lines)

def format_question_label(question_text, questionnaire_name, max_length=100):
    """Format question label as 'Q: question text (questionnaire name)'"""
    truncated_text = truncate_text(question_text, max_length)
    return f"{questionnaire_name}: {truncated_text}"

def create_visualizations_for_pairs(question_pairs, question_mapping, construct_mapping, constructs, base_path):
    """Create visualizations for question pairs, grouped by construct"""
    
    # Group pairs by construct using data from your CSV file
    pairs_by_construct = {}
    
    for item1, item2 in question_pairs:
        construct = determine_construct_for_pair(item1, item2, construct_mapping)
        if construct:
            if construct not in pairs_by_construct:
                pairs_by_construct[construct] = []
            pairs_by_construct[construct].append((item1, item2))
        else:
            print(f"Skipping pair {item1} <-> {item2}: Could not determine construct from data file")
    
    print(f"\nGrouped pairs by construct (from your CSV data):")
    for construct, pairs in pairs_by_construct.items():
        print(f"  {construct}: {len(pairs)} pairs")
    
    # Process each construct
    for construct, construct_pairs in pairs_by_construct.items():
        similarity_files = get_similarity_files(base_path, construct)
        
        if similarity_files:
            print(f"\nFound files for {construct}:")
            for model, file_path in similarity_files.items():
                print(f"  {model}: {os.path.basename(file_path)}")
            
            create_visualizations_for_construct(construct, similarity_files, question_mapping, construct_pairs)
        else:
            print(f"\nNo similarity files found for {construct}")

def create_visualizations_for_construct(construct, similarity_files, question_mapping, question_pairs):
    """Create visualizations for a specific construct using predefined question pairs"""
    print(f"\nProcessing construct: {construct}")
    
    # Load all model data for this construct
    model_data = {}
    for model, file_path in similarity_files.items():
        try:
            df = pd.read_csv(file_path, index_col=0)
            model_data[model] = df
            print(f"  Loaded {model}: {df.shape}")
        except Exception as e:
            print(f"  Error loading {model}: {e}")
    
    if not model_data:
        print(f"  No data loaded for {construct}")
        return
    
    if not question_pairs:
        print(f"  No question pairs defined for {construct}")
        return
    
    # Use predefined question pairs for this construct
    print(f"  Using {len(question_pairs)} predefined question pairs for {construct}")
    
    # Extract similarity scores for all models
    results = []
    pair_labels = {}  # Store formatted question texts for each pair
    
    for model, df in model_data.items():
        for item1, item2 in question_pairs:
            try:
                # Try to find the similarity score - check both directions
                score = None
                if item1 in df.index and item2 in df.columns:
                    score = df.loc[item1, item2]
                elif item2 in df.index and item1 in df.columns:
                    score = df.loc[item2, item1]
                elif item1 in df.columns and item2 in df.index:
                    score = df.loc[item2, item1]
                elif item2 in df.columns and item1 in df.index:
                    score = df.loc[item1, item2]
                
                if score is None or pd.isna(score):
                    print(f"    Warning: No similarity score found for {item1} <-> {item2} in {model} (construct: {construct})")
                    continue
                
                # Get question information
                q1_text, q1_questionnaire = get_question_info(item1, question_mapping)
                q2_text, q2_questionnaire = get_question_info(item2, question_mapping)
                
                # Create readable pair label with new format
                pair_key = f"{item1}|||{item2}"  # Use unique separator
                if pair_key not in pair_labels:
                    # Format: "Q1: question text (questionnaire name)"
                    q1_label = format_question_label(q1_text, q1_questionnaire, 80)
                    q2_label = format_question_label(q2_text, q2_questionnaire, 80)
                    
                    # Wrap for display
                    q1_wrapped = wrap_text(q1_label, max_length=60)
                    q2_wrapped = wrap_text(q2_label, max_length=60)
                    
                    pair_labels[pair_key] = f"{q1_wrapped}\nvs.\n{q2_wrapped}"
                
                results.append({
                    'Model': model,
                    'Question Pair': pair_labels[pair_key],
                    'Similarity Score': score,
                    'Item_ID_1': item1,
                    'Item_ID_2': item2,
                    'Full_Q1': q1_text,
                    'Full_Q2': q2_text,
                    'Questionnaire_1': q1_questionnaire,
                    'Questionnaire_2': q2_questionnaire,
                    'Pair_Key': pair_key
                })
            except Exception as e:
                print(f"    Error getting score for {model} with pair {item1}-{item2}: {e}")
    
    if not results:
        print(f"  No results generated for {construct}")
        return
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # Create heatmap - optimized for poster with custom colors
    plt.figure(figsize=(11, 7))  # Larger figure for poster
    try:
        pivot_df = results_df.pivot(index='Question Pair', columns='Model', values='Similarity Score')
        
        # Create custom colormap with ColorBrewer Yellow-Green-Blue scale
        from matplotlib.colors import LinearSegmentedColormap
        colors = ['#FFFFCC', '#C7E9B4', '#7FCDBB', '#41B6C4', '#2C7FB8', '#253494']  # Yellow to Green to Blue
        custom_cmap = LinearSegmentedColormap.from_list('YlGnBu', colors)
        
        # Create heatmap with better formatting for poster
        sns.heatmap(pivot_df, 
                   annot=True, 
                   cmap=custom_cmap, 
                   fmt='.2f', 
                   linewidths=0.5,
                   square=True,
                   cbar_kws={'shrink': 0.95},
                   annot_kws={'size': 8})
        
        plt.title(f'Semantic Similarity Matrix for {construct.title()} Questionnaires', fontsize=14, fontweight='bold', pad=20)
        
        # Add custom axis labels
        plt.xlabel('Sentence Transformer Models', fontsize=10, fontweight='bold')
        plt.ylabel('Selected Question Pairs Between Different Questionnaires', fontsize=10, fontweight='bold')
        
        # Adjust font sizes
        plt.xticks(fontsize=10, rotation=45, ha='right')
        plt.yticks(fontsize=10, rotation=0)
        
        plt.tight_layout()
        plt.subplots_adjust(left=0.4)  # More room for question text
        
        # Save heatmap
        heatmap_filename = OUTPUT_DIR / f'similarity_comparison_{construct}_heatmap.png'
        plt.savefig(heatmap_filename, dpi=300, bbox_inches='tight')
        print(f"  Saved heatmap: {heatmap_filename}")
    except Exception as e:
        print(f"  Error creating heatmap for {construct}: {e}")
    
    plt.close('all')  # Close figures to free memory
    
    # Save selected pairs info with both item IDs and question text (updated format)
    pairs_info = []
    processed_pairs = set()
    for _, row in results_df.iterrows():
        pair_key = (row['Item_ID_1'], row['Item_ID_2'])
        if pair_key not in processed_pairs:
            pairs_info.append({
                'Item_ID_1': row['Item_ID_1'],
                'Item_ID_2': row['Item_ID_2'],
                'Question_Text_1': row['Full_Q1'],
                'Question_Text_2': row['Full_Q2'],
                'Questionnaire_1': row['Questionnaire_1'],
                'Questionnaire_2': row['Questionnaire_2'],
                'Formatted_Q1': f"{row['Questionnaire_1']} {row['Full_Q1']}",
                'Formatted_Q2': f"{row['Questionnaire_2']} {row['Full_Q2']})"
            })
            processed_pairs.add(pair_key)
    
    pairs_df = pd.DataFrame(pairs_info)
    pairs_filename = OUTPUT_DIR / f'selected_pairs_{construct}.csv'
    pairs_df.to_csv(pairs_filename, index=False)
    print(f"  Saved selected pairs: {pairs_filename}")

# Main execution
if __name__ == "__main__":
    # Create output directory if it doesn't exist
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load question text mapping and construct mapping
    question_mapping, construct_mapping = load_question_text_mapping(QUESTION_TEXT_FILE)
    
    # Load question pairs
    question_pairs = load_question_pairs(QUESTION_PAIRS_FILE)
    
    if not question_pairs:
        print("No question pairs loaded. Please check your question pairs file.")
        print(f"Expected file: {QUESTION_PAIRS_FILE}")
        print("Format: CSV with two columns (item1, item2)")
        exit(1)
    
    if not construct_mapping:
        print("No construct mapping loaded. Please check your question text mapping file.")
        print(f"Expected file: {QUESTION_TEXT_FILE}")
        print("Expected columns: construct, questionnaire_name, question_number, question_text")
        exit(1)
    
    # Get all constructs
    constructs = get_constructs(base_path)
    print(f"Found constructs: {constructs}")
    
    if not constructs:
        print("No construct folders found!")
        exit(1)
    
    # Process pairs grouped by construct (based on your CSV data)
    create_visualizations_for_pairs(question_pairs, question_mapping, construct_mapping, constructs, base_path)
    
    print("\nAll visualizations completed!")