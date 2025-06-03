import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform
import seaborn as sns
from sklearn.manifold import MDS
import os
import glob
from pathlib import Path


def load_thematic_categories(csv_path):
    """
    Load thematic categories from CSV file.
    
    Parameters:
    csv_path (str): Path to CSV file with columns: construct, questionnaire_name, 
                   question_number, question_text, answer_options, scoring, construct_category
                   Note: question_number should contain full question IDs like "BDI_01", "GDS_02"
    
    Returns:
    dict: Dictionary mapping construct to question-category mappings
    """
    df = pd.read_csv(csv_path)
    
    # Group by construct
    constructs_data = {}
    
    for construct in df['construct'].unique():
        construct_df = df[df['construct'] == construct]
        
        # Create question identifier to category mapping for this construct
        question_id_to_category = {}
        thematic_categories = {}
        
        for _, row in construct_df.iterrows():
            # Use question_number directly as it already contains the full question ID
            question_id = row['question_number']
            
            category = row['construct_category']
            question_text = row['question_text']
            
            # Add to question ID-to-category mapping
            question_id_to_category[question_id] = category
            
            # Add to thematic categories grouping (using question text for display)
            if category not in thematic_categories:
                thematic_categories[category] = []
            thematic_categories[category].append(question_text)
        
        constructs_data[construct] = {
            'thematic_categories': thematic_categories,
            'question_id_to_category': question_id_to_category
        }
    
    return constructs_data


def find_model_files(data_directory, construct_name):
    """
    Find similarity CSV files for a specific construct in the data directory.
    
    Parameters:
    data_directory (str): Path to directory containing construct subdirectories
    construct_name (str): Name of the construct to find files for
    
    Returns:
    dict: Dictionary mapping model names to file paths
    """
    model_files = {}
    data_path = Path(data_directory)
    
    # Look for the construct-specific subdirectory
    construct_subdir = None
    for subdir in data_path.iterdir():
        if subdir.is_dir() and construct_name.lower() in subdir.name.lower():
            construct_subdir = subdir
            break
    
    if construct_subdir is None:
        print(f"No subdirectory found for construct '{construct_name}' in {data_directory}")
        return model_files
    
    print(f"Found construct directory: {construct_subdir}")
    
    # Look for similarity files in the construct subdirectory
    similarity_files = list(construct_subdir.glob("*similarity*.csv"))
    
    for file_path in similarity_files:
        # Extract model name from filename
        # Expected format: ModelName_construct_similarity-results.csv
        filename = file_path.stem  # Remove .csv extension
        
        # Try to extract model name (everything before the construct name)
        parts = filename.split('_')
        if len(parts) >= 2:
            # Find where the construct name appears and take everything before it
            model_name_parts = []
            for part in parts:
                if construct_name.lower() in part.lower():
                    break
                model_name_parts.append(part)
            
            if model_name_parts:
                model_name = '_'.join(model_name_parts)
                model_files[model_name] = str(file_path)
            else:
                # Fallback: use the first part as model name
                model_name = parts[0]
                model_files[model_name] = str(file_path)
    
    return model_files


def find_best_match(target, options):
    """Find the best matching string in a list"""
    # First try exact match
    if target in options:
        return target
    
    # Try case-insensitive match
    lower_options = {opt.lower(): opt for opt in options}
    if target.lower() in lower_options:
        return lower_options[target.lower()]
    
    # Try to find a partial match (substring)
    for opt in options:
        if target in opt or opt in target:
            return opt
    
    return None


def similarity_to_distance(similarity_matrix):
    """Convert similarity matrix to a distance matrix"""
    # Convert similarities (0-1) to distances (0-1)
    # Higher similarity = lower distance
    return 1 - similarity_matrix


def evaluate_thematic_clustering(linkage_matrix, labels, question_id_to_category):
    """Function to evaluate cluster quality based on thematic categories"""
    # Extract flat clusters at different distance thresholds
    # Use a range that can accommodate the number of thematic categories
    max_categories = len(set(question_id_to_category.values()))
    max_clusters = min(max_categories + 2, len(labels))  # Allow a bit more than categories
    num_clusters_to_try = range(2, max_clusters + 1)
    results = []
    
    for n_clusters in num_clusters_to_try:
        # Get cluster assignments
        clusters = hierarchy.fcluster(linkage_matrix, n_clusters, criterion='maxclust')
        
        # Calculate how well clusters align with thematic categories
        # Initialize counters for each category
        category_cluster_counts = {}
        for category in set(question_id_to_category.values()):
            category_cluster_counts[category] = {}
        
        # Count which clusters each category's questions fall into
        for i, label in enumerate(labels):
            if label in question_id_to_category:
                category = question_id_to_category[label]
                cluster = clusters[i]
                if cluster not in category_cluster_counts[category]:
                    category_cluster_counts[category][cluster] = 0
                category_cluster_counts[category][cluster] += 1
        
        # Calculate cohesion score for each category
        # (% of category members in the most common cluster for that category)
        cohesion_scores = {}
        for category, cluster_counts in category_cluster_counts.items():
            if cluster_counts:  # Only calculate if we have questions from this category
                max_count = max(cluster_counts.values()) if cluster_counts else 0
                total_questions = sum(cluster_counts.values())
                cohesion_scores[category] = max_count / total_questions if total_questions > 0 else 0
            else:
                cohesion_scores[category] = np.nan
        
        # Calculate overall cohesion (average of category cohesions)
        valid_scores = [score for score in cohesion_scores.values() if not np.isnan(score)]
        overall_cohesion = np.mean(valid_scores) if valid_scores else 0
        
        results.append({
            'n_clusters': n_clusters,
            'overall_cohesion': overall_cohesion,
            'category_cohesion': cohesion_scores
        })
    
    return results


def process_construct(construct_name, construct_data, data_directory, output_dir, model_order=None):
    """
    Process clustering analysis for a single construct.
    
    Parameters:
    construct_name (str): Name of the construct
    construct_data (dict): Dictionary containing thematic_categories and question_id_to_category
    data_directory (str): Path to directory containing construct subdirectories
    output_dir (str): Directory to save output files
    model_order (list): Optional list defining the order of models for consistent visualization
    """
    print(f"\n{'='*60}")
    print(f"Processing construct: {construct_name}")
    print(f"{'='*60}")
    
    thematic_categories = construct_data['thematic_categories']
    question_id_to_category = construct_data['question_id_to_category']
    
    print(f"Looking for question IDs: {list(question_id_to_category.keys())}")
    
    # Find model files for this specific construct
    model_files = find_model_files(data_directory, construct_name)
    print(f"Found {len(model_files)} models: {list(model_files.keys())}")
    
    if not model_files:
        print(f"No model files found for construct {construct_name}")
        return
    
    # Create output directory for this construct
    construct_output_dir = Path(output_dir) / construct_name
    construct_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process each model
    clustering_results = {}
    mds_embeddings = {}
    all_questions = set()
    
    for model_name, file_path in model_files.items():
        try:
            print(f"\nProcessing {model_name} ({file_path})...")
            
            # Read CSV - the first column will be the index
            df = pd.read_csv(file_path, index_col=0)
            
            print(f"Found columns in similarity file: {list(df.columns)[:10]}...")  # Show first 10
            
            # Filter questions relevant to this construct
            relevant_questions = []
            relevant_indices = []
            
            for idx, question_id in enumerate(df.index):
                if question_id in question_id_to_category:
                    relevant_questions.append(question_id)
                    relevant_indices.append(idx)
            
            print(f"Found {len(relevant_questions)} relevant questions: {relevant_questions}")
            
            if not relevant_questions:
                print(f"No relevant questions found for construct {construct_name} in {model_name}")
                continue
            
            # Filter the dataframe to only include relevant questions
            df_filtered = df.iloc[relevant_indices, relevant_indices]
            
            # Keep track of all questions for later analysis
            all_questions.update(df_filtered.index)
            
            # Process similarity matrix
            similarity_matrix = df_filtered.values
            distance_matrix = similarity_to_distance(similarity_matrix)
            
            # Ensure distance matrix is symmetric
            distance_matrix = (distance_matrix + distance_matrix.T) / 2
            
            # Create dendrogram
            labels = df_filtered.index.tolist()
            
            # Compute linkage for hierarchical clustering
            condensed_distance = squareform(distance_matrix)
            linkage_matrix = hierarchy.linkage(condensed_distance, method='average')
            
            # Save results
            clustering_results[model_name] = {
                'linkage': linkage_matrix,
                'labels': labels,
                'df': df_filtered,
                'distance_matrix': distance_matrix
            }
            
            # MDS for 2D visualization
            mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42)
            mds_result = mds.fit_transform(distance_matrix)
            mds_embeddings[model_name] = {
                'embeddings': mds_result,
                'labels': labels
            }
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
    
    if not clustering_results:
        print(f"No valid results for construct {construct_name}")
        return
    
    # Visualize hierarchical clustering dendrograms
    plt.figure(figsize=(20, 6 * len(clustering_results)))
    
    for i, model_name in enumerate(clustering_results.keys()):
        plt.subplot(len(clustering_results), 2, i*2 + 1)
        
        # Get clustering data
        result = clustering_results[model_name]
        linkage = result['linkage']
        labels = result['labels']
        
        # Plot dendrogram
        dendrogram = hierarchy.dendrogram(
            linkage,
            labels=labels,
            orientation='right',
            leaf_font_size=10,
            leaf_rotation=0,
        )
        
        plt.title(f'Hierarchical Clustering - {model_name}\n{construct_name}', fontsize=14)
        
        # Plot MDS 2D embedding in adjacent subplot
        plt.subplot(len(clustering_results), 2, i*2 + 2)
        
        # Get MDS data
        mds_data = mds_embeddings[model_name]
        points = mds_data['embeddings']
        point_labels = mds_data['labels']
        
        # Create color map for categories
        unique_categories = list(set(question_id_to_category.values()))
        color_map = dict(zip(unique_categories, sns.color_palette("husl", len(unique_categories))))
        
        # Plot MDS points colored by category
        for j, question_id in enumerate(point_labels):
            category = question_id_to_category.get(question_id, "Uncategorized")
            plt.scatter(points[j, 0], points[j, 1], 
                       color=color_map.get(category, 'gray'),
                       label=category)
            plt.annotate(question_id, (points[j, 0], points[j, 1]), 
                        fontsize=8, alpha=0.7)
        
        # Add legend (only unique categories)
        handles, labels_legend = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels_legend, handles))
        plt.legend(by_label.values(), by_label.keys(), 
                  title="Question Categories", loc='best', fontsize=8)
        
        plt.title(f'MDS Visualization - {model_name}\n{construct_name}', fontsize=14)
    
    plt.tight_layout()
    plt.savefig(construct_output_dir / f'{construct_name}_thematic_clustering_analysis.png', 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # Evaluate how well each model's clustering aligns with thematic categories
    evaluation_results = {}
    for model_name, result in clustering_results.items():
        evaluation = evaluate_thematic_clustering(
            result['linkage'], 
            result['labels'], 
            question_id_to_category
        )
        evaluation_results[model_name] = evaluation
    
    # Create a visualization comparing thematic cohesion across models
    plt.figure(figsize=(12, 8))
    
    # Plot overall cohesion by number of clusters for each model
    for model_name, eval_result in evaluation_results.items():
        n_clusters = [r['n_clusters'] for r in eval_result]
        cohesion = [r['overall_cohesion'] for r in eval_result]
        plt.plot(n_clusters, cohesion, marker='o', label=model_name)
    
    plt.xlabel('Number of Clusters')
    plt.ylabel('Thematic Cohesion Score')
    plt.title(f'Thematic Clustering Quality by Model - {construct_name}')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(construct_output_dir / f'{construct_name}_model_thematic_cohesion.png', dpi=300)
    plt.close()
    
    # Create a heatmap of category cohesion for each model
    # Use the actual number of thematic categories as target clusters
    n_categories = len(set(question_id_to_category.values()))
    n_clusters_for_comparison = min(n_categories, len(all_questions) - 1)
    cohesion_data = {}
    
    print(f"Using {n_clusters_for_comparison} clusters (based on {n_categories} thematic categories)")
    
    for model_name, eval_result in evaluation_results.items():
        # Find result with the chosen number of clusters
        target_result = next((r for r in eval_result if r['n_clusters'] == n_clusters_for_comparison), None)
        if target_result:
            cohesion_data[model_name] = target_result['category_cohesion']
    
    if cohesion_data:
        # Convert to DataFrame for heatmap with consistent model ordering
        if model_order:
            # Only include models that exist in cohesion_data, in the specified order
            ordered_models = [model for model in model_order if model in cohesion_data]
            cohesion_df = pd.DataFrame({model: cohesion_data[model] for model in ordered_models})
        else:
            cohesion_df = pd.DataFrame(cohesion_data)
        
        plt.figure(figsize=(5, 7))
        sns.heatmap(cohesion_df, 
                    annot=True, 
                    square=True,
                    cmap='YlGnBu', 
                    fmt='.2f', 
                    linewidths=.5,
                    cbar_kws={'shrink': 0.95}
                    )
        plt.title(f'Category Cohesion by Model - {construct_name}', fontweight='bold')
        plt.xlabel('Different Sentence Transformer Models', fontweight='bold')
        plt.ylabel('Construct Dimensions', fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(construct_output_dir / f'{construct_name}_category_cohesion_heatmap.png', dpi=300)
        plt.close()
    
    print(f"Analysis complete for {construct_name}! Generated visualizations in {construct_output_dir}")


def main(data_directory, categories_csv_path, output_directory="output"):
    """
    Main function to run the analysis.
    
    Parameters:
    data_directory (str): Path to directory containing construct subdirectories with similarity files
    categories_csv_path (str): Path to CSV file with construct categories
    output_directory (str): Directory to save all output files
    """
    # Load thematic categories from CSV
    print("Loading thematic categories...")
    constructs_data = load_thematic_categories(categories_csv_path)
    print(f"Loaded {len(constructs_data)} constructs: {list(constructs_data.keys())}")
    
    # Create main output directory
    Path(output_directory).mkdir(parents=True, exist_ok=True)
    
    # Collect all unique model names across all constructs to ensure consistent ordering
    all_model_names = set()
    for construct_name in constructs_data.keys():
        model_files = find_model_files(data_directory, construct_name)
        all_model_names.update(model_files.keys())
    
    # Define a consistent order for models (you can customize this order)
    model_order = sorted(list(all_model_names))  # Alphabetical order
    # Or define a custom order like:
    # model_order = ['BioBERT', 'Bio_ClinicalBERT', 'ClinicalBert', 'ClinicalSentenceTransformer', 
    #                'Harmony', 'MiniLM', 'MPNet', 'SimCSE']
    
    print(f"Model order for all heatmaps: {model_order}")
    
    # Process each construct separately
    for construct_name, construct_data in constructs_data.items():
        try:
            process_construct(construct_name, construct_data, data_directory, output_directory, model_order)
        except Exception as e:
            print(f"Error processing construct {construct_name}: {e}")
    
    print(f"\n{'='*60}")
    print("All constructs processed!")
    print(f"Results saved in: {output_directory}")
    print(f"{'='*60}")


# Example usage:
if __name__ == "__main__":
    # Update these paths for your setup
    DATA_DIRECTORY = "/Users/julia/Desktop/origami/ENIGMA-PD/code/similarity-analysis/output_similarity-analysis"  # Directory containing model subdirectories
    CATEGORIES_CSV = "/Users/julia/Desktop/origami/ENIGMA-PD/code/similarity-analysis/similarity-analysis-input_without-sleep.csv"  # CSV file with construct definitions
    OUTPUT_DIR = "/Users/julia/Desktop/origami/ENIGMA-PD/code/clustering/output_clustering"        # Output directory
    
    # Run the analysis
    main(DATA_DIRECTORY, CATEGORIES_CSV, OUTPUT_DIR)