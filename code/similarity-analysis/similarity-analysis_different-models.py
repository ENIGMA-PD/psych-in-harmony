from sentence_transformers import SentenceTransformer, util
import pandas as pd
import sys
import io
import glob
import os

# === Load single CSV file ===
csv_file_path = '/Users/julia/Desktop/origami/ENIGMA-PD/code/harmonization/similarity-analysis-input_without-sleep.csv'

# Check if file exists
if not os.path.exists(csv_file_path):
    print(f"ERROR: File does not exist: {csv_file_path}")
    print("Please check the path and try again.")
    sys.exit(1)

print(f"Loading CSV file: {os.path.basename(csv_file_path)}")

try:
    df_all = pd.read_csv(csv_file_path)
    print(f"Successfully loaded file with shape: {df_all.shape}")
    print(f"Columns: {list(df_all.columns)}")
    
    # Get unique questionnaires for reference
    questionnaire_names = df_all["questionnaire_name"].unique().tolist()
    print(f"Found questionnaires: {questionnaire_names}")
    
except Exception as e:
    print(f"ERROR loading {csv_file_path}: {e}")
    sys.exit(1)

# === Clean and prepare data ===
print(f"Combined data shape: {df_all.shape}")
print(f"Combined data columns: {list(df_all.columns)}")

# Check if required columns exist
required_columns = ["question_text", "questionnaire_name", "question_number", "construct"]
missing_columns = [col for col in required_columns if col not in df_all.columns]

if missing_columns:
    print(f"ERROR: Missing required columns: {missing_columns}")
    print("Available columns:", list(df_all.columns))
    sys.exit(1)

df_all["question_text"] = df_all["question_text"].astype(str)

# Create item_id in the format: questionnaire_question_number (e.g., "BDI_01")
# Make sure question_number doesn't already contain the questionnaire name
df_all["question_number_clean"] = df_all["question_number"].astype(str)

# Check if question_number already contains questionnaire prefix and clean if needed
def clean_question_number(row):
    questionnaire = str(row["questionnaire_name"])
    question_num = str(row["question_number"])
    
    # If question_number already starts with questionnaire name, use as is
    if question_num.startswith(questionnaire + "_"):
        return question_num
    # Otherwise, combine them
    else:
        return f"{questionnaire}_{question_num}"

df_all["item_id"] = df_all.apply(clean_question_number, axis=1)

print(f"Sample of loaded data:")
print(df_all[["construct", "questionnaire_name", "question_number", "item_id", "question_text"]].head())

# Verify item_id format
print(f"\nSample item_ids:")
sample_ids = df_all["item_id"].head(10).tolist()
for item_id in sample_ids:
    print(f"  {item_id}")

# Get unique constructs
constructs = df_all["construct"].unique()
print(f"Found constructs: {list(constructs)}")
print(f"Total questions: {len(df_all)}")

# Print breakdown by construct
for construct in constructs:
    count = len(df_all[df_all["construct"] == construct])
    print(f"  {construct}: {count} questions")

models = {
    # high-performance general models
    "MiniLM": "sentence-transformers/all-MiniLM-L6-v2",
    "MPNet": "sentence-transformers/all-mpnet-base-v2",
    "SimCSE": "princeton-nlp/sup-simcse-bert-base-uncased",
    # specialized clinical models
    "ClinicalBert": "medicalai/ClinicalBERT",
    "BioClinicalBERT": "menadsa/S-Bio_ClinicalBERT",
    "ClinicalSentenceTransformer": "Shobhank-iiitdwd/Clinical_sentence_transformers_mpnet_base_v2",
    "Bio_ClinicalBERT": "emilyalsentzer/Bio_ClinicalBERT",
    "BioBERT": "dmis-lab/biobert-base-cased-v1.2",
    # mental health specific
    "Harmony": "harmonydata/mental_health_harmonisation_1",
    "MentalRoberta": "mental/mental-roberta-base",
}

# Redirect stdout to capture print output
original_stdout = sys.stdout
captured_output = io.StringIO()
sys.stdout = captured_output

results = {}

# Process each construct separately
for construct in constructs:
    print(f"\n{'='*50}")
    print(f"Processing construct: {construct}")
    print(f"{'='*50}")
    
    # Filter data for current construct
    df_construct = df_all[df_all["construct"] == construct].copy()
    
    if len(df_construct) == 0:
        print(f"No data found for construct: {construct}")
        continue
    
    texts = df_construct["question_text"].tolist()
    item_ids = df_construct["item_id"].tolist()  # Use cleaned item_ids
    
    print(f"Processing {len(texts)} questions for {construct}")
    print(f"Sample questions for {construct}:")
    for i, (item_id, text) in enumerate(zip(item_ids[:3], texts[:3])):
        print(f"  {item_id}: {text}")
    
    # Verify no duplicates in item_ids for this construct
    if len(item_ids) != len(set(item_ids)):
        print(f"WARNING: Duplicate item_ids found in {construct}")
        duplicates = [x for x in set(item_ids) if item_ids.count(x) > 1]
        print(f"Duplicates: {duplicates}")
    
    # Process each model for this construct
    for model_name, model_path in models.items():
        print(f"\nModel: {model_name} - Construct: {construct}")
        
        try:
            model = SentenceTransformer(model_path)
            
            # Use question texts for embeddings, but item_ids for indexing
            embeddings = model.encode(texts, convert_to_tensor=True, normalize_embeddings=True)
            similarity_matrix = util.pytorch_cos_sim(embeddings, embeddings)
            
            # Create DataFrame with single-format item_ids (e.g., "BDI_01", not "BDI_BDI_01")
            df_sim = pd.DataFrame(
                similarity_matrix.cpu().numpy(), 
                index=item_ids, 
                columns=item_ids
            )
            
            print(f"Created similarity matrix with index format:")
            print(f"  Sample indices: {df_sim.index[:3].tolist()}")
            print(f"  Sample columns: {df_sim.columns[:3].tolist()}")
            
            # Create output directory for this construct if it doesn't exist
            construct_output_dir = f"/Users/julia/Desktop/origami/ENIGMA-PD/code/similarity-analysis/output_similarity-analysis/{construct}_similarity-analysis_output"
            os.makedirs(construct_output_dir, exist_ok=True)
            
            # Save this model's results as a CSV file
            csv_filename = f"{construct_output_dir}/{model_name}_{construct}_similarity-results.csv"
            df_sim.round(2).to_csv(csv_filename)
            print(f"Saved {model_name} similarity matrix for {construct} to {csv_filename}")
            
            # Store in dictionary with construct-specific key
            results[f"{model_name}_{construct}"] = df_sim.round(2)
            
            print(f"Similarity matrix shape for {construct}: {df_sim.shape}")
            print("Sample similarities:")
            print(df_sim.iloc[:3, :3].round(2))
            
        except Exception as e:
            print(f"Error processing {model_name} for {construct}: {str(e)}")
            continue

# Restore stdout
sys.stdout = original_stdout

# Save the captured output to a file
with open('model_comparison_results_by_construct.txt', 'w') as f:
    f.write(captured_output.getvalue())

print("Results saved to model_comparison_results_by_construct.txt")
print(f"Processed {len(constructs)} constructs: {list(constructs)}")
print(f"Generated {len(results)} similarity matrices")

# Show final verification of format
print(f"\nFinal verification - sample of generated similarity matrix indices:")
if results:
    sample_result_key = list(results.keys())[0]
    sample_matrix = results[sample_result_key]
    print(f"Sample from {sample_result_key}:")
    print(f"  Indices: {sample_matrix.index[:5].tolist()}")
    print(f"  Columns: {sample_matrix.columns[:5].tolist()}")