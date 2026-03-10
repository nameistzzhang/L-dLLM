import argparse
import os
import shutil
from datasets import load_dataset
from huggingface_hub import hf_hub_download

# Define the base directory for storing datasets
BASE_DIR = "/scratch/aszalay1/tianze/cpt_data"

# Configuration dictionary for the newly added Hugging Face datasets
NEW_DATASETS_CONFIG = {
    "OpenThoughts-114k": {"repo": "open-thoughts/OpenThoughts-114k", "subset": None, "split": "train"},
    "MathInstruct": {"repo": "TIGER-Lab/MathInstruct", "subset": None, "split": "train"},
    "starcoder2": {"repo": "bigcode/starcoder2-dataset", "subset": "python", "split": "train"},
    "CodeFeedback": {"repo": "m-a-p/CodeFeedback-Filtered-Instruction", "subset": None, "split": "train"},
    "HelpSteer2": {"repo": "nvidia/HelpSteer2", "subset": None, "split": "train"},
    "Magpie": {"repo": "Magpie-Align/Magpie-Pro-300K-Filtered", "subset": None, "split": "train"},
    "fineweb-edu": {"repo": "HuggingFaceFW/fineweb-edu", "subset": "sample-10BT", "split": "train"}
}

# Legacy datasets from Gen-Verse
LEGACY_DATASETS = [
    "PrimeIntellect", "MATH_train", "demon_openr1math", "MATH500", 
    "GSM8K", "AIME2024", "LiveBench", "LiveCodeBench", "MBPP", "HumanEval"
]

parser = argparse.ArgumentParser(description="Download a dataset from HF hub")
parser.add_argument(
    "--dataset",
    choices=LEGACY_DATASETS + list(NEW_DATASETS_CONFIG.keys()),
    required=True,
    help="Which dataset to download"
)
args = parser.parse_args()
dataset = args.dataset

# Create an independent directory for the selected dataset
dataset_dir = os.path.join(BASE_DIR, dataset)
os.makedirs(dataset_dir, exist_ok=True)

if dataset in NEW_DATASETS_CONFIG:
    config = NEW_DATASETS_CONFIG[dataset]
    
    # Load dataset with or without a specific subset
    if config["subset"]:
        ds = load_dataset(config["repo"], config["subset"], split=config["split"])
    else:
        ds = load_dataset(config["repo"], split=config["split"])
    
    # Export to JSONL format inside the dedicated folder
    output_path = os.path.join(dataset_dir, f"{dataset}.jsonl")
    ds.to_json(output_path, force_ascii=False)
    print(f"Successfully saved {dataset} to {output_path}")

else:
    # Handle the original Gen-Verse datasets
    if dataset in ["MATH_train", "PrimeIntellect", "demon_openr1math"]:
        split = "train"
    else:
        split = "test"

    cached_path = hf_hub_download(
        repo_id=f"Gen-Verse/{dataset}",
        repo_type="dataset",
        filename=f"{split}/{dataset}.json"
    )
    
    # Copy the file from HF cache to the dedicated folder
    output_path = os.path.join(dataset_dir, f"{dataset}.json")
    shutil.copy(cached_path, output_path)
    print(f"Successfully saved {dataset} to {output_path}")