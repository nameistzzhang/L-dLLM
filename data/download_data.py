import argparse
import os
import shutil
import json

# 1. Define base and cache directories on the scratch disk
BASE_DIR = "/scratch/aszalay1/tianze/cpt_data"
CACHE_DIR = os.path.join(BASE_DIR, "hf_cache")
os.makedirs(CACHE_DIR, exist_ok=True)

# 2. Force Hugging Face to use the scratch disk for all caching BEFORE importing
os.environ["HF_HOME"] = CACHE_DIR
os.environ["HF_DATASETS_CACHE"] = CACHE_DIR

# Now it is safe to import huggingface libraries
from datasets import load_dataset
from huggingface_hub import hf_hub_download

# Configuration dictionary for the newly added Hugging Face datasets
NEW_DATASETS_CONFIG = {
    "OpenThoughts-114k": {"repo": "open-thoughts/OpenThoughts-114k", "name": None, "data_dir": None, "split": "train"},
    "MathInstruct": {"repo": "TIGER-Lab/MathInstruct", "name": None, "data_dir": None, "split": "train"},
    "starcoder2": {"repo": "bigcode/the-stack-v2", "name": None, "data_dir": "Python", "split": "train"},
    "CodeFeedback": {"repo": "m-a-p/CodeFeedback-Filtered-Instruction", "name": None, "data_dir": None, "split": "train"},
    "HelpSteer2": {"repo": "nvidia/HelpSteer2", "name": None, "data_dir": None, "split": "train"},
    "Magpie": {"repo": "Magpie-Align/Magpie-Pro-300K-Filtered", "name": None, "data_dir": None, "split": "train"},
    "fineweb-edu": {"repo": "HuggingFaceFW/fineweb-edu", "name": "sample-10BT", "data_dir": None, "split": "train"}
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

if dataset in ["fineweb-edu", "starcoder2"]:
    # Special streaming logic for massive pretraining datasets
    config = NEW_DATASETS_CONFIG[dataset]
    output_path = os.path.join(dataset_dir, f"{dataset}.jsonl")
    
    # Target 0.5B tokens for web, 0.75B tokens for code
    if dataset == "fineweb-edu":
        target_docs = 500000 
    else:
        target_docs = 750000
        
    print(f"Starting streaming download for {dataset}...")
    print(f"Targeting {target_docs} documents.")
    
    # Prepare streaming arguments dynamically
    stream_kwargs = {"split": config["split"], "streaming": True}
    if config["name"]:
        stream_kwargs["name"] = config["name"]
    if config["data_dir"]:
        stream_kwargs["data_dir"] = config["data_dir"]
        
    # Enable streaming to bypass caching the entire dataset
    ds = load_dataset(config["repo"], **stream_kwargs)
    
    # Write directly to JSONL on the fly
    with open(output_path, "w", encoding="utf-8") as f:
        for i, example in enumerate(ds):
            if i >= target_docs:
                break
            
            f.write(json.dumps(example, ensure_ascii=False) + "\n")
            
            # Print progress every 50,000 documents
            if (i + 1) % 50000 == 0:
                print(f"Streamed {i + 1} / {target_docs} documents...")
                
    print(f"Successfully saved {dataset} to {output_path}")

elif dataset in NEW_DATASETS_CONFIG:
    # Standard download logic for regular sized datasets
    config = NEW_DATASETS_CONFIG[dataset]
    
    load_kwargs = {"split": config["split"], "cache_dir": CACHE_DIR}
    if config["name"]:
        load_kwargs["name"] = config["name"]
    if config["data_dir"]:
        load_kwargs["data_dir"] = config["data_dir"]
        
    ds = load_dataset(config["repo"], **load_kwargs)
    
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
        filename=f"{split}/{dataset}.json",
        cache_dir=CACHE_DIR
    )
    
    output_path = os.path.join(dataset_dir, f"{dataset}.json")
    shutil.copy(cached_path, output_path)
    print(f"Successfully saved {dataset} to {output_path}")