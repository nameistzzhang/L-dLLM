import argparse
import os
import shutil
import json
from itertools import islice

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
    "starcoder2": {"repo": "bigcode/starcoderdata", "name": None, "data_dir": "python", "split": "train"},
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

if dataset == "fineweb-edu":
    config = NEW_DATASETS_CONFIG[dataset]
    # Use flush=True to ensure the log is written immediately
    print(f"Starting optimized multi-threaded download for {dataset}...", flush=True)
    
    # 1. Download and cache using HF's optimized engine
    load_kwargs = {
        "split": config["split"], 
        "name": config["name"], 
        "cache_dir": CACHE_DIR,
        "num_proc": 8  # Enable multiprocessing for much faster downloading and processing
    }
    
    # This will download the parquet files to CACHE_DIR efficiently
    ds = load_dataset(config["repo"], **load_kwargs)
    
    output_path = os.path.join(dataset_dir, f"{dataset}.jsonl")
    print(f"Exporting dataset to {output_path}...", flush=True)
    
    # 2. Export to JSONL using Arrow memory-mapping
    ds.to_json(output_path, force_ascii=False, num_proc=8)
    
    print(f"Successfully saved {dataset} to {output_path}", flush=True)

elif dataset in NEW_DATASETS_CONFIG:
    config = NEW_DATASETS_CONFIG[dataset]
    
    # Prepare keyword arguments dynamically
    load_kwargs = {"split": config["split"], "cache_dir": CACHE_DIR}
    if config["name"]:
        load_kwargs["name"] = config["name"]
    if config["data_dir"]:
        load_kwargs["data_dir"] = config["data_dir"]
        
    # Load dataset with explicitly specified cache_dir and configs
    ds = load_dataset(config["repo"], **load_kwargs)
    
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

    # Download with explicitly specified cache_dir
    cached_path = hf_hub_download(
        repo_id=f"Gen-Verse/{dataset}",
        repo_type="dataset",
        filename=f"{split}/{dataset}.json",
        cache_dir=CACHE_DIR
    )
    
    # Copy the file from HF cache to the dedicated folder
    output_path = os.path.join(dataset_dir, f"{dataset}.json")
    shutil.copy(cached_path, output_path)
    print(f"Successfully saved {dataset} to {output_path}")