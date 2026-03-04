import torch
import os
import sys
import shutil
import argparse
from transformers import AutoModel, AutoConfig, AutoTokenizer

# Ensure the root directory is in sys.path so we can import models if running as script
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

try:
    from models.latent_llada.modeling_latent_llada import LatentLLaDAModelLM
    from models.latent_llada.configuration_latent_llada import LatentLLaDAConfig
except ImportError:
    # Attempt relative import assuming running from root
    from .models.latent_llada.modeling_latent_llada import LatentLLaDAModelLM
    from .models.latent_llada.configuration_latent_llada import LatentLLaDAConfig


def convert_llada_to_latent(model_id="/scratch/aszalay1/tianze/models/llada_8b_instruct", save_path="/scratch/aszalay1/tianze/models/latent_llada_8b"):
    """
    Converts a standard LLaDA model checkpoint to a Latent LLaDA model checkpoint.
    
    This function:  
    1. Loads the original LLaDA model (backbone).
    2. Initializes a new Latent LLaDA model structure (including the new Gate).
    3. Copies the matching backbone weights from LLaDA to Latent LLaDA.
    4. Leaves the new 'Gate' parameters randomly initialized.
    5. Saves the resulting model along with tokenizer and configuration files.
    """
    print(f"Loading Original LLaDA model from: {model_id} ...")
    
    # Load original model
    try:
        orig_model = AutoModel.from_pretrained(
            model_id, 
            trust_remote_code=True, 
            torch_dtype=torch.bfloat16,
            device_map="cpu" 
        )
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    except Exception as e:
        print(f"Error loading original model or tokenizer: {e}")
        return

    print("Original model and tokenizer loaded successfully.")
    
    # Create Latent Config based on Original Config
    print("Creating Latent LLaDA configuration...")
    orig_config = orig_model.config
    latent_config_dict = orig_config.to_dict()
    
    # Update model_type in config
    latent_config_dict['model_type'] = 'latent_llada'
    
    # Create LatentLLaDAConfig object
    latent_config = LatentLLaDAConfig(**latent_config_dict)
    
    # Initialize Latent LLaDA Model
    print("Initializing Latent LLaDA model with random weights (for new components)...")
    latent_model = LatentLLaDAModelLM(latent_config, init_params=True)
    
    # Move latent model to same dtype as original
    latent_model.to(dtype=orig_model.dtype)

    print("Copying compatible weights from Original to Latent model...")
    orig_state_dict = orig_model.state_dict()
    latent_state_dict = latent_model.state_dict()
    
    # Copy weights
    new_state_dict = latent_state_dict.copy()
    matched_keys = []
    skipped_keys = []
    shape_mismatch_keys = []

    for key, param in orig_state_dict.items():
        if key in latent_state_dict:
            if param.shape == latent_state_dict[key].shape:
                new_state_dict[key] = param
                matched_keys.append(key)
            else:
                shape_mismatch_keys.append((key, param.shape, latent_state_dict[key].shape))
        else:
            skipped_keys.append(key)
            
    # Load the updated state dict into the latent model
    latent_model.load_state_dict(new_state_dict, strict=False)
    
    print(f"Weights transfer complete.")
    print(f"  - Matched and copied: {len(matched_keys)} parameters")
    print(f"  - Skipped (not in Latent): {len(skipped_keys)} parameters")
    
    if shape_mismatch_keys:
        print(f"  - Shape Mismatches: {len(shape_mismatch_keys)}")
        for k, s1, s2 in shape_mismatch_keys:
            print(f"    {k}: {s1} -> {s2}")
            
    # Identify new components (initialized randomly)
    latent_only_keys = [k for k in latent_state_dict.keys() if k not in orig_state_dict]
    if latent_only_keys:
        print(f"\n[INFO] The following components are new in Latent LLaDA and remain randomly initialized:")
        gate_params = [k for k in latent_only_keys if 'gate' in k]
        other_params = [k for k in latent_only_keys if 'gate' not in k]
        
        if gate_params:
            print(f"  - Gate parameters ({len(gate_params)} tensors)")
        if other_params:
            print(f"  - Other parameters ({len(other_params)} tensors): {other_params[:5]} ...")

    # Save the converted model
    if save_path:
        print(f"\nSaving converted Latent LLaDA model to: {save_path}")
        os.makedirs(save_path, exist_ok=True)
        
        # 1. Register auto class in config to ensure loading via AutoModel works with trust_remote_code=True
        LatentLLaDAConfig.register_for_auto_class()
        LatentLLaDAModelLM.register_for_auto_class("AutoModel")

        # 2. Save model and tokenizer
        latent_model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)
        
        # 3. Copy python modeling files to destination for portability
        # We need to copy `modeling_latent_llada.py` and `configuration_latent_llada.py`
        try:
            # Determine source paths based on imports
            # Note: sys.modules may have different paths depending on how it was imported, 
            # but usually __file__ on the module works
            model_src = sys.modules[LatentLLaDAModelLM.__module__].__file__
            config_src = sys.modules[LatentLLaDAConfig.__module__].__file__
            
            shutil.copy(model_src, os.path.join(save_path, os.path.basename(model_src)))
            shutil.copy(config_src, os.path.join(save_path, os.path.basename(config_src)))
            print(f"Copied modeling files to {save_path}")
        except Exception as e:
            print(f"Warning: Could not copy modeling files automatically: {e}")

        print("Model saved successfully.")

    return latent_model

def export_checkpoint_to_hf(checkpoint_dir, save_path):
    """
    Converts an accelerate training checkpoint into a clean Hugging Face model release.
    
    This function:
    1. Loads the model weights and tokenizer from the training checkpoint.
    2. Discards all accelerate/optimizer states (which take up massive disk space).
    3. Registers the auto classes to ensure `trust_remote_code=True` works.
    4. Saves the pure model, tokenizer, and copies the necessary Python modeling files.
    """
    print(f"Loading model and tokenizer from training checkpoint: {checkpoint_dir} ...")
    
    # 1. Load the model and tokenizer from the checkpoint
    try:
        # Load directly using your custom class to ensure correct architecture
        model = LatentLLaDAModelLM.from_pretrained(
            checkpoint_dir, 
            torch_dtype=torch.bfloat16, 
            device_map="cpu"
        )
        tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir, trust_remote_code=True)
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return

    print("Checkpoint loaded successfully. Preparing for clean HF export...")

    # 2. Create the clean output directory
    os.makedirs(save_path, exist_ok=True)
    print(f"Saving pure HF model to: {save_path}")

    # 3. Register auto classes so it can be loaded natively via AutoModel
    LatentLLaDAConfig.register_for_auto_class()
    LatentLLaDAModelLM.register_for_auto_class("AutoModel")

    # 4. Save the model and tokenizer (This step drops all optimizer/accelerator states)
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)

    # 5. Copy the custom Python modeling files for portability
    try:
        model_src = sys.modules[LatentLLaDAModelLM.__module__].__file__
        config_src = sys.modules[LatentLLaDAConfig.__module__].__file__
        
        shutil.copy(model_src, os.path.join(save_path, os.path.basename(model_src)))
        shutil.copy(config_src, os.path.join(save_path, os.path.basename(config_src)))
        print(f"Copied custom modeling files ({os.path.basename(model_src)}, {os.path.basename(config_src)}) to {save_path}")
    except Exception as e:
        print(f"Warning: Could not copy modeling files automatically: {e}")

    print("\n[SUCCESS] Checkpoint successfully exported to a clean Hugging Face format!")
    print(f"You can now load this model using:\n  AutoModel.from_pretrained('{save_path}', trust_remote_code=True)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Latent LLaDA Model Conversion and Export Utility")
    
    # Create sub-parsers for different modes of operation
    subparsers = parser.add_subparsers(dest="command", required=True, help="Choose the conversion mode")

    # --- Mode 1: Initialize Latent LLaDA from original LLaDA ---
    parser_init = subparsers.add_parser(
        "init", 
        help="Convert standard LLaDA model to Latent LLaDA structure (randomly initialized gate)"
    )
    parser_init.add_argument(
        "--model_id", 
        type=str, 
        default="/scratch/aszalay1/tianze/models/llada_8b_instruct",
        help="Path or HuggingFace ID of the original LLaDA model"
    )
    parser_init.add_argument(
        "--save_path", 
        type=str, 
        default="/scratch/aszalay1/tianze/models/latent_llada_8b",
        help="Directory to save the new Latent LLaDA model"
    )

    # --- Mode 2: Export trained Accelerate checkpoint to pure HF model ---
    parser_export = subparsers.add_parser(
        "export", 
        help="Export an accelerate training checkpoint to a clean Hugging Face format"
    )
    parser_export.add_argument(
        "--checkpoint_dir", 
        type=str, 
        required=True,
        help="Path to the accelerate checkpoint directory (e.g., .../checkpoint-epoch-10)"
    )
    parser_export.add_argument(
        "--save_path", 
        type=str, 
        required=True,
        help="Directory to save the cleaned HF model release"
    )

    args = parser.parse_args()

    # Execute the chosen command
    if args.command == "init":
        print(">>> Running Initialization Mode <<<")
        convert_llada_to_latent(model_id=args.model_id, save_path=args.save_path)
        
    elif args.command == "export":
        print(">>> Running Checkpoint Export Mode <<<")
        export_checkpoint_to_hf(checkpoint_dir=args.checkpoint_dir, save_path=args.save_path)