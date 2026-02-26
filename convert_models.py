import torch
import os
import sys
from transformers import AutoModel, AutoConfig, PreTrainedModel, PretrainedConfig

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


def convert_llada_to_latent(model_id="GSAI-ML/LLaDA-8B-Base", save_path="models/converted_latent_llada"):
    """
    Converts a standard LLaDA model checkpoint to a Latent LLaDA model checkpoint.
    
    This function:
    1. Loads the original LLaDA model (backbone).
    2. Initializes a new Latent LLaDA model structure (including the new Gate).
    3. Copies the matching backbone weights from LLaDA to Latent LLaDA.
    4. Leaves the new 'Gate' parameters randomly initialized.
    5. Saves the resulting model.
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
    except Exception as e:
        print(f"Error loading original model: {e}")
        return

    print("Original model loaded successfully.")
    
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
        latent_model.save_pretrained(save_path)
        print("Model saved successfully.")

    return latent_model

if __name__ == "__main__":
    convert_llada_to_latent()
