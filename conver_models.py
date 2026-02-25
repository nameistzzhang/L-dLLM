import argparse
import os
import torch
from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer

# Import from the local models directory
from models.latent_llada.configuration_latent_llada import LatentLLaDAConfig
from models.latent_llada.modeling_latent_llada import LatentLLaDAModelLM
from models.llada.modeling_llada import LLaDAModelLM

def convert_llada_to_latent(llada_path: str, save_path: str):
    print(f"Loading LLaDA model from {llada_path}...")
    
    try:
        # 建议加载为 float16 或 bfloat16 以节省内存，视原模型而定，或者保持 auto
        llada_model = AutoModelForCausalLM.from_pretrained(llada_path, trust_remote_code=True, device_map="cpu", torch_dtype="auto")
        print("Loaded with AutoModelForCausalLM")
    except Exception as e:
        print(f"Error loading with AutoModelForCausalLM: {e}")
        print("Trying LLaDAModelLM directly...")
        try:
             llada_model = LLaDAModelLM.from_pretrained(llada_path, trust_remote_code=True, device_map="cpu", torch_dtype="auto")
        except Exception as e2:
             print(f"Error loading model with LLaDAModelLM: {e2}")
             return

    print("Creating LatentLLaDA config...")
    llada_config = llada_model.config
    config_dict = llada_config.to_dict()
    
    keys_to_remove = ["architectures", "model_type", "_name_or_path", "auto_map"]
    for k in keys_to_remove:
        config_dict.pop(k, None)
    
    latent_config = LatentLLaDAConfig(**config_dict)
    
    print("Initialize LatentLLaDA model...")
    # [建议] 保持和原模型一样的 dtype，否则可能是 fp32
    latent_model = LatentLLaDAModelLM(latent_config).to(llada_model.dtype)
    
    print("Copying weights...")
    llada_state_dict = llada_model.state_dict()
    
    has_model_prefix = any(k.startswith("model.") for k in llada_state_dict.keys())
    # [注意] 这里假设 LatentLLaDAModelLM 内部的主干变量名也是 "model"
    target_has_model_prefix = any(k.startswith("model.") for k in latent_model.state_dict().keys())
    
    new_state_dict = {}
    
    if not has_model_prefix and target_has_model_prefix:
        print("Detected source model missing 'model.' prefix, adding it...")
        for k, v in llada_state_dict.items():
            if k == "lm_head.weight": 
                new_state_dict[k] = v
            else:
                new_state_dict[f"model.{k}"] = v
    else:
        new_state_dict = llada_state_dict

    # Load weights
    missing_keys, unexpected_keys = latent_model.load_state_dict(new_state_dict, strict=False)
    
    print(f"Missing keys: {len(missing_keys)}")
    print(f"Unexpected keys: {len(unexpected_keys)}")
    
    # [修改] 扩充允许缺失的 keys，包含 gate 和 timestep embedding 相关的参数
    # 因为 Latent 模型通常会有这些额外参数，而原版 LLaDA 没有
    allowed_missing_keywords = ["gate", "time_embed", "timestep", "embed_t"] # 增加可能的关键词
    
    real_missing = []
    for k in missing_keys:
        if not any(keyword in k for keyword in allowed_missing_keywords):
            real_missing.append(k)
    
    if real_missing:
        print(f"CRITICAL ERROR: The following CRITICAL keys are missing: {real_missing}")
        print("Conversion Aborted to prevent saving broken model.")
        return 
    else:
        print(f"Verification 1/2 Pass: Missing keys verified as safe (new latent parameters).")

    if unexpected_keys:
        print(f"WARNING: The following keys were in source but NOT in target: {unexpected_keys}")
    else:
        print("Verification 2/2 Pass: No unexpected keys from source model.")

    print(f"Saving LatentLLaDA model to {save_path}...")
    latent_model.save_pretrained(save_path)
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(llada_path, trust_remote_code=True)
        tokenizer.save_pretrained(save_path)
        print("Tokenizer saved.")
    except Exception as e:
        print(f"Could not load/save tokenizer (non-critical): {e}")

    print("Conversion complete.")

if __name__ == "__main__":
    # ... (保持不变)
    parser = argparse.ArgumentParser(description="Convert LLaDA model to LatentLLaDA model")
    parser.add_argument("--llada_path", type=str, required=True, help="Path to the input LLaDA model")
    parser.add_argument("--save_path", type=str, required=True, help="Path to save the LatentLLaDA model")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    # 调用
    convert_llada_to_latent(args.llada_path, args.save_path)