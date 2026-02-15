import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["TOKENIZERS_PARALLELISM"] = "true"
import json
import logging
import math
import shutil
import time
from pathlib import Path
from typing import Union

import numpy as np
from PIL import Image
from omegaconf import OmegaConf
import wandb
import torch
from torch.optim import AdamW

from transformers import AutoTokenizer
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed


from train.utils import get_config, flatten_omega_conf, AverageMeter

from models import LatentLLaDAModelLM, LLaDAConfig
from train.prompting_utils import UniversalPrompting
from models.lr_schedulers import get_scheduler
from models.logging import set_verbosity_info, set_verbosity_error

from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from safetensors.torch import load_file

try:
    import apex
    is_apex_available = True
except ImportError:
    is_apex_available = False

logger = get_logger(__name__, log_level="INFO")

def main():

    config = get_config()
    accelerator = Accelerator()

    model_path = config.model.pretrained_model

    logger.info("Loading models, tokenizer and optimizer")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # 使用 from_pretrained 直接加载，strict=False 会忽略不匹配的键（如新增的 time_mlp 等）
    # 这样既能加载预训练权重，又能自动处理分片、safetensors/bin 格式兼容性
    model = LatentLLaDAModelLM.from_pretrained(model_path, strict=False)
    
    logger.info("All weights loaded (with strict=False). New layers remain randomized.")

if __name__ == "__main__":
    main()
