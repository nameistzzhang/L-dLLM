import json
import os
import logging
from jinja2 import Template
from transformers import AutoTokenizer
from datasets import Dataset

# =============================================================================
# 0. LOGGING SETUP & DEBUG SWITCH
# =============================================================================
DEBUG_MODE = True  # Set to False when doing the actual 60GB full run
DEBUG_MAX_SAMPLES = 3  # How many extremely detailed sample logs to print per dataset

# Configure the logger
log_level = logging.DEBUG if DEBUG_MODE else logging.INFO
logging.basicConfig(
    level=log_level,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# =============================================================================
# 1. GLOBAL CONFIGURATION
# =============================================================================
MODEL_PATH = "/scratch/aszalay1/tianze/models/latent_llada_8b"
MAX_SEQ_LEN = 4096
NUM_EOS_PADDING = 32

INPUT_BASE_DIR = "/scratch/aszalay1/tianze/cpt_data"
OUTPUT_ARROW_DIR = "/scratch/aszalay1/tianze/cpt_data/3b_pyarrow/"
CACHE_DIR = "/scratch/aszalay1/tianze/cpt_data/hf_cache/"

GENERAL_TEMPLATE = """<|startoftext|><|start_header_id|>user<|end_header_id|>\n{{problem}}<|eot_id|><|startoftext|><|start_header_id|>assistant<|end_header_id|>\n"""

logger.info(f"Loading Tokenizer from {MODEL_PATH}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
EOS_ID = tokenizer.convert_tokens_to_ids("<|eot_id|>") 
EOT_ID = tokenizer.convert_tokens_to_ids("<|endoftext|>")
PAD_ID = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else EOT_ID

logger.info(f"Token IDs resolved -> EOS_ID: {EOS_ID}, PAD_ID: {PAD_ID}")

# =============================================================================
# 2. DATASET ROUTING & FORMATTING
# =============================================================================
def format_sft(prompt, response):
    formatted_prompt = Template(GENERAL_TEMPLATE).render(problem=prompt)
    return {"type": "sft", "prompt": formatted_prompt, "response": response}

def format_pretrain(text):
    return {"type": "pretrain", "text": text}

def parse_line(dataset_name, item):
    """Parses raw JSON dicts into standardized track format."""
    try:
        if dataset_name == "OpenThoughts-114k":
            raw_prompt = item["conversations"][0]["value"]
            val = item["conversations"][1]["value"]
            if "<|begin_of_solution|>" in val and "<|end_of_solution|>" in val:
                raw_response = val.split("<|begin_of_solution|>")[1].split("<|end_of_solution|>")[0].strip()
            else:
                raw_response = val.strip()
            return format_sft(raw_prompt, raw_response)
            
        elif dataset_name == "MathInstruct":
            return format_sft(item["instruction"], item["output"])
            
        elif dataset_name == "CodeFeedback":
            return format_sft(item["query"], item["answer"])
            
        elif dataset_name == "HelpSteer2":
            return format_sft(item["prompt"], item["response"])
            
        elif dataset_name == "Magpie":
            return format_sft(item["conversations"][0]["value"], item["conversations"][1]["value"])
            
        elif dataset_name == "starcoder2":
            return format_pretrain(item["content"])
        
        elif dataset_name == "fineweb-edu":
            return format_pretrain(item["text"])
            
    except Exception as e:
        logger.warning(f"Failed to parse a line in {dataset_name}. Error: {e}")
        return None
    return None

# =============================================================================
# 3. FLOW MATCHING PACKER (Zero-Crossing for SFT)
# =============================================================================
class SafeFlowMatchingPacker:
    def __init__(self, max_seq_len, pad_id):
        self.max_seq_len = max_seq_len
        self.pad_id = pad_id
        self.current_doc_id = 0
        self.total_chunks_yielded = 0
        self.reset_buffer()

    def reset_buffer(self):
        self.buffer_input_ids = []
        self.buffer_diffusion_mask = []
        self.buffer_position_ids = []
        self.buffer_document_ids = []

    def add_document(self, input_ids, diffusion_mask, is_sft, doc_idx=None):
        doc_len = len(input_ids)
        chunks_to_yield = []
        
        # Rule 1: Drop oversized SFT documents entirely
        if is_sft and doc_len > self.max_seq_len:
            logger.debug(f"[Packer Drop] Dropping SFT Doc {doc_idx} because its length ({doc_len}) > MAX ({self.max_seq_len}).")
            return chunks_to_yield

        current_len = len(self.buffer_input_ids)
        
        # Rule 2: Dynamic Padding for SFT to prevent semantic splitting
        if is_sft and (current_len + doc_len > self.max_seq_len):
            pad_len = self.max_seq_len - current_len
            logger.debug(f"[Packer Pad] SFT Doc {doc_idx} (len {doc_len}) crosses boundary. Padding current buffer (len {current_len}) with {pad_len} PADs.")
            
            self.buffer_input_ids.extend([self.pad_id] * pad_len)
            self.buffer_diffusion_mask.extend([-1] * pad_len) 
            self.buffer_position_ids.extend([0] * pad_len)
            self.buffer_document_ids.extend([self.current_doc_id] * pad_len)
            self.current_doc_id += 1
            
            chunks_to_yield.append(self._flush_current_buffer())
            current_len = 0

        # Rule 3: Add document to buffer
        position_ids = list(range(doc_len))
        document_ids = [self.current_doc_id] * doc_len
        
        self.buffer_input_ids.extend(input_ids)
        self.buffer_diffusion_mask.extend(diffusion_mask)
        self.buffer_position_ids.extend(position_ids)
        self.buffer_document_ids.extend(document_ids)
        self.current_doc_id += 1
        
        # Rule 4: Slice into perfect chunks if buffer exceeds max_seq_len
        while len(self.buffer_input_ids) >= self.max_seq_len:
            chunks_to_yield.append(self._flush_current_buffer(slice_only=True))
            
        return chunks_to_yield

    def _flush_current_buffer(self, slice_only=False):
        chunk = {
            "input_ids": self.buffer_input_ids[:self.max_seq_len],
            "diffusion_mask": self.buffer_diffusion_mask[:self.max_seq_len],
            "position_ids": self.buffer_position_ids[:self.max_seq_len],
            "document_ids": self.buffer_document_ids[:self.max_seq_len]
        }
        self.total_chunks_yielded += 1
        
        if slice_only:
            # Keep remainder
            self.buffer_input_ids = self.buffer_input_ids[self.max_seq_len:]
            self.buffer_diffusion_mask = self.buffer_diffusion_mask[self.max_seq_len:]
            self.buffer_position_ids = self.buffer_position_ids[self.max_seq_len:]
            self.buffer_document_ids = self.buffer_document_ids[self.max_seq_len:]
            logger.debug(f"[Packer Chunk] Yielded chunk #{self.total_chunks_yielded}. Remainder left in buffer: {len(self.buffer_input_ids)}")
        else:
            # Full wipe
            self.reset_buffer()
            logger.debug(f"[Packer Chunk] Yielded chunk #{self.total_chunks_yielded}. Buffer reset to 0.")
            
        return chunk

    def flush_final(self):
        if not self.buffer_input_ids:
            return []
            
        pad_len = self.max_seq_len - len(self.buffer_input_ids)
        logger.info(f"Flushing final chunk. Padding with {pad_len} PADs to reach {self.max_seq_len}.")
        
        self.buffer_input_ids.extend([self.pad_id] * pad_len)
        self.buffer_diffusion_mask.extend([-1] * pad_len)
        self.buffer_position_ids.extend([0] * pad_len)
        self.buffer_document_ids.extend([self.current_doc_id] * pad_len)
        
        return [self._flush_current_buffer(slice_only=False)]

# =============================================================================
# 4. GENERATOR PIPELINE
# =============================================================================
DATASETS_TO_PROCESS = [
    "OpenThoughts-114k", 
    "MathInstruct", 
    "CodeFeedback", 
    "HelpSteer2", 
    "Magpie", 
    "starcoder2"
]

def flow_matching_data_generator():
    packer = SafeFlowMatchingPacker(max_seq_len=MAX_SEQ_LEN, pad_id=PAD_ID)
    
    for ds_name in DATASETS_TO_PROCESS:
        file_path = os.path.join(INPUT_BASE_DIR, ds_name, f"{ds_name}.jsonl")
        if not os.path.exists(file_path):
            logger.warning(f"Dataset file not found: {file_path}. Skipping.")
            continue
            
        logger.info(f"--- Started streaming and processing: {ds_name} ---")
        samples_logged_this_ds = 0
        doc_counter = 0
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip(): continue
                doc_counter += 1
                
                raw_item = json.loads(line)
                parsed = parse_line(ds_name, raw_item)
                if not parsed: continue
                
                # --- Tokenization and Mask Generation ---
                if parsed["type"] == "sft":
                    prompt_ids = tokenizer.encode(parsed["prompt"], add_special_tokens=False)
                    response_ids = tokenizer.encode(parsed["response"], add_special_tokens=False)
                    
                    input_ids = prompt_ids + response_ids + ([EOS_ID] * NUM_EOS_PADDING)
                    diffusion_mask = ([0] * len(prompt_ids)) + ([1] * (len(response_ids) + NUM_EOS_PADDING))
                    
                    is_sft_flag = True
                    
                else: # pretrain
                    text_ids = tokenizer.encode(parsed["text"], add_special_tokens=False)
                    
                    input_ids = text_ids + ([EOT_ID] * NUM_EOS_PADDING)
                    diffusion_mask = [1] * len(input_ids)
                    
                    is_sft_flag = False

                # ---> Detailed Debug Logger for the first few samples <---
                if samples_logged_this_ds < DEBUG_MAX_SAMPLES:
                    logger.debug(f"\n[{ds_name} | Doc #{doc_counter}] Detailed Inspection:")
                    if is_sft_flag:
                        logger.debug(f"  Prompt string (trunc): {parsed['prompt'][:100]!r}...")
                        logger.debug(f"  Response string (trunc): {parsed['response'][:100]!r}...")
                        logger.debug(f"  Token Lens -> Prompt: {len(prompt_ids)}, Response: {len(response_ids)}, Total+EOS: {len(input_ids)}")
                        num_zeros = diffusion_mask.count(0)
                        num_ones = diffusion_mask.count(1)
                        logger.debug(f"  Mask Checks -> Zeros (Lock): {num_zeros}, Ones (Diff): {num_ones}. Match? {num_zeros == len(prompt_ids)}")
                    else:
                        logger.debug(f"  Text string (trunc): {parsed['text'][:150]!r}...")
                        logger.debug(f"  Token Lens -> Text: {len(text_ids)}, Total+EOS: {len(input_ids)}")
                        logger.debug(f"  Mask Checks -> All Ones? {all(m == 1 for m in diffusion_mask)}")
                    samples_logged_this_ds += 1

                # Feed to packer
                chunks = packer.add_document(input_ids, diffusion_mask, is_sft=is_sft_flag, doc_idx=doc_counter)
                
                # Yield ready chunks to Hugging Face Datasets
                for chunk in chunks:
                    yield chunk
                    
    # Yield the final remaining items padded to 4096
    for chunk in packer.flush_final():
        yield chunk

# =============================================================================
# 5. EXECUTION
# =============================================================================
if __name__ == "__main__":
    logger.info("🚀 Initializing PyArrow Dataset building process...")
    if DEBUG_MODE:
        logger.info("⚠️ DEBUG MODE IS ON. Extremely detailed logs will be printed for the first few samples.")
    
    # Consume generator
    hf_dataset = Dataset.from_generator(flow_matching_data_generator, cache_dir=CACHE_DIR)
    
    logger.info(f"\n📊 Process Complete! Total 4096-length chunks generated: {len(hf_dataset)}")
    logger.info(f"💾 Saving to disk at: {OUTPUT_ARROW_DIR}")
    
    hf_dataset.save_to_disk(OUTPUT_ARROW_DIR)
    
    logger.info("✅ Done! Data is successfully packed and ready for DeepSpeed.")