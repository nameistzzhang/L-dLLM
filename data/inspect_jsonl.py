import json
import os
import argparse

def inspect_jsonl(file_path, num_samples=2, max_text_len=300):
    print(f"\n{'='*80}")
    print(f"🚀 Inspecting dataset: {os.path.basename(file_path)}")
    print(f"📁 File path: {file_path}")
    print(f"{'='*80}\n")
    
    if not os.path.exists(file_path):
        print(f"❌ Error: File not found. Please check the path.\n")
        return
        
    with open(file_path, 'r', encoding='utf-8') as f:
        for i in range(num_samples):
            line = f.readline()
            if not line:
                print("End of file reached or file is empty.")
                break
            
            try:
                data = json.loads(line.strip())
                
                # 1. Print data structure (Keys)
                print(f"🔍 [Sample {i+1}] Keys available: {list(data.keys())}")
                
                # 2. Truncate long text and pretty print
                preview_data = {}
                for k, v in data.items():
                    if isinstance(v, str) and len(v) > max_text_len:
                        preview_data[k] = v[:max_text_len] + f"\n... [Text too long, truncated. Original length: {len(v)} characters]"
                    elif isinstance(v, list) and len(v) > 5:
                        # If it's a long conversation list, show only the first few elements
                        preview_data[k] = v[:3] + ["... [List too long, truncated]"]
                    else:
                        preview_data[k] = v
                        
                print(f"📄 [Sample {i+1}] Content preview:")
                print(json.dumps(preview_data, indent=4, ensure_ascii=False))
                print("-" * 60)
                
            except json.JSONDecodeError:
                print(f"❌ Failed to parse line {i+1}. Please ensure it is a valid JSONL format.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect the structure and content of a JSONL file.")
    parser.add_argument("--file", type=str, help="Absolute path to the JSONL file you want to inspect")
    parser.add_argument("--n", type=int, default=2, help="Number of samples to preview")
    args = parser.parse_args()

    if args.file:
        inspect_jsonl(args.file, args.n)
    else:
        # Default testing behavior if no arguments are provided
        base_dir = "/scratch/aszalay1/tianze/cpt_data"
        
        # Example files, modify as needed
        test_files = [
            os.path.join(base_dir, "OpenThoughts-114k", "OpenThoughts-114k.jsonl"),
            os.path.join(base_dir, "MathInstruct", "MathInstruct.jsonl")
        ]
        
        for f in test_files:
            inspect_jsonl(f, num_samples=2)