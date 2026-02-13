import json
from jinja2 import Template
import os

# llada and mmada:
system_prompts = """<|startoftext|><|start_header_id|>user<|end_header_id|>You need to put your final answer in \\boxed{}. This is the problem:\n{{problem}}<|eot_id|><|startoftext|><|start_header_id|>assistant<|end_header_id|>\n"""
eos_token = "<|eot_id|>"

def get_formatted_prompt(problem_text):
    return Template(system_prompts).render(problem=problem_text)

input_file = "./OpenThoughts-114k.jsonl"
chunk_size = 10000
current_chunk = []
chunk_index = 0

print(f"Start processing {input_file}...")

# Check if input file exists to avoid immediate crash if run
if not os.path.exists(input_file):
    print(f"Warning: {input_file} not found. Please ensure the data file is present.")
else:
    with open(input_file, 'r') as f:
        for line_idx, line in enumerate(f):
            try:
                item = json.loads(line)
                
                # Logic from upper part to extract prompt and response
                if "conversations" in item:
                    raw_prompt = item["conversations"][0]["value"]
                    # Logic from upper part: split by tokens to get the solution content
                    raw_response = item["conversations"][1]["value"].split("<|begin_of_solution|>")[1].split("<|end_of_solution|>")[0]
                    
                    # Logic from lower part to format prompt and response
                    formatted_prompt = get_formatted_prompt(raw_prompt)
                    formatted_response = raw_response + eos_token
                    
                    current_chunk.append({
                        "prompt": formatted_prompt,
                        "response": formatted_response
                    })
                    
                    # Write chunk if size limit reached
                    if len(current_chunk) >= chunk_size:
                        output_filename = f"openthoughts_sft_{chunk_index}.json"
                        print(f"Writing {output_filename} with {len(current_chunk)} items...")
                        with open(output_filename, 'w') as out_f:
                            json.dump(current_chunk, out_f, indent=2)
                        current_chunk = []
                        chunk_index += 1
                else:
                    # Fallback or different format handling if needed, though user stressed specifically this logic
                    pass

            except Exception as e:
                # Handle cases where format might differ or split fails
                # e.g. KeyError or IndexError if structure varies
                # print(f"Error processing line {line_idx}: {e}") 
                continue

    # Write remaining items
    if len(current_chunk) > 0:
        output_filename = f"openthoughts_sft_{chunk_index}.json"
        print(f"Writing {output_filename} with {len(current_chunk)} items...")
        with open(output_filename, 'w') as out_f:
            json.dump(current_chunk, out_f, indent=2)

    print("Done processing.")
