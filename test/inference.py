import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from datetime import datetime
from accelerate import Accelerator

model_name = "meta-llama/Llama-2-13b-chat-hf"
bnb_config = BitsAndBytesConfig(load_in_4bit=True)
source_csv = "custom_dataset.csv"
prompt_column = "variants"
output_csv_prefix = "model_outputs"

# Initialize accelerator
accelerator = Accelerator()

try:
    df = pd.read_csv(source_csv)
    if accelerator.is_main_process:
        print(f"Loaded {len(df)} rows from {source_csv}")
except Exception as e:
    if accelerator.is_main_process:
        print(f"Error loading CSV: {e}")
    exit()

if accelerator.is_main_process:
    print(f"Processing all {len(df)} samples...")

try:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        dtype=torch.float16,
        device_map="auto",
        quantization_config=bnb_config

    )
    model.eval()
    
    if accelerator.is_main_process:
        print(f"Model {model_name} loaded successfully on {accelerator.num_processes} process(es).")
except Exception as e:
    if accelerator.is_main_process:
        print(f"Error loading model: {e}")
    exit()


chunk_size = len(df) // accelerator.num_processes
start_idx = accelerator.process_index * chunk_size
end_idx = start_idx + chunk_size if accelerator.process_index < accelerator.num_processes - 1 else len(df)
df_chunk = df.iloc[start_idx:end_idx]

output_rows = []
batch_size = 4  

for i in range(0, len(df_chunk), batch_size):
    batch = df_chunk.iloc[i:i + batch_size]
    prompts = batch[prompt_column].tolist()

    try:
        inputs = tokenizer(
            prompts, 
            return_tensors="pt", 
            truncation=True, 
            max_length=1024, 
            padding=True
        ).to(accelerator.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.1
            )

        for j, (idx, row) in enumerate(batch.iterrows()):
            response = tokenizer.decode(
                outputs[j][inputs["input_ids"].shape[1]:], 
                skip_special_tokens=True
            ).strip()
            result = row.to_dict()
            result["model_response"] = response
            output_rows.append(result)

    except Exception as e:
        for idx, row in batch.iterrows():
            result = row.to_dict()
            result["model_response"] = f"[ERROR: {str(e)}]"
            output_rows.append(result)

    if (i + batch_size) % 50 == 0 and accelerator.is_main_process:
        print(f"Process {accelerator.process_index}: Processed {min(i + batch_size, len(df_chunk))}/{len(df_chunk)} samples...")


all_results = accelerator.gather_for_metrics(output_rows)


if accelerator.is_main_process:
    out_df = pd.DataFrame(all_results)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file = f"{output_csv_prefix}_{model_name.split('/')[-1]}_{timestamp}.csv"
    out_df.to_csv(out_file, index=False)
    print(f"\nAll {len(all_results)} responses saved in {out_file}")
