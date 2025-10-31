import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datetime import datetime

model_name = "microsoft/Phi-3-mini-4k-instruct"
source_csv = "roleplay_datasets_clean.csv"
prompt_column = "variants"
output_csv_prefix = "model_outputs"


try:
    df = pd.read_csv(source_csv)
    print(f"Loaded {len(df)} rows from {source_csv}")
except Exception as e:
    print(f"Error loading CSV: {e}")
    exit()

print(f"Processing all {len(df)} samples...")


try:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype)
    if torch.cuda.is_available():
        model = model.cuda()
    model.eval()
    print(f"Model {model_name} loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

output_rows = []
for i, row in df.iterrows():
    prompt = row[prompt_column]
    
    messages = [{"role": "user", "content": prompt}]
    chat_prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    try:
        
        inputs = tokenizer(
            chat_prompt, return_tensors="pt", truncation=True, max_length=2000
        )
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.1
            )
        response = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:], 
            skip_special_tokens=True
        ).strip()
    except Exception as e:
        response = f"[ERROR: {str(e)}]"
    result = row.to_dict()
    result["model_response"] = response
    output_rows.append(result)
    if (i + 1) % 50 == 0:
        print(f"Processed {i + 1}/{len(df)} samples...")

out_df = pd.DataFrame(output_rows)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
out_file = f"{output_csv_prefix}_{model_name.split('/')[-1]}_{timestamp}.csv"
out_df.to_csv(out_file, index=False)
print(f"\nAll {len(output_rows)} responses saved in {out_file}")
