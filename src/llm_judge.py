import pandas as pd
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, BitsAndBytesConfig
from accelerate import Accelerator


csv_path = "eval_dataset_falcon7b.csv"   
input_column = "llamaguard3_input"
output_column = "llamaguard3_label"
model_name = "meta-llama/Llama-Guard-3-8B"       

accelerator = Accelerator()
local_idx = accelerator.local_process_index
num_proc = accelerator.num_processes

bnb_config = BitsAndBytesConfig(load_in_4bit=True)
if accelerator.is_main_process:
    print(f"[INFO] Loading {model_name} on {num_proc} GPUs.")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(
    model_name, device_map="auto", quantization_config=bnb_config
)
model.eval()

df = pd.read_csv(csv_path)
chunk_size = len(df) // num_proc
start_idx = local_idx * chunk_size
end_idx = start_idx + chunk_size if local_idx < num_proc - 1 else len(df)
df_chunk = df.iloc[start_idx:end_idx]

labels = []
for idx, input_text in tqdm(enumerate(df_chunk[input_column]), total=len(df_chunk)):
    inputs = tokenizer(
        input_text, return_tensors="pt", truncation=True, max_length=1024
    ).to(accelerator.device)
    with torch.no_grad():
        logits = model(**inputs).logits
        label = logits.argmax().item()
    labels.append(label)
    if idx < 5 or idx % 200 == 0:
        print(f"[DEBUG] Proc {local_idx} Row {start_idx+idx}: {input_text[:60]}... → Label: {label}")

df_chunk[output_column] = labels

all_results = accelerator.gather_for_metrics(df_chunk.to_dict(orient='records'))

if accelerator.is_main_process:
    out_df = pd.DataFrame(all_results)
    out_path = "llamaguard3_results_falcon7b.csv"
    out_df.to_csv(out_path, index=False)
    print(f"[INFO] Results written to: {out_path}")
