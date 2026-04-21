import torch
import time
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base_model = "Qwen/Qwen2.5-7B"
adapter_path = "GraphInstruct/LLaMAFactory/saves/Qwen2.5-7B/lora/train_3_epochs_fix"

tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(base_model, device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code=True)
model = PeftModel.from_pretrained(model, adapter_path)
model.eval()

SYSTEM_PROMPT = "You are a graph reasoning agent. Given a graph description and a question, perform these three steps:\n1. Extract the graph structure as an adjacency list.\n2. Identify the correct graph algorithm tool to use.\n3. Extract the parameters required by that tool.\nOutput your answer as a single valid JSON object."
question = "Node 0 is connected to nodes 1 (weight: 8). Question: Calculate the distance of the shortest path from node 6 to node 7."

eos = tokenizer.eos_token or "<|endoftext|>"
text = f"System: {SYSTEM_PROMPT}{eos}\nHuman: {question}{eos}\nAssistant:"

inputs = tokenizer([text], return_tensors="pt").to(model.device)

print("Starting generation...")
start = time.time()
outputs = model.generate(**inputs, max_new_tokens=40, temperature=0.1, top_p=0.7, do_sample=False, pad_token_id=tokenizer.eos_token_id)
print(f"Done in {time.time()-start:.1f}s")
generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
print("Output:", repr(tokenizer.decode(generated_ids, skip_special_tokens=False)))
