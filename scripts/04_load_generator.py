import time
import numpy as np
from vllm import LLM, SamplingParams
import os

# --- CONFIG ---
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
TP_SIZE = 2
PROMPT_LEN = 100 # Approximate token count for input
OUTPUT_LEN = 100 # Tokens to generate
CONCURRENCIES = [1, 16, 32, 64, 128, 256, 512, 1024, 2048, 4196] # Stress Levels

def run_benchmark():
    
    # 1. Init Engine
    print(f"Loading {MODEL_ID} with TP={TP_SIZE}")
    llm = LLM(
        model=MODEL_ID,
        tensor_parallel_size=TP_SIZE, # split model across <TP_SIZE> worker processes
        trust_remote_code=True,
        gpu_memory_utilization=0.9 # (0.9 * Total) - Weight memory = Allocation for KV Cache
    )
    
    # 2. Define Workload
    dummy_prompt = "Explain the detailed history of the Roman Empire and its fall, including economic factors, military strategies, and political corruption. " * 3
    sampling_params = SamplingParams(
        max_tokens=OUTPUT_LEN, # hard limit for num tokens allowed to be generated
        temperature=0. # deterministic measurement 0 - always pick highest probability
    )
    
    results = {}
    
    # 3. Batch Sweep loop
    for batch_size in CONCURRENCIES:
        print(f"\nRunning Batch Size: {batch_size}")
        prompts = [dummy_prompt] * batch_size
        
        # Synchronize GPU before start
        import torch
        torch.cuda.synchronize()
    
        start_time = time.time() # begin generation
        outputs = llm.generate(prompts, sampling_params) # generation
        total_time_taken_for_generation = time.time() - start_time # end generation
        
        # 4. Metric Calc
        # Offline mode - Processing work - measure throughput
        total_tokens_generated = sum([len(o.outputs[0].token_ids) for o in outputs])
        tokens_per_sec = total_tokens_generated / total_time_taken_for_generation
        
        print(f"Done in {total_time_taken_for_generation:.2f}s")
        print(f"Throughput: {tokens_per_sec:.2f} tokens / sec")
        
        results[batch_size] = tokens_per_sec
        
    # 5. Report
    print("\n--- Baseline Report Card ---")
    print("| Concurrency | Throughput (tok/s) |")
    print("|-------------|--------------------|")
    for bs, tps in results.items():
        print(f"| {bs:<11} | {tps:<18.2f} |")

if __name__ == "__main__":
    run_benchmark()