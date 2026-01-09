import os
import gc
import torch
from vllm import LLM

# --- CONFIG ---
os.environ["CUDA_VISIBLE_DEVICES"] = "2,3" # Desired GPU
os.environ["VLLM_ALLOW_LONG_MAX_MODEL_LEN"] = "1" # allow vLLM to allow ctx_len larger than model

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
TP_SIZE = 2

# Standard RAG -> 4k - 8k
# Long Context -> 32k
# CONTEXT_SIZES = [32768, 65536, 131072, 262144]
CONTEXT_SIZES = [524288, 1048576]

def run_stress_test():
    print(f"Starting Context-Window Stress Test")
    max_passed = 0
    
    for ctx_len in CONTEXT_SIZES:
        try:
            
            # 1. Init vLLM engine with stress-test context size
            llm = LLM(
                model=MODEL_ID,
                tensor_parallel_size=TP_SIZE,
                max_model_len=ctx_len,
                gpu_memory_utilization=0.95,
                trust_remote_code=True,
                enforce_eager=True # don't use graph memory, fit larger model at expense of speed
            )
            
            # 2. Verify generation
            print(f"PASSED: H200 allocated KV blocks for {ctx_len} tokens")
            max_passed = ctx_len
            
            # 3. clean up for next run
            del llm
            gc.collect()
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"FAILED at {ctx_len} tokens.")
            print(f" Error: {e}")
            break
            
        
    print(f"Final Results: Max Stable Context: {max_passed} tokens")

if __name__ == "__main__":
    run_stress_test()