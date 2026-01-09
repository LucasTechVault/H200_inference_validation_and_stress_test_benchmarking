import multiprocessing
import os
import time

# --- WORKER ---
def benchmark_worker(dtype, return_dict):
    import torch
    from vllm import LLM, SamplingParams
    import gc
    
    print(f"\n[Worker] Testing Precision: {dtype}...", flush=True)
    
    try:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
        MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
        TP_SIZE = 2
        
        llm = LLM(
            model=MODEL_ID,
            tensor_parallel_size=TP_SIZE,
            dtype=dtype,  # <--- The Variable
            gpu_memory_utilization=0.90,
            trust_remote_code=True,
            enforce_eager=True
        )
        
        # 500 requests for stability
        dummy_prompt = "Explain quantum entanglement in simple terms." * 3
        sampling_params = SamplingParams(max_tokens=50, temperature=0.0)
        prompts = [dummy_prompt] * 500 

        torch.cuda.synchronize()
        start = time.time()
        
        outputs = llm.generate(prompts, sampling_params)
        
        torch.cuda.synchronize()
        duration = time.time() - start
        
        total_tokens = sum([len(o.outputs[0].token_ids) for o in outputs])
        tps = total_tokens / duration
        
        print(f"[Worker] {dtype} Throughput: {tps:.2f} tok/s", flush=True)
        return_dict[dtype] = tps
        
        del llm
        gc.collect()
        
    except Exception as e:
        print(f"[Worker] Crashed: {e}", flush=True)
        return_dict[dtype] = 0.0

# --- MAIN ---
if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)
    
    DTYPES = ["float16", "bfloat16"]
    manager = multiprocessing.Manager()
    results = manager.dict()

    print(f"---Section 9: Precision Sweep ---")

    for dtype in DTYPES:
        p = multiprocessing.Process(target=benchmark_worker, args=(dtype, results))
        p.start()
        p.join()
        
    print("\n---Precision Report ---")
    print("| Dtype      | Throughput (tok/s) |")
    print("|------------|--------------------|")
    for dt in DTYPES:
        val = results.get(dt, 0.0)
        print(f"| {dt:<10} | {val:<18.2f} |")