import multiprocessing
import os
import time

# Process WORKER FUNCTION (Runs in a new process every time) 
def benchmark_worker(max_seqs, return_dict):
    import torch
    from vllm import LLM, SamplingParams
    import gc
    
    # Isolate imports to ensure no global state leaks
    print(f"\n[Worker] Initializing engine with max_num_seqs = {max_seqs}...", flush=True)
    
    try:
        # Configs
        os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
        MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
        TP_SIZE = 2
        
        llm = LLM(
            model=MODEL_ID,
            tensor_parallel_size=TP_SIZE,
            max_num_seqs=max_seqs,
            gpu_memory_utilization=0.90,
            trust_remote_code=True,
            enforce_eager=True # Cleaner memory usage for benchmarks
        )
        
        # Prepare Data
        dummy_prompt = "Explain quantum entanglement in simple terms." * 3
        sampling_params = SamplingParams(max_tokens=50, temperature=0.0)
        prompts = [dummy_prompt] * 500 

        # Warmup / Sync
        torch.cuda.synchronize() # force cpu to wait for gpu before continuing
        start = time.time()
        
        # Run Generation
        outputs = llm.generate(prompts, sampling_params)
        
        # End Sync
        torch.cuda.synchronize()
        duration = time.time() - start
        
        # Calculate Stats
        total_tokens = sum([len(o.outputs[0].token_ids) for o in outputs])
        tps = total_tokens / duration
        
        print(f" [Worker] Done - Throughput: {tps:.2f} tok/s", flush=True)
        return_dict[max_seqs] = tps
        
        # Cleanup - Not strictly necessary as process death cleans up
        del llm
        gc.collect()
        
    except Exception as e:
        print(f"[Worker] Crashed: {e}", flush=True)
        return_dict[max_seqs] = 0.0

# --- MAIN CONTROLLER ---
if __name__ == "__main__":
    # 'spawn' REQUIRED for CUDA multiprocessing
    multiprocessing.set_start_method('spawn', force=True)
    
    BATCH_LIMITS = [16, 64, 128, 256, 512, 2048, 32768, 131072, 524288]
    manager = multiprocessing.Manager()
    results = manager.dict()

    print(f"---Starting Batching Sweep Processes ---")

    for seq in BATCH_LIMITS:
        # Create a fresh process for this configuration
        p = multiprocessing.Process(target=benchmark_worker, args=(seq, results))
        p.start()
        p.join() # Wait for the worker to die completely
        
        # If worker died, the OS has cleaned up the VRAM
        # Safely start the next one
        
    print("\n---Optimization Report ---")
    print("| Max Batch | Throughput (tok/s) |")
    print("|-----------|--------------------|")
    for seq in BATCH_LIMITS:
        # Get result from the shared dictionary, default to 0 if failed
        val = results.get(seq, 0.0)
        print(f"| {seq:<9} | {val:<18.2f} |")