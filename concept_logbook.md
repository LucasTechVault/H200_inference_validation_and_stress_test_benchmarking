##### 8-Jan-2026 14:31

## Section 1 - Hardware & Software Profiling Learnings

- run_cmd(cmd: list[str], timeout: int=30) : helper to execute shell commands in python
- torch_stack_info() -> dict[str, Any]

  - torch_version | device count |
  - torch_cuda_build | nccl_version |
  - cuda_available | gpus [list]

- p2p_matrix (check if gpus linked)

  - torch.cuda.can_device_access_peer(i, j)
  - torch.cuda.set_device(i) | torch.cuda.enable_peer_access(j) -> minimum effort enablement

- nvidia_smi_bundle() -> query gpu using nvidia commands, using run_cmd() helper specified

  - `nvidia-smi`
  - `nvidia-smi topo -m`
  - `nvidia-smi` + many GPU related queries

## Section 2 - Compute & Bandwidth Benchmarking Learnings (8-Jan-26 15:52)

**GEMM Benchmark Learnings:**

- time_cuda_op_ms(op_fn, device, warmup, iters)
  - op_fn -> C = A @ B (return C else modern compilers will not compile unused references)
- gemm_tflops (2 \* M \_ N \* K) - fused multiply-add (2 operations)
  - A = MxK [2, 5, 7] (1 x 3)
  - B = KxN [3, 11, 13]^T (3 x 1)
  - C = MxK \* KxN -> MxN (2\*3) \+ (5\*11) + (7\*13) - fused multiply add

**P2P Direct Benchmark Learnings:**

- op_fn -> dst_t.copy(src_t) -> PyTorch API
  - will work only for local GPU copy
  - check if P2P (DMA) allowed -> NVLink or PCIe
  - else, create bridge via GPU -> CPU -> GPU

**NCCL Remote GPU Communication Learnings:**

- Pytorch uses:
  - dist.init_process_group(backend="nccl", init_method=...)
  - dist.all_reduce(x, op=dist.ReduceOp.SUM)
    - takes tensor x from every GPU, sums them, distribute final sum to all GPU
  - dist.barrier(): blocks process until every GPU arrives at declaration
    - for benchmarking
  - NCCL -> detects NVLinks, PCIe switches, Network interface (Infiniband / RoCE)
    - Ring Algorithm
    - Tree Algorithm

## Section 3 - vLLM initialization Learnings (8-Jan-26 17:34)

- ensuring vLLM can start up properly
  - dependency check - vLLM works with CUDA version of remote
  - memory check - vLLM loads model appropriately
  - Tokenizer check - CPU-based tokenizer compatible with GPU

**vLLM Concept**:

- vLLM takes over entire GPU management lifecycle

---

##### How vLLM Works (Flow):

1. Model Loading: Model weights are loaded from disk to GPU VRAM
2. Profiling: vLLM fires a dummy request to see how much RAM model weights take.
3. Block Allocation: vLLM analyzes allocated memory (e.g. 120GB) and divides 120GB into thousands of small, empty "Pages" (blocks). These are empty blocks waiting to hold future conversation history (KV Cache)
4. LLM class runs the tokenizer to convert text to suitable format
5. Incoming Requests are processed by the Scheduler
   - if sufficient memory, all processed in a batch
   - if insufficient memory, add to queue
6. During Model Generation
   - GPU calculates attention, retrieves history from KV Cache blocks
   - SamplingParams will define the temperature

---

---

##### vLLM Initialization

- tensor_parallel_size -> splits model across multiple GPUs

  - [4096, 4096], TP_SIZE=2
    - GPU0 : [2048, 4096]
    - GPU1: [2048, 4096]

- world_size=2 rank=0/1 -> 2 worker process, 1 for GPU 0 1 for GPU 1
- backend=nccl -> NVLink connection
- load model -> shard model to split between GPU 0 and GPU 1
- engine check GPU -> select FlashAttention2 (fastest possible kernel for attention math)
- vLLM runs dist.init_process_group under hood for NCCL comms
- when model run, every matmul is followed by all_reduce to compile answers
- gpu_memory_utilization: amount of memory reserved fo KV Cache
  - vLLM grabs memory upfront for efficient management
- trust_remote_code -> models saved in huggingface repo and we are technically downloading and running arbitrary python code
- SamplingParams() -> defines how model generates text after receiving prompt
  - temperature -> adjust probability distribution of next token before sampling (choosing) occurs
    - 0 -> no nonsense -> for practical tasks
    - 0.1 - 0.4 -> conservative -> technical documentation
    - 0.7 - 0.8 -> balanced -> natural conversation chatbot
    - 1. -> raw distribution -> creative writing

---

##### Measurement Metric

- Online (user is waiting)
  - Goal -> How fast before user see text
  - metric -> latency (time to first token)
- Offline (Batch processing) - massive file being processed
  - Goal -> How fast can work be done?
  - metric -> throughput (token / sec)

---

##### Before vLLM (8-Jan-26 20:48)

- served model via Hugging Face Accelerate
- raw PyTorch
- Problem:
  - Engine must reserve contiguous block of VRAM for context window (KV Cache)
  - 4k tokens context = reserve memory for 4k tokens -> if user said hello, other 4095 tokens wasted

---

##### Breakthrough (PagedAttention - UC Berkeley)

- Using concept of Virtual Memory (Paging)
- chops KV Cache into small, fixed sized blocks (16 tokens)
- blocks do not need to be contiguous in physical GPU memory
- vLLM maintains Lookup table (like OS) maps logical token 1 2 3 to physical block A, block B etc.
- no memory waste, can squeeze 20x-50x more concurrent users into GPU
- allow for continuous batching (new user request mid batching)
- Nvidia updated proprietary engine TensorRT-LLM with PagedAttention as well.

---

## Section 4 - Stress Test & Optimization

### 1) KV Cache vs Paged Attention

#### KV Cache

    - To predict 10th word, need attention from 1-9.
    - Naive = Re-compute math for words 1-9 every single time
    - KV Cache = Save attention for words 1-9, next time just load
    - Cache gets HUGE, 70B model with long context, cache can be larger than model weights itself

#### Problem

    - KV Cache grows dynamically
    - User can say "hi" (2 tokens)
    - User can say "write a book" (10k tokens)

#### Old way

    - PyTorch reserve massive contiguous chunk of VRAM for KV Cache (just in case)

#### vLLM PagedAttention

    - breaks KV Cache into non-contiguous block
    - "Rome" attention stored in VRAM address 100
    - "Empire" attention stored in VRAM address 5000
    - vLLM maintains Hash Table to track "Rome" and "Empire" & attention

#### Implication

    - No memory wasted

### 2) Batching Sweep Test

- GPU is like a bus (can hold 512 passengers)
  - if only transport 32 passengers, not maximizing space
  - goal is to maximize 512 (full batching)

##### Problem (How to actually determine "512"?)

- different GPUs have different limits
- Simply measure throughput = Generated Tokens / Time Taken to generate
- Specify test batches [512, 1024, 2048, 4096, 8192, 16384 ...]
- these test batches represent "number of requests (aka num people)"
  - so if we have 8192 requests \* 50 tokens = 409600 (50 < 320000 is fine)
- calculate the throughput -> once saturate (no more increase) = compute-bound
- if still can increase = memory bound (waiting for more data)

- **Tokens for different models differ in size**

  - View each token like a hiker
  - Token for small model = hiker carry small bagpack
  - Token for large model = hiker carrying camping rucksack
  - Token size = 2 _ num_layers _ num*kv_heads * dhead \_ Weight Precision

- **Grouped Query Attention (GQA)**
  - Multi-head attention, eg 32 heads = 32 Key and 32 Value matrix
  - Newer Models uses Grouped Query Heads
    - 32 Query heads for thinking
    - only 4 or 8 Key-Value heads for memory
    - memory footprint drops by 4x to 8x
