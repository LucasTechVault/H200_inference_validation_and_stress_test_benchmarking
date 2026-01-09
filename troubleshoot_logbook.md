#### 8-Jan-2026

##### 11:01

- **Blocker:** Installing Nsys on remote server
- **Fix:** Remote Profiling (local Mac as Host, Remote as Target)
- **Takeaways:** Professionals rarely profile directly in terminal but local desktop to profile remote server

**Learning Points**

- Nsight uses Host-Target architecture

1. checks remote (target) if nsys CLI & backend libraries found (else SCP binaries from host)
1. CPU & GPU Profiling
   - CPU via OS Sampling
   - GPU via CUDA
1. Data flow
   - Live Mode via SSH tunnel
   - Offline Mode, save data in file & downloadable later for analysis

---

#### 9-Jan-2026

##### 10:56

- **Blocker**: Model Qwen/Qwen2.5-1.5B-Instruct trained up till only 32768 tokens specified in config.json. But, performing stress test requires pushing more than specified tokens
- **Fix:** add VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 to environment
- **Takeaways:**
  - Goal is to stress test GPU, not accuracy of model's intelligence.
  - Pushing past trained tokens will cause model to generate rubbish but that does not matter

##### 11:18

- **Blocker:** When running experiment in loops, vLLM unable to startup in next iteration due to previous run still holding to VRAM
- **Fix:** Set delay of 5 to 10 seconds to allow vLLM to clear memory

##### Problem: This also failed

##### 12:42

- Found fix for the above issue.
- Instead of script running in single process, every iteration spawns new linux process
- **Fix:** Kills process after iteration and begin new process using multiprocessing.Process
- **Takeaways:**
  - vLLM uses Ray backend, designed to be long-running server
  - when loop starts in same process, old artifacts conflict with new ones.

##### 13:09

- **Blocker:** Trying to explore batch sweeping (num of users), say [512, 32768, 65536] num of requests. Each user provides small token, say 50. vLLM defaults to processing maximum of 16k tokens per step.
- **Fix:**: Change limit by specifying
- **Takeaways**:
  - even if increase limit, might soon be compute-bound = diminishing return
