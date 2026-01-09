import pynvml
import time
import csv
import os
from datetime import datetime

# --- CONFIG ---
LOG_FILE = "gpu_metrics.csv"
GPU_INDICES = [0, 1]  # The GPUs to track (set as required)
SAMPLE_INTERVAL = 0.1 # 100ms resolution

def get_gpu_metrics(handle):
    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
    temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
    power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0 # Convert mW to W
    
    return {
        "memory_used_mb": mem_info.used / 1024**2,
        "gpu_util_pct": util.gpu,
        "temp_c": temp,
        "power_w": power
    }

def monitor():
    print(f"--- Section 5: GPU Resource Monitor ---")
    print(f"Logging to {LOG_FILE} every {SAMPLE_INTERVAL}s...")
    
    pynvml.nvmlInit()
    handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(pynvml.nvmlDeviceGetCount())]
    
    # (Simplified: log all visible GPUs to be safe)
    
    with open(LOG_FILE, mode='w', newline='') as f:
        writer = csv.writer(f)
        # Header
        headers = ["timestamp"]
        for i in range(len(handles)):
            headers.extend([f"gpu{i}_util", f"gpu{i}_mem", f"gpu{i}_pwr"])
        writer.writerow(headers)
        
        try:
            while True:
                row = [datetime.now().strftime("%H:%M:%S.%f")]
                for handle in handles:
                    metrics = get_gpu_metrics(handle)
                    row.extend([
                        metrics["gpu_util_pct"],
                        f"{metrics['memory_used_mb']:.0f}",
                        f"{metrics['power_w']:.1f}"
                    ])
                
                writer.writerow(row)
                f.flush() # Ensure data is written immediately
                
                # Pretty print to console (overwrite line)
                print(f"\rCaptured: {row[1:]} ...", end="")
                time.sleep(SAMPLE_INTERVAL)
                
        except KeyboardInterrupt:
            print("\nStopping Monitor.")
            pynvml.nvmlShutdown()

if __name__ == "__main__":
    monitor()