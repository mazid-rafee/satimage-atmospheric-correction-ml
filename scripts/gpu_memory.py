import argparse
import torch
import time
import os
import subprocess
import signal
import sys

GPU_ID = 5  # You can also make this configurable if needed
PID_FILE = f"gpu_alloc_gpu{GPU_ID}.pid"

def allocate_memory(bytes_to_allocate):
    torch.cuda.set_device(GPU_ID)
    print(f"[ALLOC] Using GPU {GPU_ID}")
    print(f"[ALLOC] Allocating {bytes_to_allocate / 1e9:.2f} GB on GPU...")
    tensor = torch.empty(int(bytes_to_allocate // 4), dtype=torch.float32, device=f'cuda:{GPU_ID}')
    print(f"[ALLOC] Memory allocated and held on GPU {GPU_ID}.")
    try:
        while True:
            time.sleep(60)
    except KeyboardInterrupt:
        print("[ALLOC] Received termination signal. Releasing memory.")
        del tensor
        torch.cuda.empty_cache()

def start_allocator_process(alloc_size_gb):
    cmd = [sys.executable, __file__, "--_alloc_background", str(alloc_size_gb)]
    proc = subprocess.Popen(cmd)
    with open(PID_FILE, "w") as f:
        f.write(str(proc.pid))
    print(f"[ALLOC] Started background memory holder on GPU {GPU_ID} with PID {proc.pid}")

def stop_allocator_process():
    if not os.path.exists(PID_FILE):
        print(f"[FREE] No memory holder process found for GPU {GPU_ID}.")
        return
    with open(PID_FILE, "r") as f:
        pid = int(f.read())
    try:
        os.kill(pid, signal.SIGTERM)
        print(f"[FREE] Terminated memory holder process (PID {pid}) for GPU {GPU_ID}")
    except ProcessLookupError:
        print(f"[FREE] Process {pid} not running.")
    os.remove(PID_FILE)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--alloc", nargs="?", const=37, type=float,
                        help="Allocate and hold GPU memory. Optionally specify GB (default: 15)")
    parser.add_argument("--free", action="store_true", help="Free previously allocated memory")
    parser.add_argument("--_alloc_background", type=float, help=argparse.SUPPRESS)

    args = parser.parse_args()

    if args.alloc is not None:
        start_allocator_process(alloc_size_gb=args.alloc)
    elif args.free:
        stop_allocator_process()
    elif args._alloc_background is not None:
        torch.cuda.set_device(GPU_ID)
        alloc_bytes = args._alloc_background * (1024 ** 3)
        allocate_memory(alloc_bytes)
    else:
        print("Usage:")
        print("  python gpu_memory.py --alloc [GB]   # Allocate and hold GPU memory on GPU 5 (default 15GB)")
        print("  python gpu_memory.py --free         # Free the allocated memory on GPU 5")

if __name__ == "__main__":
    main()
