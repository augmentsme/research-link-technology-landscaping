#!/bin/bash

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:2
#SBATCH --time=10:00:00
#SBATCH --partition=milan-gpu
#SBATCH --mail-user lcheng@swin.edu.au
#SBATCH --mail-type=ALL



# 1. Set proxy variables for this shell session.
#    This part runs for EVERY login to a gina host.
export HTTP_PROXY=http://localhost:8887
export HTTPS_PROXY=http://localhost:8887
echo "Proxy variables set for gina host (http://localhost:8887)"

# 2. Check if the remote proxy service is running on tooarrana1.
#    This command is executed FROM this gina host TO tooarrana1.
if ! ssh tooarrana1 'pgrep -f "tinyproxy -c .*/tinyproxy.conf"' > /dev/null; then
    echo "Tinyproxy is not running on tooarrana1. Attempting to start it..."
    ssh tooarrana1 'tinyproxy -c ~/.tinyproxy/tinyproxy.conf'
    # Give the remote service a moment to initialize before we connect.
    sleep 1
fi

# 3. Check if the local SSH tunnel process is already running on this gina host.
#    Start the tunnel only if it's not already established.
if ! pgrep -f "ssh -N -f -L 8887:127.0.0.1:8887 tooarrana1" > /dev/null; then
    echo "Starting port forward tunnel from this gina host to tooarrana1..."
    ssh -N -f -L 8887:127.0.0.1:8887 tooarrana1
fi
ml load cuda/12.8.0



source /fred/oz318/luhanc/vllmenv/bin/activate
vllm serve Qwen/Qwen3-4B-Instruct-2507 -dp 2 --gpu-memory-utilization 0.95 --max-num-batched-tokens 65536 --max-num-seqs 512 --max-seq-len 4096 &
deactivate

# Wait for the server to initialize. This is important to ensure the server
# is ready to accept requests before the make tasks start.
echo "Waiting for vLLM server to initialize..."
until curl -sf localhost:8000/v1/models > /dev/null; do
    sleep 10
done

# Run the data processing pipeline
echo "Starting data processing pipeline..."
uv run inspect eval --display plain extract.py@extract
make generate-keywords 
make categorise
