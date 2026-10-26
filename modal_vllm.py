import modal

vllm_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("vllm==0.32.0") 
    .env({"VLLM_USE_FLASHINFER_SAMPLER": "0"})
)

app = modal.App("qwen-collusion-sweep")

MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"

hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)

@app.function(
    image=vllm_image,
    gpu="A10G", 
    timeout=3600,
    volumes={"/root/.cache/huggingface": hf_cache},
    
    secrets=[modal.Secret.from_name("vllm-api-key")],
)
@modal.concurrent(max_inputs=100)  
@modal.web_server(port=8000, startup_timeout=600)
def serve():
    import os
    import subprocess
    
    subprocess.Popen(
        [
            "python", "-m", "vllm.entrypoints.openai.api_server",
            "--model", MODEL_NAME,
            "--host", "0.0.0.0", 
            "--port", "8000",
            "--max-model-len", "4096",
            "--enforce-eager", 
            "--api-key", os.environ["VLLM_API_KEY"],
        ]
    )
