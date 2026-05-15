from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="runwayml/stable-diffusion-v1-5",
    local_dir="./weights/sd-1.5",
    allow_patterns=["*.safetensors", "*.json", "*.txt"],
)
print("Download complete.")