from huggingface_hub import HfFileSystem
from dotenv import load_dotenv
from pathlib import Path
import os

# Load Env
env_path = Path(__file__).parent.parent / "DL_cologne_green" / ".env"
load_dotenv(env_path)

token = os.getenv("HF_TOKEN")
dataset_id = os.getenv("DATASET_ID", "Rahul-fix/cologne-green-data")

print(f"Inspecting Dataset: {dataset_id}")
if not token:
    print("WARNING: No HF_TOKEN found.")

fs = HfFileSystem(token=token)
base_path = f"datasets/{dataset_id}"

try:
    files = fs.glob(f"{base_path}/**/*")
    print(f"Found {len(files)} files:")
    for f in files:
        # Show relative path from dataset root
        rel_path = f.replace(base_path + "/", "")
        if not f.endswith(".tif") and not f.endswith(".jp2"):
            print(f" - {rel_path}")
except Exception as e:
    print(f"Error listing files: {e}")
