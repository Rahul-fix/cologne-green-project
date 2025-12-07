import argparse
import os
from huggingface_hub import snapshot_download
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
load_dotenv("DL_cologne_green/.env")

def download_data(dataset_id, folders, token=None):
    """
    Downloads specific folders from a Hugging Face dataset to the local directory.
    """
    if not token:
        token = os.getenv("HF_TOKEN")
        
    print(f"Downloading {folders} from {dataset_id}...")
    
    # Construct patterns to match the requested folders
    # Assuming structure is data/{folder}/...
    allow_patterns = [f"data/{folder}/*" for folder in folders]
    
    try:
        local_path = snapshot_download(
            repo_id=dataset_id,
            repo_type="dataset",
            local_dir=".", # Download to current directory, preserving structure
            allow_patterns=allow_patterns,
            token=token,
            resume_download=True
        )
        print(f"✅ Download complete! Files are in: {local_path}")
    except Exception as e:
        print(f"❌ Download failed: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download Cologne Green data from Hugging Face.")
    parser.add_argument("--dataset", type=str, default="Rahul-fix/cologne-green-data", help="Dataset ID")
    parser.add_argument("--folders", nargs="+", default=["processed", "boundaries"], help="Folders to download (e.g. processed boundaries raw)")
    
    args = parser.parse_args()
    
    download_data(args.dataset, args.folders)
