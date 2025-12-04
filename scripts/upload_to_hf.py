import os
import argparse
from pathlib import Path
from huggingface_hub import HfApi, create_repo
from dotenv import load_dotenv

# Load environment variables
# Try loading from root or DL_cologne_green
load_dotenv()
load_dotenv("DL_cologne_green/.env")

def upload_to_hf(dataset_id, token=None, private=True):
    """
    Uploads processed data to a Hugging Face Dataset.
    """
    if not token:
        token = os.getenv("HF_TOKEN")
    
    if not token:
        print("Error: HF_TOKEN not found in environment variables or arguments.")
        print("Please set HF_TOKEN in .env or pass it as an argument.")
        return

    api = HfApi(token=token)
    
    # Create repo if it doesn't exist
    try:
        print(f"Creating/Checking dataset repository: {dataset_id}...")
        create_repo(dataset_id, token=token, repo_type="dataset", private=private, exist_ok=True)
    except Exception as e:
        print(f"Error creating repo: {e}")
        return

    # Define paths
    base_dir = Path("data")
    folders_to_upload = ["processed", "stats", "boundaries"]
    
    print(f"Uploading data to {dataset_id}...")
    
    for folder in folders_to_upload:
        local_path = base_dir / folder
        if not local_path.exists():
            print(f"Warning: {local_path} does not exist. Skipping.")
            continue
            
        print(f"Uploading {folder}...")
        api.upload_folder(
            folder_path=str(local_path),
            repo_id=dataset_id,
            repo_type="dataset",
            path_in_repo=f"data/{folder}",
            token=token
        )
        
    print("Upload complete!")
    print(f"View your dataset at: https://huggingface.co/datasets/{dataset_id}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upload Cologne Green data to Hugging Face.")
    parser.add_argument("--dataset", type=str, help="Dataset ID (e.g., username/cologne-green-data)")
    parser.add_argument("--token", type=str, help="Hugging Face Write Token")
    parser.add_argument("--public", action="store_true", help="Make dataset public (default is private)")
    parser.add_argument("--auto", action="store_true", help="Automatically accept default dataset ID")
    
    args = parser.parse_args()
    
    # Interactive prompt if dataset not provided
    dataset_id = args.dataset
    if not dataset_id:
        # Try to guess username
        api = HfApi(token=args.token or os.getenv("HF_TOKEN"))
        try:
            user = api.whoami()
            username = user['name']
            default_id = f"{username}/cologne-green-data"
        except Exception as e:
            print(f"Could not determine username: {e}")
            username = "Rahul-fix"
            default_id = "Rahul-fix/cologne-green-data"
            
        if args.auto:
            dataset_id = default_id
            print(f"Auto-selected dataset ID: {dataset_id}")
        else:
            dataset_id = input(f"Enter Dataset ID [{default_id}]: ").strip() or default_id

    upload_to_hf(dataset_id, token=args.token, private=not args.public)
