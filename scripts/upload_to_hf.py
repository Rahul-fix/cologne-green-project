import os
import argparse
from pathlib import Path
from huggingface_hub import HfApi, create_repo
from dotenv import load_dotenv

# Load environment variables
# Try loading from root or DL_cologne_green
load_dotenv()
load_dotenv("DL_cologne_green/.env")

def upload_to_hf(dataset_id, token=None, private=True, auto_confirm=False, force=False, folders_to_upload=None):
    if not token:
        token = os.getenv("HF_TOKEN")
    
    if not token:
        print("❌ HF_TOKEN not found. Please set it in .env or pass as argument.")
        return

    api = HfApi(token=token)
    
    # Create dataset if not exists
    try:
        api.create_repo(repo_id=dataset_id, repo_type="dataset", private=private, exist_ok=True)
        print(f"✅ Dataset repo {dataset_id} ready.")
    except Exception as e:
        print(f"❌ Error creating repo: {e}")
        return

    # Define paths
    base_dir = Path("data")
    if folders_to_upload is None:
        folders_to_upload = ["raw", "processed", "boundaries", "stats"]

    print(f"Uploading data to {dataset_id}...")
    
    for folder_name in folders_to_upload:
        local_path = base_dir / folder_name
        if not local_path.exists():
            print(f"Warning: {local_path} does not exist. Skipping.")
            continue
            
        print(f"Uploading {folder_name}...")
        # Check if folder already exists in repo to avoid re-uploading/hashing large data
        try:
            repo_files = api.list_repo_files(repo_id=dataset_id, repo_type="dataset")
            folder_exists = any(f.startswith(f"data/{folder}/") for f in repo_files)
        except Exception:
            folder_exists = False

        if folder_exists:
            print(f"Found existing data in 'data/{folder}' on Hugging Face.")
            if folder == "raw": # Special handling for raw data which is huge
                if not args.force:
                    response = input(f"Skipping 'data/{folder}' to save time? (y/n) [y]: ").strip().lower()
                    if response in ["", "y", "yes"]:
                        print(f"Skipping {folder}...")
                        continue
        
        print(f"Uploading {folder_name}...")
        try:
            # Use upload_folder with multi_commits=True for large datasets
            # This is the recommended way to handle large uploads in newer hf_hub versions
            api.upload_folder(
                folder_path=str(local_path),
                repo_id=dataset_id,
                repo_type="dataset",
                path_in_repo=f"data/{folder_name}",
                multi_commits=True,
                run_as_future=False
            )
            print(f"✅ Uploaded {folder_name}!")
        except Exception as e:
            print(f"Error uploading {folder_name}: {e}")
            # Fallback without multi_commits if it fails (e.g. older version)
            print("Retrying without multi_commits...")
            api.upload_folder(
                folder_path=str(local_path),
                repo_id=dataset_id,
                repo_type="dataset",
                path_in_repo=f"data/{folder_name}",
                token=token
            )
        
    print("Upload complete!")
    print(f"View your dataset at: https://huggingface.co/datasets/{dataset_id}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upload Cologne Green data to Hugging Face.")
    parser.add_argument("--dataset", type=str, help="Dataset ID (e.g., username/cologne-green-data)")
    parser.add_argument("--token", type=str, help="Hugging Face Write Token")
    parser.add_argument("--public", action="store_true", help="Make dataset public")
    parser.add_argument("--auto", action="store_true", help="Auto-confirm uploads")
    parser.add_argument("--force", action="store_true", help="Force upload even if files exist")
    parser.add_argument("--folders", nargs="+", default=["raw", "processed", "boundaries", "stats", "metadata"], help="Folders to upload") # (skip prompt)
    
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

    upload_to_hf(
        dataset_id=dataset_id, 
        token=args.token, 
        private=not args.public, 
        auto_confirm=args.auto,
        force=args.force,
        folders_to_upload=args.folders
    )
