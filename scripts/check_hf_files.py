from huggingface_hub import HfFileSystem
from dotenv import load_dotenv
import os

load_dotenv("DL_cologne_green/.env")
HF_TOKEN = os.getenv("HF_TOKEN")
DATASET_ID = os.getenv("DATASET_ID", "Rahul-fix/cologne-green-data")

def main():
    print(f"Checking files in {DATASET_ID}...")
    fs = HfFileSystem(token=HF_TOKEN)
    
    # Check for masks
    masks = fs.glob(f"datasets/{DATASET_ID}/data/processed/*_mask.tif")
    print(f"Found {len(masks)} mask files.")
    
    # Check for NDVI
    ndvis = fs.glob(f"datasets/{DATASET_ID}/data/processed/*_ndvi.tif")
    print(f"Found {len(ndvis)} NDVI files.")
    
    # Check for Raw
    raws = fs.glob(f"datasets/{DATASET_ID}/data/raw/*.jp2")
    print(f"Found {len(raws)} raw files.")

    # Check for Boundaries
    boundaries = fs.glob(f"datasets/{DATASET_ID}/data/boundaries/*.parquet")
    print(f"Found {len(boundaries)} boundary files:")
    for b in boundaries:
        print(f" - {b}")

    # Check for Stats
    stats = fs.glob(f"datasets/{DATASET_ID}/data/stats/*.parquet")
    print(f"Found {len(stats)} stats files:")
    for s in stats:
        print(f" - {s}")

if __name__ == "__main__":
    main()
