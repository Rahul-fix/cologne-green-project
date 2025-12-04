import rasterio
import numpy as np
from pathlib import Path

# Paths
DATA_DIR = Path("data")
PROCESSED_DIR = DATA_DIR / "processed"

def main():
    # Find a mask file
    mask_files = list(PROCESSED_DIR.glob("*_mask.tif"))
    if not mask_files:
        print("No mask files found.")
        return

    sample_file = mask_files[0]
    print(f"Analyzing {sample_file.name}...")
    
    with rasterio.open(sample_file) as src:
        data = src.read(1)
        unique_values = np.unique(data)
        print(f"Unique values found: {unique_values}")
        
        # Check if it matches expected COSIA range (0-18)
        if np.max(unique_values) > 2:
            print("Detected multi-class Land Cover data (likely COSIA).")
        else:
            print("Detected binary/ternary mask (likely simplified).")

if __name__ == "__main__":
    main()
