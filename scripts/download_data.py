"""
Download and prepare Caltech-101 dataset
"""

import os
import sys
import shutil
import argparse
from pathlib import Path
import kagglehub


def download_caltech101_kagglehub(data_dir: str):
    """Download Caltech-101 dataset using kagglehub."""
    print("\nDownloading Caltech-101 dataset using kagglehub...")
    print("This may take a few minutes depending on your internet connection...")
    
    try:
        # Download latest version
        download_path = kagglehub.dataset_download("imbikramsaha/caltech-101")
        print(f"\n✓ Dataset downloaded to: {download_path}")
        
        # Copy to target directory if different
        target_dir = Path(data_dir)
        source_dir = Path(download_path)
        
        # Find 101_ObjectCategories directory or caltech-101 directory with categories
        object_categories = None
        
        # Check for 101_ObjectCategories
        for item in source_dir.rglob("101_ObjectCategories"):
            if item.is_dir():
                object_categories = item
                break
        
        # If not found, check for caltech-101 directory with image categories
        if not object_categories:
            caltech_dir = source_dir / "caltech-101"
            if caltech_dir.exists() and caltech_dir.is_dir():
                # Check if it contains category directories
                subdirs = [d for d in caltech_dir.iterdir() if d.is_dir()]
                if len(subdirs) > 50:  # Caltech-101 has 101 categories
                    object_categories = caltech_dir
        
        if object_categories:
            target_path = target_dir / "101_ObjectCategories"
            if not target_path.exists():
                print(f"\nCopying dataset to {target_path}...")
                shutil.copytree(object_categories, target_path, dirs_exist_ok=True)
                print("✓ Dataset copied successfully!")
            return True
        else:
            print("⚠ Could not find dataset categories in downloaded data")
            print(f"  Downloaded path: {download_path}")
            print(f"  Contents: {list(source_dir.iterdir())}")
            return False
            
    except Exception as e:
        print(f"\n✗ Error downloading dataset: {e}")
        import traceback
        traceback.print_exc()
        return False


def download_caltech101_manual():
    """Instructions for manual download from Kaggle."""
    print("\n" + "="*80)
    print("MANUAL DOWNLOAD REQUIRED")
    print("="*80)
    print("\nDue to Kaggle API authentication requirements, please follow these steps:")
    print("\n1. Go to: https://www.kaggle.com/datasets/imbikramsaha/caltech-101")
    print("2. Click 'Download' button (you may need to sign in)")
    print("3. Save the downloaded 'caltech-101.zip' file to the 'data/' directory")
    print("4. Run this script again to extract the dataset")
    print("\nAlternatively, set up Kaggle API credentials:")
    print("  - Go to https://www.kaggle.com/account")
    print("  - Create an API token (downloads kaggle.json)")
    print("  - Place kaggle.json in ~/.kaggle/ directory")
    print("  - Install kaggle: pip install kaggle")
    print("  - Run: kaggle datasets download -d imbikramsaha/caltech-101 -p data/")
    print("\n" + "="*80 + "\n")


def extract_dataset(zip_path: str, extract_dir: str):
    """Extract zip file."""
    print(f"\nExtracting dataset to {extract_dir}...")
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        # Get list of files
        file_list = zip_ref.namelist()
        
        # Extract with progress bar
        for file in tqdm(file_list, desc="Extracting"):
            zip_ref.extract(file, extract_dir)
    
    print(f"✓ Dataset extracted successfully!")


def verify_dataset(data_dir: str):
    """Verify dataset structure."""
    data_path = Path(data_dir)
    
    # Look for the main dataset directory
    possible_paths = [
        data_path / "caltech-101" / "101_ObjectCategories",
        data_path / "101_ObjectCategories",
        data_path
    ]
    
    for path in possible_paths:
        if path.exists() and path.is_dir():
            categories = [d for d in path.iterdir() if d.is_dir() and not d.name.startswith('.')]
            num_categories = len(categories)
            
            if num_categories > 0:
                print(f"\n✓ Dataset verified!")
                print(f"  Location: {path}")
                print(f"  Categories found: {num_categories}")
                
                # Count images
                total_images = 0
                for category in categories:
                    images = list(category.glob("*.jpg")) + list(category.glob("*.png"))
                    total_images += len(images)
                
                print(f"  Total images: {total_images}")
                return str(path)
    
    print("\n✗ Dataset structure not found!")
    print(f"  Please ensure the dataset is extracted properly in: {data_dir}")
    return None


def main():
    parser = argparse.ArgumentParser(description='Download Caltech-101 dataset')
    parser.add_argument(
        '--data_dir',
        type=str,
        default='data',
        help='Directory to save dataset'
    )
    args = parser.parse_args()
    
    # Create data directory
    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*80)
    print("Caltech-101 Dataset Downloader")
    print("="*80)
    
    # Check if dataset already exists
    existing_path = verify_dataset(str(data_dir))
    if existing_path:
        print("\n✓ Dataset already exists and is ready to use!")
        print(f"  Use this path: {existing_path}")
        return
    
    # Download using kagglehub
    print("\nDownloading dataset using kagglehub...")
    if download_caltech101_kagglehub(str(data_dir)):
        verify_dataset(str(data_dir))
        print("\n✓ Dataset is ready to use!")
    else:
        print("\n⚠ Download failed. Showing manual download instructions...")
        download_caltech101_manual()
        sys.exit(1)


if __name__ == "__main__":
    main()
