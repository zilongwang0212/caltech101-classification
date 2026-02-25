"""
Download and prepare Caltech-101 dataset
"""

import os
import sys
import zipfile
import argparse
from pathlib import Path
import urllib.request
from tqdm import tqdm


class DownloadProgressBar(tqdm):
    """Progress bar for download."""
    
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_url(url: str, output_path: str):
    """Download file from URL with progress bar."""
    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=url.split('/')[-1]) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)


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
        if path.exists():
            categories = [d for d in path.iterdir() if d.is_dir()]
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
                return True
    
    print("\n✗ Dataset structure not found!")
    print(f"  Please ensure the dataset is extracted properly in: {data_dir}")
    return False


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
    zip_path = data_dir / "caltech-101.zip"
    
    if verify_dataset(str(data_dir)):
        print("\n✓ Dataset already exists and is ready to use!")
        return
    
    # Check if zip file exists
    if zip_path.exists():
        print(f"\nFound existing zip file: {zip_path}")
        extract_dataset(str(zip_path), str(data_dir))
        verify_dataset(str(data_dir))
    else:
        download_caltech101_manual()
        
        # Check again if user has downloaded manually
        if zip_path.exists():
            extract_dataset(str(zip_path), str(data_dir))
            verify_dataset(str(data_dir))
        else:
            print("\n⚠ No dataset found. Please follow the manual download instructions above.")
            sys.exit(1)


if __name__ == "__main__":
    main()
