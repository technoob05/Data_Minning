
import os
import shutil
import zipfile

# Configuration
SOURCE_DIR = os.getcwd()
PACKAGE_NAME = "ReplicationPackage"
OUTPUT_ZIP = "zenodo-msr2026-replication-package.zip"

# Define structure mapping: Source Path -> Destination Path inside package
# If Source Path is a directory, it copies recursively.
STRUCTURE = {
    # Root files
    "README.md": "README.md",
    "LICENSE": "LICENSE",
    "CITATION.cff": "CITATION.cff",
    ".zenodo.json": ".zenodo.json",
    "requirements.txt": "requirements.txt",
    "run_pipeline.py": "run_pipeline.py",
    "Makefile": "Makefile",
    "DATASET.md": "DATASET.md",
    
    # Directories
    "src": "src",
    "scripts": "scripts",
    "notebooks": "notebooks",
}

def create_package():
    # 1. Create staging directory
    if os.path.exists(PACKAGE_NAME):
        shutil.rmtree(PACKAGE_NAME)
    os.makedirs(PACKAGE_NAME)
    
    print(f"Created staging directory: {PACKAGE_NAME}")

    # 2. Copy files/directories maintaining structure
    files_count = 0
    for src_name, dst_name in STRUCTURE.items():
        src_path = os.path.join(SOURCE_DIR, src_name)
        dst_path = os.path.join(PACKAGE_NAME, dst_name)
        
        if not os.path.exists(src_path):
            print(f"Warning: Source not found: {src_name}")
            continue
            
        if os.path.isdir(src_path):
            # Copy directory tree
            # Ignore __pycache__ and other patterns
            shutil.copytree(src_path, dst_path, ignore=shutil.ignore_patterns('__pycache__', '*.pyc', '*.png', '*.pdf', '*.parquet'))
            print(f"Copied directory: {src_name} -> {dst_name}")
            files_count += sum([len(files) for r, d, files in os.walk(src_path)])
        else:
            # Copy file
            shutil.copy2(src_path, dst_path)
            print(f"Copied file: {src_name}")
            files_count += 1

    # 3. Zip the staging directory
    print(f"Zipping package...")
    if os.path.exists(OUTPUT_ZIP):
        os.remove(OUTPUT_ZIP)
        
    shutil.make_archive(OUTPUT_ZIP.replace('.zip', ''), 'zip', root_dir=os.getcwd(), base_dir=PACKAGE_NAME)
    
    # Get stats
    zip_size = os.path.getsize(OUTPUT_ZIP) / (1024 * 1024)
    print(f"\nSUCCESS! Package created: {OUTPUT_ZIP}")
    print(f"Size: {zip_size:.2f} MB")
    print(f"Structure preserved in subdirectory: {PACKAGE_NAME}/")

    # Cleanup staging
    shutil.rmtree(PACKAGE_NAME)

if __name__ == "__main__":
    create_package()
