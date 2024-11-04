import shutil

import gdown
import tarfile
import os
from pathlib import Path
import subprocess

def move_files_to_main(directory):
    # Walk through all subdirectories of the specified directory
    for root, dirs, files in os.walk(directory):
        # Skip the main directory itself
        if root == directory:
            continue

        # Move each file in the current subdirectory to the main directory
        for file in files:
            source_path = os.path.join(root, file)
            destination_path = os.path.join(directory, file)

            # Handle filename conflicts by renaming if necessary
            if os.path.exists(destination_path):
                base, extension = os.path.splitext(file)
                count = 1
                while os.path.exists(destination_path):
                    destination_path = os.path.join(directory, f"{base}_{count}{extension}")
                    count += 1

            # Move the file to the main directory
            shutil.move(str(source_path), str(destination_path))

    print(f"All files have been moved to {directory}")

def download_and_extract(file_id, output_folder):
    """
    Download a file from Google Drive and extract it if it's a tar.gz file
    """
    # Create output folder if it doesn't exist
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    # Temporary location for the downloaded file
    temp_file = output_folder / "temp.tar.gz"

    try:
        # Download the file
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, str(temp_file), quiet=False)

        # Extract if it's a tar.gz file
        if temp_file.exists() and temp_file.suffix == '.gz':
            with tarfile.open(str(temp_file), 'r:gz') as tar:
                # Extract all files
                tar.extractall(path=str(output_folder))
            print(f"Successfully extracted to {output_folder}")

    except Exception as e:
        print(f"Error processing file: {e}")

    finally:
        # Clean up the temporary file
        if temp_file.exists():
            temp_file.unlink()

def download_imagenet_images():
    data_dir = Path(__file__).parent / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    subprocess.run([
        "python3",
        "./extern/ImageNet-Datasets-Downloader/downloader.py",
        "-data_root", str(data_dir / "nonface"),
        "-number_of_classes", "20",
        "-images_per_class", "1000",
    ])

def main():
    #Download UTK Face
    # Set up the relative output path (relative to this script)
    script_dir = Path(__file__).parent
    utk_output_dir = script_dir / "data" / "utk-face"

    if not os.path.exists(utk_output_dir):
        print("Downloading UTK Face images")

        # List of Google Drive file IDs for UTK Face
        file_ids = [
            "1mb5Z24TsnKI3ygNIlX6ZFiwUj0_PmpAW",  # part1
            "19vdaXVRtkP-nyxz1MYwXiFsh_m_OL72b", # part2
            "1oj9ZWsLV2-k2idoW_nRSrLQLUP3hus3b" # part3
        ]

        # Download and extract each file
        for file_id in file_ids:
            download_and_extract(file_id, utk_output_dir)
            print(f"Processed file ID: {file_id}")

        move_files_to_main(utk_output_dir)

    #Download Imagenet
    print("Downloading Imagenet images")
    download_imagenet_images()

if __name__ == "__main__":
    main()