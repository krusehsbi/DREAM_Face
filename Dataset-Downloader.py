import argparse
import shutil
import sys
from random import random, randrange, sample
import cv2
import gdown
import tarfile
import os
from pathlib import Path
import subprocess
import requests
import pandas as pd
from mtcnn import MTCNN

"""
Dataset Downloader and Preprocessor

This script is designed to download and preprocess datasets for training and testing machine learning models, with a focus on facial recognition tasks.

The script has the following main functionalities:
1. **Downloading datasets**:
    - Downloads the UTK Face dataset, which contains images of faces with demographic information (age, gender, ethnicity).
    - Downloads images from OpenImages V7, which is a large collection of annotated images, from Google Drive.
    
2. **Filtering images**:
    - Filters out images that contain faces using the MTCNN (Multi-task Cascaded Convolutional Networks) face detector.
    - The filtering step is optional and can be skipped by passing specific command-line arguments.
    
3. **Handling downloaded files**:
    - After downloading, the script can move images from subdirectories into a main directory.
    - It checks for file conflicts and renames files as necessary before moving them.
    - The script also ensures that the required number of images is downloaded and provides options to fill the dataset to a specified size if needed.

Usage:
    The script accepts several command-line arguments to customize the behavior:
    
    - `-noutkface`: Skips downloading the UTK Face dataset.
    - `-noopenimages`: Skips downloading images from OpenImages V7.
    - `-noopenimagesfilter`: Skips the filtering of OpenImages V7 dataset images that contain faces.
    - `-nofillup`: Prevents downloading additional images if the dataset does not meet the specified target size.
"""

# Set to keep track of already processed images
filtered_files = set()

def delete_images_with_faces_mtcnn(directory_path):
    """
    Deletes images that contain faces from the specified directory using MTCNN face detection.
    
    Args:
    directory_path (str): The path to the directory containing images to be filtered.
    """
    detector = MTCNN()  # Initialize the MTCNN face detector

    total_files = os.listdir(directory_path)  # List all files in the directory
    num_total_files = len(total_files)
    processed_files = 0
    
    for filename in total_files:
        processed_files += 1
        # Process only image files and skip those already filtered
        if not filename in filtered_files and filename.endswith((".jpg", ".jpeg", ".png")):
            filtered_files.add(filename)

            file_path = os.path.join(directory_path, filename)
            image = cv2.imread(file_path)

            # Detect faces in the image
            faces = detector.detect_faces(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            if faces:
                os.remove(file_path)  # Delete the image if faces are detected
                print(f"{processed_files / num_total_files * 100:.3f}% Deleted {filename} as it contains a face.")
            else:
                print(f"{processed_files / num_total_files * 100:.3f}% No face detected in {filename}. Keeping file.")

def move_files_to_main(directory):
    """
    Moves all files from subdirectories to the main directory, handling filename conflicts.
    
    Args:
    directory (str): The path to the main directory where files will be moved.
    """
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
    Downloads a file from Google Drive and extracts it if it's a tar.gz file.
    
    Args:
    file_id (str): The Google Drive file ID to download.
    output_folder (str): The folder to extract the files into.
    """
    # Create output folder if it doesn't exist
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    # Temporary location for the downloaded file
    temp_file = output_folder / "temp.tar.gz"

    try:
        # Download the file from Google Drive
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, str(temp_file), quiet=False)

        # Extract if it's a tar.gz file
        if temp_file.exists() and temp_file.suffix == '.gz':
            with tarfile.open(str(temp_file), 'r:gz') as tar:
                # Extract all files to the output folder
                tar.extractall(path=str(output_folder))
            print(f"Successfully extracted to {output_folder}")

    except Exception as e:
        print(f"Error processing file: {e}")

    finally:
        # Clean up the temporary file
        if temp_file.exists():
            temp_file.unlink()

def download_open_images_v7(no_filter, no_fillup, total_number_images=24000):
    """
    Downloads images from OpenImages V7 and filters out images that contain faces.
    
    Args:
    no_filter (bool): If True, skips the filtering step.
    no_fillup (bool): If True, prevents trying to fill up the dataset with additional images if needed.
    total_number_images (int): The total number of images to download.
    """
    # Step 1: Download the metadata CSV file
    metadata_url = "https://storage.googleapis.com/openimages/2018_04/train/train-images-boxable-with-rotation.csv"
    metadata_file = Path(__file__).parent / "data/tmp/train-images-boxable-with-rotation.csv"

    (Path(__file__).parent / "data/tmp").mkdir(parents=True, exist_ok=True)

    # Download metadata if it doesn't exist
    if not os.path.exists(metadata_file):
        print("Downloading metadata CSV...")
        response = requests.get(metadata_url)
        with open(metadata_file, "wb") as f:
            f.write(response.content)
        print("Metadata downloaded.")

    # Define the data locations
    images_path = Path(__file__).parent / "data/nonface/openimages"
    images_path.mkdir(exist_ok=True, parents=True)
    filter_images_file = Path(__file__).parent / "openimages_filtered_images.txt"

    # Check if we have a store of known images that we can download first
    if os.path.isfile(filter_images_file):
        with open(filter_images_file, "r") as f:
            filtered_image_ids = f.readlines()
            print(f'Number of known images: {len(filtered_image_ids)}')
    else:
        filtered_image_ids = []

    # Prepare the list of new image IDs to download
    sampled_ids = []
    existing_images = set([x.replace('.jpg', '') for x in os.listdir(images_path)])
    for filtered_image_id in [x.replace('\n', '') for x in filtered_image_ids]:
        filtered_files.add(filtered_image_id + '.jpg')
        if not filtered_image_id in existing_images:
            sampled_ids.append(filtered_image_id)
    print(f"{len(existing_images)} images already downloaded, {len(sampled_ids)} new images found in openimages_filtered_images.txt")

    print(f"Loading metadata")
    metadata = pd.read_csv(metadata_file)

    while True:
        # Step 1: Check how many images are needed
        os.makedirs(images_path, exist_ok=True)
        existing_images = os.listdir(images_path)
        num_need_images = total_number_images - len(existing_images)

        # Step 2: Load the CSV and randomly sample enough images to get to the total number of images
        if num_need_images > 0:
            # Load as many entries as we need to fill up our known dataset
            sampled_data = metadata.sample(
                n=max(0, num_need_images-len(sampled_ids)),
                random_state=randrange(0, total_number_images))
            for sample_id in sampled_data['ImageID'].tolist():
                sampled_ids.append(sample_id)

            for i in range(max(0, len(sampled_ids) - num_need_images)):
                sampled_ids.pop()

            # Write the ids of the files wanted to a download file
            download_images_file = Path(__file__).parent / "data/tmp/openimages_downloads.txt"
            with open(download_images_file, "w") as f:
                for image_id in sampled_ids:
                    f.write(f"train/{image_id}\n")

            # Step 3: Download images
            script_path = Path(__file__).parent / "extern/OpenImagesV7/downloader.py"
            arguments = ['--download_folder', images_path,
                         '--num_processes', '5',
                         str(download_images_file)]

            subprocess.run([sys.executable, str(script_path)] + arguments)

        # Step 4: Filter out the images that contain faces
        if not no_filter:
            print("Filtering out the OpenImages7 images with faces")
            filter_openimages_v7()

        existing_images = os.listdir(images_path)
        num_need_images = total_number_images - len(existing_images)

        if no_fillup or num_need_images <= 0:
            break
        else:
            print(f"Downloading {num_need_images} more images to get to {total_number_images} random images...")
            sampled_ids = []

    print("Download completed.")

    # Save final list of image IDs to prevent future re-downloads
    final_image_ids = [Path(x).stem for x in os.listdir(images_path)]
    with open(filter_images_file, "w") as f:
        for image_id in final_image_ids:
            f.write(f"{image_id}\n")

def filter_openimages_v7():
    """
    Filters the OpenImagesV7 dataset by removing images with faces.
    """
    data_dir = Path(__file__).parent / "data" / "nonface" / "openimages"
    data_dir.mkdir(parents=True, exist_ok=True)
    delete_images_with_faces_mtcnn(data_dir)

def download_data():
    """
    Main function to download datasets including UTK-face and OpenImagesV7, 
    with options for filtering and managing the downloaded images.
    """
    parser = argparse.ArgumentParser(
        prog='Dataset-Downloader',
        description='Downloads utk-face dataset and some OpenImagesV7 images for training and testing'
    )
    # Argument parsing for various options
    parser.add_argument('-noutkface', action='store_true', help='Do not download the utk-face dataset')
    parser.add_argument('-noopenimages', action='store_true', help='Do not download the OpenImagesV7 dataset')
    parser.add_argument('-noopenimagesfilter', action='store_true', help='Do not filter out the OpenImagesV7 images with faces')
    parser.add_argument('-nofillup', action="store_true", help='Do not try to fill up the filtered out images with new ones until the desired amount is reached')
    parser.print_help()
    args = parser.parse_args()

    # Download UTK Face dataset if not skipped
    if not args.noutkface:
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

            # Move the downloaded files to the main directory
            move_files_to_main(utk_output_dir)

    # Download OpenImagesV7 dataset if not skipped
    if not args.noopenimages:
        print("Downloading OpenImagesV7 images")
        download_open_images_v7(args.noopenimagesfilter, args.nofillup, 24000)

if __name__ == '__main__':
    download_data()
