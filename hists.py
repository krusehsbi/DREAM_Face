import matplotlib.pyplot as plt
from utils import load_data
import numpy as np

def plot_age_histogram(ages):
    plt.figure(figsize=(8, 6))
    plt.hist(ages, bins=range(0, 120, 10), color='blue', edgecolor='black')
    plt.title('Age Distribution')
    plt.xlabel('Age')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()

def plot_ethnicity_histogram(ethnicities):
    ethnicity_labels = ['White', 'Black', 'Asian', 'Indian', 'Others']
    plt.figure(figsize=(8, 6))
    unique, counts = np.unique(ethnicities, return_counts=True)
    plt.bar(ethnicity_labels, counts, color=['red', 'green', 'blue', 'purple', 'orange'], edgecolor='black')
    plt.title('Ethnicity Distribution')
    plt.xlabel('Ethnicity')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()

# Directory containing the images
image_directory = 'data/utk-face/UTKFace'  # Replace with the correct path

# Load the dataset
images, labels = load_data(image_directory)

# Separate ages and ethnicities from the labels
ages = labels[:, 0]  # First column is age
ethnicities = labels[:, 2]  # Third column is ethnicity (race)

# Plot histograms
plot_age_histogram(ages)
plot_ethnicity_histogram(ethnicities)
