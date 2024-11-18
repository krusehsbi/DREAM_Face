import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from utils import load_data
from keras import saving, applications
from sklearn.model_selection import train_test_split
import FaceIdentifier

# Define directories
face_directory = ['data/utk-face', 'data/utk-face/UTKFace']
non_face_directory = ['data/nonface']

# Load images and labels
images, labels = load_data(face_directory, non_face_directory, preprocess_fnc=None)

# Separate labels for age prediction
y_test_age = labels[:, 1]

# Load the trained model
model = saving.load_model('saved_models/Face.keras')

# Predict age on the test set
predictions = model.predict(applications.efficientnet.preprocess_input(images))
predicted_ages = predictions['age_output'].flatten()  # Assuming age predictions are in index 1

# Exclude non-face placeholders
valid_age_mask = y_test_age != 200
y_test_age_valid = y_test_age[valid_age_mask]
predicted_ages_valid = predicted_ages[valid_age_mask]

# Group ages into 10-year bins and calculate MAE for each bin
age_bins = np.arange(0, 101, 10)
age_groups = np.digitize(y_test_age_valid, age_bins) - 1  # Bin indices (0-indexed)

mae_per_group = []
for i in range(len(age_bins) - 1):  # Iterate over each bin
    group_mask = age_groups == i  # Get indices of samples in the current bin
    if np.any(group_mask):  # If there are samples in the bin
        group_mae = mean_absolute_error(y_test_age_valid[group_mask], predicted_ages_valid[group_mask])
        mae_per_group.append(group_mae)
    else:  # No samples in the bin
        mae_per_group.append(0)

# Plot the MAE per age group
plt.figure(figsize=(12, 6))
plt.bar([f'{age_bins[i]}-{age_bins[i + 1]}' for i in range(len(age_bins) - 1)], mae_per_group, color='skyblue')
plt.xlabel('Age Group (Years)')
plt.ylabel('Mean Absolute Error (MAE)')
plt.title('Age Prediction MAE per 10-Year Age Group')
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('age_group_mae_barplot.png')
print("Age group MAE bar plot saved as 'age_group_mae_barplot.png'")

# Group ages into 10-year bins and calculate mean error (without absolute value) for each bin
mean_error_per_group = []
for i in range(len(age_bins) - 1):  # Iterate over each bin
    group_mask = age_groups == i  # Get indices of samples in the current bin
    if np.any(group_mask):  # If there are samples in the bin
        group_mean_error = np.mean(y_test_age_valid[group_mask] - predicted_ages_valid[group_mask])
        mean_error_per_group.append(group_mean_error)
    else:  # No samples in the bin
        mean_error_per_group.append(0)

# Plot the Mean Error per age group
plt.figure(figsize=(12, 6))
plt.bar([f'{age_bins[i]}-{age_bins[i + 1]}' for i in range(len(age_bins) - 1)], mean_error_per_group, color='salmon')
plt.xlabel('Age Group (Years)')
plt.ylabel('Mean Error')
plt.title('Age Prediction Mean Error per 10-Year Age Group')
plt.xticks(rotation=45)
plt.axhline(0, color='black', linewidth=0.8, linestyle='--')  # Add a horizontal line at y=0
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('age_group_mean_error_barplot.png')
print("Age group mean error bar plot saved as 'age_group_mean_error_barplot.png'")

