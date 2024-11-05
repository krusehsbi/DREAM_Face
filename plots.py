import pandas as pd
import matplotlib.pyplot as plt

# Load the data from the CSV file
history = pd.read_csv('saved_models/training_history.csv')

# Plot for Age Mean Absolute Error
plt.plot(history['age_output_mae'], label='Train MAE')
plt.plot(history['val_age_output_mae'], label='Validation MAE')
plt.title('Age Mean Absolute Error')
plt.ylabel('Error')
plt.xlabel('Epoch')
plt.legend(loc='upper left')
plt.show()

# Plot for Age Loss
plt.plot(history['age_output_loss'], label='Train Loss')
plt.plot(history['val_age_output_loss'], label='Validation Loss')
plt.title('Age Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(loc='upper left')
plt.show()

# Plot for Face Accuracy
plt.plot(history['face_output_accuracy'], label='Train Accuracy')
plt.plot(history['val_face_output_accuracy'], label='Validation Accuracy')
plt.title('Face Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(loc='upper left')
plt.show()

# Plot for Face Loss
plt.plot(history['face_output_loss'], label='Train Loss')
plt.plot(history['val_face_output_loss'], label='Validation Loss')
plt.title('Face Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(loc='upper left')
plt.show()

# Plot for Gender Accuracy
plt.plot(history['gender_output_accuracy'], label='Train Accuracy')
plt.plot(history['val_gender_output_accuracy'], label='Validation Accuracy')
plt.title('Gender Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(loc='upper left')
plt.show()

# Plot for Gender Loss
plt.plot(history['gender_output_loss'], label='Train Loss')
plt.plot(history['val_gender_output_loss'], label='Validation Loss')
plt.title('Gender Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(loc='upper left')
plt.show()
