import numpy as np
import h5py
import os
import time
import gc
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

from tqdm import trange
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
for g in gpus:
    tf.config.experimental.set_memory_growth(g, True)
# tf.debugging.set_log_device_placement(True)
from tensorflow import keras
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, average_precision_score, f1_score
import matplotlib.pyplot as plt
from tensorflow_addons.metrics import F1Score
from tensorflow.keras.callbacks import CSVLogger
import seaborn as sns

# print(f"Current PID: {os.getpid()}")

# Clean current memory
keras.backend.clear_session()
gc.collect()
with h5py.File('ppg_test.h5', 'r') as f:
    X_test = f['signal'][:]   #(N_test, 800)
    y_test = f['rhythm'][:]    #(N_test, 2)
X_test = np.expand_dims(X_test, 2)    #(N_test, 800, 1)
# Predict on Test Set
print("Predicting on test set...")
model = keras.models.load_model("models/best_model.h5")
y_pred_prob = model.predict(X_test)
y_pred_class = np.argmax(y_pred_prob, axis=1)
y_true_class = np.argmax(y_test, axis=1)

print("Finished predictions.")
print(f"y_pred shape: {y_pred_prob.shape}, y_true shape: {y_true_class.shape}")
model.compile(
    optimizer=model.optimizer,  
    loss=model.loss,            
    metrics=[
        'accuracy',
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall'),
        tf.keras.metrics.AUC(name='auc'),
        F1Score(num_classes=2, average='macro', name='f1_score')
    ]
)
results = model.evaluate(X_test, y_test, return_dict=True, verbose=0)
print(results)
# Evaluate Model Performance
print("Evaluating model performance...") 
test_loss, test_acc, test_precision, test_recall, test_auc, test_f1 = model.evaluate(X_test, y_test, verbose=0) 
print(f"Test Loss: {test_loss:.4f}") 
print(f"Test Accuracy: {test_acc:.4f}")
print(f"Test Precision: {test_precision:.4f}")
print(f"Test Recall: {test_recall:.4f}")
print(f"Test AUC: {test_auc:.4f}")
print(f"Test F1 Score: {test_f1:.4f}")

# Confusion Matrix and Classification Report
print("Generating confusion matrix and classification report...")
print("Classification Report:", classification_report(y_true_class, y_pred_class))
#Confusion Matrix
cm = confusion_matrix(y_true_class, y_pred_class)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')

plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix On Test Set')
plt.savefig('test_confusion_matrix.png', dpi=300)
plt.show()

os.makedirs("logs", exist_ok=True)
with open("logs/classification_report.txt", "w", encoding="utf-8") as f:
    f.write(classification_report(y_true_class, y_pred_class))

print("Confusion matrix:", cm)

# Save predictions
print("Saving predictions...")
np.save('test_predictions_prob.npy', y_pred_prob)
np.save('test_predictions_class.npy', y_pred_class)
np.save('test_true_labels.npy', y_true_class)

print("Predictions saved.")
print("- test_predictions_prob.npy ")
print("- test_predictions_class.npy")
print("- test_true_labels.npy")




