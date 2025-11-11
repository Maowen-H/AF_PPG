
import numpy as np
import h5py
import os
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

#print(f"Current PID: {os.getpid()}")

# Clean current memory
keras.backend.clear_session()
gc.collect()

try:
    print("Loading Dataset...")
    print('Loading training data from ppg_train.h5')
    with h5py.File('ppg_train.h5', 'r') as f:
        X_train = f['signal'][:]  #(N_train, 800)
        y_train = f['rhythm'][:]   #(N_train, 2)
    # X_train = np.load("train.npy")  #(N_train, 800)
    # y_train = X_train['rhythm']  #(518782,2)
    print('Loading validation data from ppg_validate.h5')
    with h5py.File('ppg_validate.h5', 'r') as f:
        X_val = f['signal'][:]    #(N_val, 800)
        y_val = f['rhythm'][:]     #(N_val, 2)
    # X_val = np.load("validate.npy")  #(N_val, 800)
    # y_val = X_val['rhythm']
    print('Loading test data from ppg_test.h5')
    with h5py.File('ppg_test.h5', 'r') as f:
        X_test = f['signal'][:]   #(N_test, 800)
        y_test = f['rhythm'][:]    #(N_test, 2)
    # X_test = np.load("test.npy") # (N_test, 800)
    # y_test = X_test['rhythm']

    print(f"Train set shape:{X_train.shape}, {y_train.shape}")
    print(f"Validation set shape:{X_val.shape}, {y_val.shape}")
    print(f"Test set shape:{X_test.shape}, {y_test.shape}")


    def check_X(name, X):
        print(name, X.shape, X.dtype,
                "min:", np.nanmin(X), "max:", np.nanmax(X),
                "NaN:", np.isnan(X).any(), "Inf:", np.isinf(X).any())

    check_X("X_train", X_train); check_X("X_val", X_val)
    check_X("X_test", X_test)
    X = X_train
    if X.ndim == 2:
        X = X[..., None]
    bad_mask = np.isnan(X).any(axis=(1,2))
    print("Count of badmask samples in X_train:", np.sum(bad_mask))

    X_train = X_train[~bad_mask]
    y_train = y_train[~bad_mask]

    # Data reshape
    print("Reshaping data...")
    X_train = np.expand_dims(X_train, 2)  #(N_train, 800, 1)
    X_val = np.expand_dims(X_val, 2)      #(N_val, 800, 1)
    X_test = np.expand_dims(X_test, 2)    #(N_test, 800, 1)

    print("After reshaping:")
    print(f"Train set shape:{X_train.shape}")
    check_X("X_train", X_train)
    print(f"Validation set shape:{X_val.shape}")
    print(f"Test set shape:{X_test.shape}")

    # Load Model and Predict Compile
    print("Loading Model...")

    base = keras.models.load_model("models/deepbeat.h5")
    prev = base.get_layer('dense_18').output
    af_soft = keras.layers.Dense(2, activation='softmax', name='af_head')(prev)
    # rhythm_logits= base.get_layer('rhythm_output').output
    # r_soft= keras.layers.Activation('softmax', name='rhythm_output_softmax')(rhythm_logits)
    model = keras.Model(inputs=base.input, outputs=af_soft)
    layer_r = model.get_layer("af_head")
    print("af_head activation:", layer_r.activation.__name__)
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)

    model.compile(optimizer=optimizer,
                    loss = 'categorical_crossentropy',
                    metrics = ['accuracy',
                        # keras.metrics.Precision(name='precision'),
                        # keras.metrics.Recall(name='recall'),
                        # keras.metrics.AUC(name='auc'),
                        F1Score(num_classes=2, average='macro', name='f1_score')
                    ])

    # Training the model
    print("Training the model...")

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath="best_model.h5",
            monitor="val_f1_score",
            save_best_only=True,
            mode = 'max',
            verbose = 1
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_f1_score",
            patience=7,
            mode = 'max',
            restore_best_weights=True,
            verbose = 1
        ),
        keras.callbacks.ReduceLROnPlateau(
        monitor="val_f1_score",
        factor=0.5,
        patience=5,
        min_lr=1e-5,
        verbose=1
    )
    ]

    csv_cb = CSVLogger("logs/train_log_test.csv", append=False)  

    history = model.fit(
        X_train, y_train,
        batch_size = 256,
        epochs = 40,
        validation_data = (X_val, y_val),
        callbacks = callbacks + [csv_cb],
        verbose = 1
    )

    # Visualize training history
    print("Visualizing training history...")
    plt.figure(figsize=(15, 8))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['f1_score'], label='Train F1 Score')
    plt.plot(history.history['val_f1_score'], label='Validation F1 Score')
    plt.title('F1 Score Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.savefig('training_history.png', dpi=300)
    plt.show()
except keyboardInterrupt:
    print("Training interrupted. Proceeding to testing with the best saved model...")
except Exception as e:
    print("Error during training:", e)
finally:
    print("Clean memory....")
    keras.backend.clear_session()
    gc.collect()
    print("Memory cleaned.")
