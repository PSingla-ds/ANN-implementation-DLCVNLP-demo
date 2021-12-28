import tensorflow as tf
import os
import numpy as np
import time

def get_timestamp(name):
    timestamp= time.asctime().replace(" ", "_").replace(":", "_")
    unique_name = f"{name}_at_{timestamp}"
    return unique_name

def get_callbacks(config, X_train):
    logs = config["logs"]
    unique_dir_name = get_timestamp("tb_logs")

    ## Tensorboard callbacks
    tensorboard_root_log_dir = os.path.join(logs["logs_dir"], logs["Tensorboard_root_log_dir"], unique_dir_name)

    os.makedirs(tensorboard_root_log_dir, exist_ok = True)

    tensorboard_cb = tf.keras.callbacks.TensorBoard(log_dir=tensorboard_root_log_dir)

    file_writer = tf.summary.create_file_writer(logdir= tensorboard_root_log_dir)

    with file_writer.as_default():
        images = np.reshape(X_train[10:30], (-1, 28, 28, 1)) ### <<< 20, 28, 28, 1
        tf.summary.image("20 handwritten digit samples", images, max_outputs=25, step=0)

    # Early stopping Callbacks

    params = config["params"]  
    
    early_stopping_cb = tf.keras.callbacks.EarlyStopping(patience=params["patience"],
                                            restore_best_weights=params["restore_best_weights"])

    # Checkpoint Callbacks

    artifacts = config["artifacts"]
    CKPT_dir = os.path.join(artifacts["artifacts_dir"], artifacts['Checkpoint_dir'])
    os.makedirs(CKPT_dir, exist_ok = True)

    CKPT_path = os.path.join(CKPT_dir, "model_ckpt.h5")

    checkpointing_cb = tf.keras.callbacks.ModelCheckpoint(CKPT_path, save_best_only=True)

    return [tensorboard_cb, early_stopping_cb, checkpointing_cb ]