# utils.py
import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

def prepare_notmnist_dataset(img_size=(28, 28), batch_size=128, seed=1234, data_dir='notmnist_data'):
    """
    Loads notMNIST dataset from .npz file and prepares TensorFlow datasets.
    
    Returns:
        ds_train: tf.data.Dataset for training
        ds_val: tf.data.Dataset for validation
        (seq_len, input_dim): tuple with sequence length and input feature size
    """
    npz_path = os.path.join(data_dir, 'notmnist.npz')
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"{npz_path} not found. Please run make_notmnist_npz.py first.")

    # Load data
    data = np.load(npz_path)
    images = data['images']  # shape (N, H, W)
    labels = data['labels']  # shape (N,)

    # Normalize images to [0,1]
    images = images.astype(np.float32) / 255.0

    # Flatten images along width (for RNN input: sequence_length=height)
    N, H, W = images.shape
    images = images.reshape(N, H, W)  # already H x W
    seq_len = H
    input_dim = W

    # Split into train and validation
    X_train, X_val, y_train, y_val = train_test_split(
        images, labels, test_size=0.2, random_state=seed, stratify=labels)

    # Create TensorFlow datasets
    ds_train = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    ds_train = ds_train.shuffle(buffer_size=10000, seed=seed).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    ds_val = tf.data.Dataset.from_tensor_slices((X_val, y_val))
    ds_val = ds_val.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return ds_train, ds_val, (seq_len, input_dim)
