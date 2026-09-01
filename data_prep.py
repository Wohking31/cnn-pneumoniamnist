import os
import numpy as np
import medmnist
from medmnist import PneumoniaMNIST

def prepare_data():
    os.makedirs('data', exist_ok=True)
    
    # Download and load PneumoniaMNIST dataset
    train_dataset = PneumoniaMNIST(split='train', download=True)
    val_dataset = PneumoniaMNIST(split='val', download=True)
    test_dataset = PneumoniaMNIST(split='test', download=True)
    
    # Normalize images to [0, 1] range and add channel axis (28, 28, 1)
    X_train = (train_dataset.imgs / 255.0)[..., np.newaxis].astype(np.float32)
    y_train = train_dataset.labels.astype(np.int64)
    
    X_val = (val_dataset.imgs / 255.0)[..., np.newaxis].astype(np.float32)
    y_val = val_dataset.labels.astype(np.int64)
    
    X_test = (test_dataset.imgs / 255.0)[..., np.newaxis].astype(np.float32)
    y_test = test_dataset.labels.astype(np.int64)
    
    # Save .npy arrays to local disk
    np.save('data/X_train.npy', X_train)
    np.save('data/y_train.npy', y_train)
    np.save('data/X_val.npy', X_val)
    np.save('data/y_val.npy', y_val)
    np.save('data/X_test.npy', X_test)
    np.save('data/y_test.npy', y_test)
    
    print("PneumoniaMNIST data preparation complete. Files saved in data/ directory.")

if __name__ == '__main__':
    prepare_data()