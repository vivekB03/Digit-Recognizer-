from tensorflow.keras.datasets import mnist
import numpy as np

# Load from manually downloaded file
(x_train, y_train), (x_test, y_test) = mnist.load_data(path="D:/ml.py/mnist.npz")

print("✅ Data loaded successfully:")
print("Train shape:", x_train.shape)
print("Test shape:", x_test.shape)
