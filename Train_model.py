import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

def load_and_prepare_data():
    """Load and prepare the MNIST dataset"""
    print("🔍 Loading MNIST dataset...")
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    
    # Reshape and normalize images
    x_train = x_train.reshape((-1, 28, 28, 1)).astype("float32") / 255.0
    x_test = x_test.reshape((-1, 28, 28, 1)).astype("float32") / 255.0
    
    # Convert labels to one-hot encoding
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)
    
    print(f"📊 Dataset loaded: {len(x_train)} training, {len(x_test)} test images")
    return x_train, y_train, x_test, y_test

def build_model():
    """Build the CNN model architecture"""
    print("🏗️ Building model architecture...")
    
    model = Sequential([
        # First convolutional block
        Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1),
               kernel_initializer='he_normal', name='conv1'),
        MaxPooling2D((2, 2), name='pool1'),
        Dropout(0.25, name='drop1'),
        
        # Second convolutional block
        Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', name='conv2'),
        MaxPooling2D((2, 2), name='pool2'),
        Dropout(0.25, name='drop2'),
        
        # Classification block
        Flatten(name='flatten'),
        Dense(128, activation='relu', kernel_initializer='he_normal', name='dense1'),
        Dropout(0.5, name='drop3'),
        Dense(10, activation='softmax', name='output')
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
    )
    
    model.summary()
    return model

def train_model(model, x_train, y_train, x_test, y_test):
    """Train the model with callbacks and validation"""
    print("🎓 Starting model training...")
    
    # Callbacks for better training
    callbacks = [
        ModelCheckpoint(
            "best_mnist_model.keras",
            save_best_only=True,
            monitor='val_accuracy',
            mode='max'
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        )
    ]
    
    history = model.fit(
        x_train, y_train,
        validation_data=(x_test, y_test),
        epochs=20,  # Increased from 5 to allow for early stopping
        batch_size=128,
        callbacks=callbacks,
        verbose=1
    )
    
    print("✅ Training complete!")
    return history

def save_model(model):
    """Save the final model"""
    model.save("mnist_cnn_model.keras")
    print("💾 Model saved as mnist_cnn_model.keras")

def main():
    # Load and prepare data
    x_train, y_train, x_test, y_test = load_and_prepare_data()
    
    # Build model
    model = build_model()
    
    # Train model
    history = train_model(model, x_train, y_train, x_test, y_test)
    
    # Save final model
    save_model(model)
    
    # Evaluate final performance
    print("\n📈 Final Evaluation:")
    loss, accuracy, precision, recall = model.evaluate(x_test, y_test, verbose=0)
    print(f"Test Accuracy: {accuracy*100:.2f}%")
    print(f"Test Precision: {precision*100:.2f}%")
    print(f"Test Recall: {recall*100:.2f}%")

if __name__ == "__main__":
    main()