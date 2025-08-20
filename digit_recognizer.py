import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import cv2
import numpy as np
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import mnist


MODEL_FILENAME = "mnist_cnn_model.keras"

if not os.path.exists(MODEL_FILENAME):
    print("Training model... (this only happens once)")

    # Load data
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape(-1, 28, 28, 1).astype("float32") / 255.0
    x_test = x_test.reshape(-1, 28, 28, 1).astype("float32") / 255.0
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)

    # Build improved CNN model
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(10, activation='softmax')
    ])

    model.compile(optimizer='adam', 
                 loss='categorical_crossentropy', 
                 metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))

    model.save(MODEL_FILENAME)
    print(f"Model saved as {MODEL_FILENAME}")
else:
    print("Loading existing model...")

# Load model
model = load_model(MODEL_FILENAME)

# -------------------------------
# DRAWING INTERFACE
# -------------------------------
drawing = False
image = np.zeros((400, 400), np.uint8)  # Black drawing board

def draw_circle(event, x, y, flags, param):
    global drawing, image
    
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        cv2.circle(image, (x, y), 8, (255, 255, 255), -1)  # Thicker drawing
        
    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        cv2.circle(image, (x, y), 8, (255, 255, 255), -1)
        
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False

cv2.namedWindow('Draw a Digit (0-9)')
cv2.setMouseCallback('Draw a Digit (0-9)', draw_circle)

instructions = [
    "Press 'p' to predict",
    "Press 'c' to clear",
]

while True:
    # Display instructions on the image
    display_img = image.copy()
    for i, text in enumerate(instructions):
        cv2.putText(display_img, text, (10, 20 + i*20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
    
    cv2.imshow('Draw a Digit (0-9)', display_img)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('p'):  # Predict
        # Process the image to look like MNIST digits
        processed = cv2.resize(image, (28, 28), interpolation=cv2.INTER_AREA)
        processed = cv2.bitwise_not(processed)  # White digit on black background
        processed = processed.astype("float32") / 255.0
        processed = np.reshape(processed, (1, 28, 28, 1))
        
        # Predict
        prediction = model.predict(processed, verbose=0)
        digit = np.argmax(prediction)
        confidence = np.max(prediction)
        
        print(f"\nPredicted Digit: {digit}")
        print(f"Confidence: {confidence*100:.1f}%")
        print("Top predictions:")
        for i, prob in enumerate(prediction[0]):
            print(f"{i}: {prob*100:.1f}%")
        
        # Show the processed image that was fed to the model
        cv2.imshow('Processed Image', processed.reshape(28, 28))
        
    elif key == ord('c'):  # Clear
        image = np.zeros((400, 400), np.uint8)
        print("\nDrawing board cleared!")
        
    elif key == ord('q'):  # Quit
        break

cv2.destroyAllWindows()