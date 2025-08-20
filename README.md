# Digit-Recognizer-
# ✍️ Handwritten Digit Recognition using CNN & OpenCV

 📌 Project Overview  
This project demonstrates a **handwritten digit recognition system** using a **Convolutional Neural Network (CNN)** trained on the **MNIST dataset**.  
It allows users to draw a digit (0–9) on a virtual canvas using **OpenCV**, and the trained model predicts the digit in real-time with confidence scores.  

The project is built with **Python**, leveraging **TensorFlow/Keras, NumPy, and OpenCV**.  



🛠️ Libraries Used  
- **TensorFlow / Keras** → Building and training the CNN model  
- **OpenCV** → Drawing interface & real-time user interaction  
- **NumPy** → Numerical computations  



 🧠 CNN Model Logic  
1. Load the **MNIST dataset**.  
2. Normalize pixel values (0–255 → 0–1) and reshape images.  
3. Build a CNN with:  
   - Two **Conv2D + MaxPooling2D** layers  
   - **Flatten** and **Dense** layers  
   - Softmax output layer for 10 classes (digits 0–9)  
4. Train for **10 epochs** and save the model as `mnist_cnn_model.keras`.  
5. If the model already exists, it loads directly without retraining.  



 🎨 Drawing Interface (OpenCV)  
- A **400x400 black canvas** is created for the user to draw digits.  
- **Mouse events** capture strokes with `cv2.setMouseCallback()`.  
- Controls:  
  - Press **'p'** → Predict the drawn digit  
  - Press **'c'** → Clear the canvas  
  - Press **'q'** → Quit the application  

**Prediction Pipeline:**  
1. Resize drawn image to **28×28 pixels**.  
2. Invert colors (white digit on black background).  
3. Normalize & reshape input.  
4. Predict digit using trained CNN.  
5. Print **digit + confidence scores** in the console.  



 📂 Project Files  
- `digit_recognizer.py` → Main program (training + interface)  
- `mnist_cnn_model.keras` → Saved CNN model (auto-generated after training)  


