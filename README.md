# Fruit-and-Vegetables-Classifier
# Fruit and Vegetable Classifier

A **deep learning project** that classifies fruits and vegetables using a **Convolutional Neural Network (CNN)**. Trained on the **Fruits-360 dataset**, the model can predict new images in real-time with confidence scores.

---

## 🧠 Key Highlights

- **CNN Architecture**: Built with Conv2D, MaxPooling, Dense, and Dropout layers.
- **Preprocessing**: Resizes, normalizes, and prepares images for the model.
- **Real-time Predictions**: Outputs predicted class labels with confidence.
- **Visualization**: Displays images with predictions using Matplotlib.

---

## ⚙️ Tech Stack

Python | TensorFlow/Keras | OpenCV | NumPy | Matplotlib

---

## 💻 Usage

1. Install dependencies:
   ```bash
   pip install tensorflow opencv-python matplotlib

2. Load model and predict:
   ```bash
  from tensorflow.keras.models import load_model
  ```bash
  model = load_model('fruit_classifier_model.h5')
  ```bash
  img = preprocess_image('path_to_image.jpg')
  ```bash
  predicted_class, confidence = predict_class(model, img, class_labels)
  ```bash
  print(predicted_class, confidence)

3. Visualize prediction:
   ```bash
  import matplotlib.pyplot as plt
  ```bash
  plt.imshow(img)
  ```bash
  plt.title(f"Prediction: {predicted_class} ({confidence:.2f})")
  ```bash
  plt.show()

## Files

- **fruit_classifier_model.h5**: Trained CNN model
- **train.py**: Model training code
- **predict.py**: Image preprocessing & prediction
- **test_images/**: Images for testing

Author of Dataset: Harika – Machine Learning & Deep Learning Enthusiast, focusing on Computer Vision projects.
