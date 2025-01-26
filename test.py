# Import necessary libraries
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import numpy as np
import matplotlib.pyplot as plt
from tkinter import Tk, filedialog, Button, Label
from PIL import Image, ImageTk

# CIFAR-10 Class Names
class_names = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer',
               'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

# Load CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0  # Normalize data

# Simplified Model
model = models.Sequential([
    layers.Conv2D(16, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10)
])

# Compile and train the model
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5, batch_size=32, validation_data=(x_test, y_test))

# Function to preprocess the image
def preprocess_image(image_path):
    img = Image.open(image_path).resize((32, 32))  # Resize image to 32x32
    img_array = np.array(img) / 255.0  # Normalize
    img_array = img_array[np.newaxis, ...]  # Add batch dimension
    return img_array

# Function to predict the class of the image
def predict_image():
    file_path = filedialog.askopenfilename()  # Open file dialog to select image
    if file_path:
        img_array = preprocess_image(file_path)
        predictions = model.predict(img_array)
        predicted_class = class_names[np.argmax(predictions[0])]
        result_label.config(text=f"Predicted Class: {predicted_class}")
        # Display the selected image
        img = Image.open(file_path).resize((150, 150))
        img_tk = ImageTk.PhotoImage(img)
        image_label.config(image=img_tk)
        image_label.image = img_tk

# Create a simple UI using Tkinter
window = Tk()
window.title("CIFAR-10 Image Classifier")

# Add buttons and labels
select_button = Button(window, text="Select Image", command=predict_image)
select_button.pack(pady=10)

result_label = Label(window, text="Predicted Class: None", font=("Arial", 14))
result_label.pack(pady=10)

image_label = Label(window)
image_label.pack(pady=10)

# Run the UI loop
window.mainloop()
