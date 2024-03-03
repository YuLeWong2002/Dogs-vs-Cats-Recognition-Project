import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import warnings
from keras.preprocessing.image import load_img
from keras.preprocessing.image import ImageDataGenerator
from keras import Sequential
from keras.layers import Conv2D, MaxPool2D, Flatten, Dense
import PIL
from sklearn.model_selection import train_test_split
import tkinter as tk
from tkinter import filedialog


warnings.filterwarnings('ignore')


def read_data():
    input_path = []
    label = []

    for class_name in os.listdir("PetImages"):
        for path in os.listdir("PetImages/" + class_name):
            if class_name == 'Cat':
                label.append(0)
            else:
                label.append(1)
            input_path.append(os.path.join("PetImages", class_name, path))

    return input_path, label


def create_df(input_path, label):
    df = pd.DataFrame()
    df['images'] = input_path
    df['label'] = label
    df = df.sample(frac=1).reset_index(drop=True)
    df['label'] = df['label'].astype('str')

    return df


def check_img(df):
    l = []
    for i, image in enumerate(df['images']):
        try:
            img = PIL.Image.open(image)
        except Exception as e:
            l.append(image)
            print(f"Error opening image {i}: {e}")


def prepare_data(df):
    train, test = train_test_split(df, test_size=0.2, random_state=42)

    train_generator = ImageDataGenerator(
        rescale=1. / 255,  # normalization of images
        rotation_range=40,  # augmentation of images to avoid overfitting
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    val_generator = ImageDataGenerator(rescale=1. / 255)

    train_iterator = train_generator.flow_from_dataframe(
        train,
        x_col='images',
        y_col='label',
        target_size=(128, 128),
        batch_size=128,
        class_mode='binary'
    )

    val_iterator = val_generator.flow_from_dataframe(
        test,
        x_col='images',
        y_col='label',
        target_size=(128, 128),
        batch_size=128,
        class_mode='binary'
    )

    return train_iterator, val_iterator


def setup_model():
    model = Sequential([
        Conv2D(16, (3, 3), activation='relu', input_shape=(128, 128, 3)),
        MaxPool2D((2, 2)),
        Conv2D(32, (3, 3), activation='relu'),
        MaxPool2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPool2D((2, 2)),
        Flatten(),
        Dense(512, activation='relu'),
        Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()
    model.save("trained_model.h5")
    return model


def print_graph(model, train_iterator, val_iterator):
    history = model.fit(train_iterator, epochs=10, validation_data=val_iterator)
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    epochs = range(len(acc))
    plt.plot(epochs, acc, 'b', label='Training Accuracy')
    plt.plot(epochs, val_acc, 'r', label='Validation Accuracy')
    plt.title('Accuracy Graph')
    plt.legend()
    plt.figure()
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    plt.plot(epochs, loss, 'b', label='Training Loss')
    plt.plot(epochs, val_loss, 'r', label='Validation Loss')
    plt.title('Loss Graph')
    plt.legend()
    plt.show()


def open_file_dialog():
    root = tk.Tk()
    root.withdraw()
    root.call('wm', 'attributes', '.', '-topmost', True)

    file_path = filedialog.askopenfilename(initialdir="/",
                                           title="Select a File",
                                           filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.gif;*.bmp;*.tiff"),
                                                      ("All files", "*.*")])

    if file_path:
        print("Selected file:", file_path)
    else:
        print("No file selected")
    return file_path


def predict_image(model):
    image_path = open_file_dialog()  # path of the testing image
    img = load_img(image_path, target_size=(128, 128))  # load the image and set its size to corresponding size
    img = np.array(img)
    img = img / 255.0  # normalize the image
    img = img.reshape(1, 128, 128, 3)  # reshape for prediction
    prediction = model.predict(img)
    if prediction[0] > 0.5:
        label = 'Cat'
    else:
        label = 'Dog'
    print(f"Prediction for {image_path}: {label}")
    print(f"Probability: {prediction}")


def train_model():
    input_path, label = read_data()
    df = create_df(input_path, label)
    train_iterator, val_iterator = prepare_data(df)
    model = setup_model()
    print_graph(model, train_iterator, val_iterator)
