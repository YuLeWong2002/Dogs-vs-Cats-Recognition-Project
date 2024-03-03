import warnings
from keras.models import load_model
from model import train_model, predict_image

warnings.filterwarnings('ignore')

while True:
    print("Select the following options: ")
    print("1. Train model")
    print("2. Test an image")
    print("3. Exit the program")

    selection = input("Enter your choice: ")

    if selection == "1":
        train_model()

    elif selection == "2":
        model_path = "trained_model.h5"
        model = load_model(model_path)
        predict_image(model)

    elif selection == "3":
        print("Exited")
        break

    else:
        print("Invalid choice, pls enter your choice again(1, 2, 3)")

