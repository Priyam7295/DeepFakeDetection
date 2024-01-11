import numpy as np
import os
from PIL import Image
import numpy as np

# Specify the path to the directory containing the images
directory_path_1 = "GANS/biggan/biggan"
directory_path_2 = "GANS/stargan/stargan"
directory_path_3 = "GANS/gaugan/gaugan"
directory_path_4 = "GANS/whichfaceisreal/whichfaceisreal"

# Create an empty list to store the loaded images
fake_image_array = []
def func(folder_path,array):
    count=0
    for filename in os.listdir(folder_path):
        # Check if the file is a valid image file (you can customize this check based on your needs)
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            # Create the full path to the image using os.path.join()
            image_path = os.path.join(folder_path, filename)

            # Open the image using the PIL library
            image = Image.open(image_path)
            resized_image = image.resize((128, 128))
            image_array = np.array(resized_image)/255
            array.append(image_array)
        # print("->",count)
        count += 1

func(directory_path_1,fake_image_array)
func(directory_path_2,fake_image_array)
func(directory_path_3,fake_image_array)
func(directory_path_4,fake_image_array)

fake_image_array = np.array(fake_image_array)

# print(len(fake_image_array))
print(fake_image_array.shape)
print(fake_image_array[0].shape)
print(fake_image_array[0])

directory_path_5 = "diffusion_datasets/imagenet/0_real"
directory_path_6 = "diffusion_datasets/laion/0_real"

real_image_array=[]

func(directory_path_5,real_image_array)
func(directory_path_6,real_image_array)


print(len(real_image_array))
k=[]

for i in range(0,len(real_image_array)):
    if(real_image_array[i].shape ==  (128,128,3) ):
        k.append(real_image_array[i])

print(len(k))
k=np.array(k)
print(k.shape)

X =[]

for i in range(len(fake_image_array)):
    X.append(fake_image_array[i])
for i in range(len(k)):
    X.append(k[i])
X=np.array(X)
print(len(X))  
print(X[0])  

y =[]
for i in range(len(fake_image_array)):
    y.append(0)
for i in range(len(k)): 
    y.append(1)
y=np.array(y)
print(len(y))


from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

# X_train_resized = np.array([np.array(Image.fromarray(img).resize((128, 128))) for img in X_train])
# X_test_resized = np.array([np.array(Image.fromarray(img).resize((128, 128))) for img in X_test])

# Data augmentation using ImageDataGenerator
datagen = ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.3,
    height_shift_range=0.3,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Create a simpler CNN model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
early_stopping = EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True)

# Data augmentation using ImageDataGenerator
batch_size =16
train_datagen = datagen.flow(X_train, y_train, batch_size=batch_size)
# Train the simplified model with increased data augmentation
epochs=20
history = model.fit(
    train_datagen,
    steps_per_epoch=len(X_train) // batch_size,
    epochs=epochs,
    validation_data=(X_test, y_test)
)


import matplotlib.pyplot as plt



# Plot accuracy and loss graphs
plt.figure(figsize=(12, 4))

# Plot Training & Validation Accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Plot Training & Validation Loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()

# Evaluate the model on the testing data
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"\nTest Accuracy: {test_accuracy * 100:.2f}%")
print(f"Test Loss: {test_loss:.4f}")


import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# Evaluate the model on the testing data
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"\nTest Accuracy: {test_accuracy * 100:.2f}%")
print(f"Test Loss: {test_loss:.4f}")

# Generate predictions for the test set
y_pred = model.predict(X_test)
threshhold=0.5
y_pred_classes = (y_pred > threshhold).astype(int)  

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


cm = confusion_matrix(y_test, y_pred_classes)

# Visualize the confusion matrix with text annotations
plt.figure(figsize=(8, 6))
plt.imshow(cm, cmap=plt.cm.Blues)
plt.colorbar()
plt.xlabel("Predicted label")
plt.ylabel("True label")
plt.xticks(ticks=range(2), labels=["0:fake", "1:real"])  
plt.yticks(ticks=range(2), labels=["0:fake", "1:real"])
plt.title("Confusion Matrix")


for i in range(2):
    for j in range(2):
        text = plt.text(j, i, cm[i, j], ha="center", va="center", fontsize=12, color="white")

plt.show()

directory_path_validate = "diffusion_datasets/ldm_100/1_fake"
directory_path_validate_2 = "diffusion_datasets/ldm_200/1_fake"
directory_path_validate_3 = "diffusion_datasets/laion/0_real"
directory_path_validate_4 = "gans_family/gans_family/biggan"
validate_array = []
validate_array_2 = []
# validate_array_3 = []
validate_array_4 = []

def func(folder_path, array):
    count = 0
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(folder_path, filename)
            image = Image.open(image_path)
            resized_image = image.resize((128, 128))
            image_array = np.array(resized_image)/255

            array.append(image_array)
            print("->", count)
            count += 1

func(directory_path_validate, validate_array)
func(directory_path_validate_2, validate_array_2)
# func(directory_path_validate_3, validate_array_3)
func(directory_path_validate_4, validate_array_4)

validate_array = np.array(validate_array)
validate_array_2 = np.array(validate_array_2)
# validate_array_3 = np.array(validate_array_3)
validate_array_4 = np.array(validate_array_4)


print(validate_array[0])
print(validate_array[0].shape)


def prediction_on_diff_images(validate_array):

    predictions = model.predict(validate_array)
    deepFake=0

    for i in range(len(predictions)):
        if( predictions[i] < 0.5):
            deepFake +=1
    # print("Number of  fake images out of -> ",len(predictions)," is ",deepFake)
    
    return deepFake/len(validate_array)


a1=prediction_on_diff_images(validate_array)
a2=prediction_on_diff_images(validate_array_2)
a3=prediction_on_diff_images(validate_array_4)
a4=prediction_on_diff_images(k)

print("Accuracy on 1st DUFFUSION family- >",a1)
print("Accuracy on 2nd DUFFUSION family- >",a2)
print("Accuracy on 1st GAN family- >",a3)

