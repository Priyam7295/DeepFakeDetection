import numpy as np
import os
from PIL import Image
import numpy as np

# Specify the path to the directory containing the images

# Loading the fake GAN images 
directory_path_1 = "GANS/biggan/biggan"
directory_path_2 = "GANS/stargan/stargan"
directory_path_3 = "GANS/gaugan/gaugan"
directory_path_4 = "GANS/whichfaceisreal/whichfaceisreal"

# Create an empty list to store the loaded images
fake_image_array = []
def func(folder_path,array):
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            # Create the full path to the image using os.path.join()
            image_path = os.path.join(folder_path, filename)
            image = Image.open(image_path)
            resized_image = image.resize((256, 256))
            image_array = np.array(resized_image)
            array.append(image_array)
        # print("->",count)
  

func(directory_path_1,fake_image_array)
func(directory_path_2,fake_image_array)
func(directory_path_3,fake_image_array)
func(directory_path_4,fake_image_array)

# This fake_image_Array conatins the path of all GAN images 
fake_image_array = np.array(fake_image_array)


# Loadig real images 
directory_path_5 = "diffusion_datasets/imagenet/0_real"
directory_path_6 = "diffusion_datasets/laion/0_real"


real_image_array=[]

# This real_image_array contains the real images
func(directory_path_5,real_image_array)
func(directory_path_6,real_image_array)




print(len(real_image_array))
k=[]

for i in range(0,len(real_image_array)):
    if(real_image_array[i].shape ==  (256,256,3) ):
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

# X - contains total of fake and real images  


#  Y is the label where 
# 0 -- Fake Images
# 1 -- Real Images 
y =[]
for i in range(len(fake_image_array)):
    y.append(0)
for i in range(len(k)): 
    y.append(1)
y=np.array(y)
print(len(y))        


# importing temsorflow and dufferent dependencies for our purpose
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# Splitting the dataset into rotio 80:20 for training and testing purpose
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocess input for ResNet
# Using Image encoder for generalisation , which is the bigegst isuue in classifactaion 
datagen = ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.3,
    height_shift_range=0.3,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Preprocess input for ResNet
X_train = preprocess_input(X_train)
X_test = preprocess_input(X_test)

# Load pre-trained ResNet50 model without the top classification layer
resnet_base = ResNet50(weights='imagenet', include_top=False, input_shape=(256, 256, 3))

# Freeze the layers of the pre-trained ResNet model
for layer in resnet_base.layers:
    layer.trainable = False

# Build a new model on top of the pre-trained ResNet base with Batch Normalization and Dropout
model = models.Sequential([
    resnet_base,
    layers.GlobalAveragePooling2D(),
    layers.BatchNormalization(),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
early_stopping = EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True)
# Train the model with data augmentation
batch_size = 32
epochs = 21  

train_datagen = datagen.flow(X_train, y_train, batch_size=batch_size)

# Train the model with increased data augmentation
history = model.fit(
    train_datagen,
    steps_per_epoch=len(X_train) // batch_size,
    epochs=epochs,
    validation_data=(X_test, y_test)
)

'''                     Our Model is completed                                    '''


import matplotlib.pyplot as plt

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



# Now we are suing two DIFFUSION datsets model for validation that whether our model generalises to various unseen model as well.
# And we got satisfying reults

directory_path_validate = "diffusion_datasets/ldm_100/1_fake"
directory_path_validate_2 = "diffusion_datasets/ldm_200"
validate_array = []
validate_array_2 = []

def func(folder_path, array):
    count = 0
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(folder_path, filename)
            image = Image.open(image_path)
            resized_image = image.resize((256, 256))
            image_array = np.array(resized_image)

            # Apply the same preprocessing as for training images
            preprocessed_image = preprocess_input(image_array)

            array.append(preprocessed_image)
            print("->", count)
            count += 1

func(directory_path_validate, validate_array)
func(directory_path_validate_2, validate_array_2)

validate_array = np.array(validate_array)
validate_array_2 = np.array(validate_array_2)
print("Validation array length: ",validate_array.shape)

print(validate_array[0].shape)


def prediction_on_diff_images(validate_array):

    predictions = model.predict(validate_array)
    predicted_classes = np.argmax(predictions, axis=1)  # Assuming categorical predictions

    deepFake=0

    for i in range(len(predicted_classes)):
        if( predicted_classes[i] == 0):
            deepFake +=1
    print("Number of  fake images out of ",len(predicted_classes)," is ",deepFake)

prediction_on_diff_images(validate_array)
prediction_on_diff_images(validate_array_2)
