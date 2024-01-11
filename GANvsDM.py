import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os
from PIL import Image
import numpy as np

def load_images_from_folder(folder_path, target_size=(64, 64)):
    image_list = []
    for filename in os.listdir(folder_path):
        if filename.endswith(('.jpg', '.jpeg', '.png')):
            img_path = os.path.join(folder_path, filename)
            image = Image.open(img_path)
            image = image.resize(target_size)
            image_list.append(image)
    return image_list

def convert_images_to_matrix(image_list):
    matrix_list = []
    for img in image_list:
        
        matrix = np.array(img) / 255.0
        matrix_list.append(matrix)
    return matrix_list

gan_folder_path1 = "GANS/biggan/biggan"
gan_folder_path2 = "GANS/gaugan/gaugan"
gan_folder_path3 = "GANS/stargan/stargan"
# gan_folder_path4 = "GANS/whichfaceisreal/whichfaceisreal"
gan_folder_path5 = "gans_family/gans_family/biggan"
gan_folder_path6 = "gans_family/gans_family/stargan"
# gan_folder_path7 = "gans_family/gans_family/stylegan"

# Define the target size for resizing
target_size = (64,64)

gan_images1 = load_images_from_folder(gan_folder_path1, target_size)
gan_images2 = load_images_from_folder(gan_folder_path2, target_size)
gan_images3 = load_images_from_folder(gan_folder_path3, target_size)
# gan_images4 = load_images_from_folder(gan_folder_path4, target_size)
gan_images5 = load_images_from_folder(gan_folder_path5, target_size)
gan_images6 = load_images_from_folder(gan_folder_path6, target_size)
# gan_images7 = load_images_from_folder(gan_folder_path7, target_size)

# Convert images to matrices
gan_matrices1 = convert_images_to_matrix(gan_images1)
gan_matrices2 = convert_images_to_matrix(gan_images2)
gan_matrices3 = convert_images_to_matrix(gan_images3)
# gan_matrices4 = convert_images_to_matrix(gan_images4)
gan_matrices5 = convert_images_to_matrix(gan_images5)
gan_matrices6 = convert_images_to_matrix(gan_images6)
# gan_matrices7 = convert_images_to_matrix(gan_images7)


print(np.shape(gan_matrices1[0]))

gan_matrices1=np.array(gan_matrices1)
gan_matrices2=np.array(gan_matrices2)
gan_matrices3=np.array(gan_matrices3)
# gan_matrices4=np.array(gan_matrices4)
gan_matrices5=np.array(gan_matrices5)
gan_matrices6=np.array(gan_matrices6)
# gan_matrices7=np.array(gan_matrices7)

print(gan_matrices1.shape)
print(gan_matrices2.shape)
print(gan_matrices3.shape)
# print(gan_matrices4.shape)
print(gan_matrices5.shape)
print(gan_matrices6.shape)
# print(gan_matrices7.shape)


# Total Gan IMAGES

gan_images = []
for img in gan_matrices1:
    gan_images.append(img)
for img in gan_matrices2:
    gan_images.append(img)
for img in gan_matrices3:
    gan_images.append(img)
# for img in gan_matrices4:
#     gan_images.append(img)
for img in gan_matrices5:
    gan_images.append(img)
for img in gan_matrices6:
    gan_images.append(img)
# for img in gan_matrices7:
#     gan_images.append(img)

print("Total Gan Images ",len(gan_images))


dm_folder_path1 = "diffusion_datasets/dalle/1_fake"
dm_folder_path2 = "diffusion_datasets/glide_50_27/1_fake"
dm_folder_path3 = "diffusion_datasets/ldm_100/1_fake"
dm_folder_path4 = "diffusion_datasets/ldm_200/1_fake"
dm_folder_path5 = "diffusion_datasets/ldm_200_cfg/1_fake"
# dm_folder_path6 = "diffusion_datasets/guided/1_fake"
# dm_folder_path7 = "diffusion_datasets/glide_100_10/1_fake"

dm_images1 = load_images_from_folder(dm_folder_path1, target_size)
dm_images2 = load_images_from_folder(dm_folder_path2, target_size)
dm_images3 = load_images_from_folder(dm_folder_path3, target_size)
dm_images4 = load_images_from_folder(dm_folder_path4, target_size)
dm_images5 = load_images_from_folder(dm_folder_path5, target_size)
# dm_images6 = load_images_from_folder(dm_folder_path6, target_size)
# dm_images7 = load_images_from_folder(dm_folder_path7, target_size)

dm_matrices1 = convert_images_to_matrix(dm_images1)
dm_matrices2 = convert_images_to_matrix(dm_images2)
dm_matrices3 = convert_images_to_matrix(dm_images3)
dm_matrices4 = convert_images_to_matrix(dm_images4)
dm_matrices5 = convert_images_to_matrix(dm_images5)
# dm_matrices6 = convert_images_to_matrix(dm_images6)
# dm_matrices7 = convert_images_to_matrix(dm_images7)


dm_matrices1 = np.array(dm_matrices1)
dm_matrices2 = np.array(dm_matrices2)
dm_matrices3 = np.array(dm_matrices3)
dm_matrices4 = np.array(dm_matrices4)
dm_matrices5 = np.array(dm_matrices5)
# dm_matrices6 = np.array(dm_matrices6)
# dm_matrices7 = np.array(dm_matrices7)

print(dm_matrices1.shape)
print(dm_matrices2.shape)
print(dm_matrices3.shape)
print(dm_matrices4.shape)
print(dm_matrices5.shape)
# print(dm_matrices6.shape)
# print(dm_matrices7.shape)


dm_images = []

for im in dm_matrices1:
    dm_images.append(im)
for im in dm_matrices2:
    dm_images.append(im)
for im in dm_matrices3:
    dm_images.append(im)
for im in dm_matrices4:
    dm_images.append(im)
for im in dm_matrices5:
    dm_images.append(im)
# for im in dm_matrices6:
#     dm_images.append(im)
# for im in dm_matrices7:
#     dm_images.append(im)

print(len(dm_images))    

fake = []
                       
for img in gan_images:
    fake.append(img)
for img in dm_images:
    fake.append(img)
fake=np.array(fake)
print(fake.shape)    

y = []
for i in range(4185):
    y.append(1)
for i in range(6000-1000):
    y.append(0)
y= np.array(y)

dm_images = np.array(dm_images)
gan_images = np.array(gan_images)


from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

# Assuming fake, y, and other necessary imports are defined

# Split the data
X_train, X_test, y_train, y_test = train_test_split(fake, y, test_size=0.2, random_state=42)

# Data augmentation
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Build the model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(1, activation='sigmoid')  
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Train the model with data augmentation
history = model.fit(
    datagen.flow(X_train, y_train, batch_size=32),
    steps_per_epoch=len(X_train) // 32,
    epochs=20,
    validation_data=(X_test, y_test)
)

# Evaluate the model
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_accuracy}")

# Plot training and validation accuracy values
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# Plot training and validation loss values
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()


dm_test = "diffusion_datasets/ldm_200_cfg/1_fake"

dm_images_test = load_images_from_folder(dm_test, target_size)

dm_matrices_test = convert_images_to_matrix(dm_images_test)

dm_matrices_test = np.array(dm_matrices_test)

ys = model.predict(dm_matrices_test)
print(ys)



thershold=0.5

def findaccuracy(prediction,threshold=0.5):
    gan=0
    dm=0
    for i in range(len(prediction)):
        if(prediction[i]>=thershold):
            gan +=1
        else:
            dm +=1

    print("GAN---->>",gan)
    print("DM---->>",dm)
    print("Accuracy is --->>",dm/len(prediction))

findaccuracy(ys)

        
dm_test2 = "diffusion_datasets/glide_100_10/1_fake"

dm_images_test2 = load_images_from_folder(dm_test2, target_size)

dm_matrices_test2 = convert_images_to_matrix(dm_images_test2)

dm_matrices_test2 = np.array(dm_matrices_test2)

y2= model.predict(dm_matrices_test2)

def findaccuracy(prediction,threshold=0.5):
    gan=0
    dm=0
    for i in range(len(prediction)):
        if(prediction[i]>=thershold):
            gan +=1
        else:
            dm +=1

    print("GAN---->>",gan)
    print("DM---->>",dm)

    print("Accuracy is --->>",dm/len(prediction))

findaccuracy(y2)

dm_test3 = "diffusion_datasets/guided/1_fake"

dm_images_test3 = load_images_from_folder(dm_test3, target_size)

dm_matrices_test3 = convert_images_to_matrix(dm_images_test3)

dm_matrices_test3 = np.array(dm_matrices_test3)

y3= model.predict(dm_matrices_test3)


findaccuracy(y3)

dm_test4 = "diffusion_datasets/glide_100_27/1_fake"

dm_images_test4 = load_images_from_folder(dm_test4, target_size)

dm_matrices_test4 = convert_images_to_matrix(dm_images_test4)

dm_matrices_test4 = np.array(dm_matrices_test4)

y4= model.predict(dm_matrices_test4)



findaccuracy(y4)

gan_test = "GANS/whichfaceisreal/whichfaceisreal"
gan_images_test = load_images_from_folder(gan_test, target_size)

gan_matrices_test = convert_images_to_matrix(gan_images_test)

gan_matrices_test = np.array(gan_matrices_test)

y5= model.predict(gan_matrices_test)
g=0
for i in y5: 
    if(i>0.5):
        g +=1
        
print(g/len(y5))

