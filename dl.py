import tensorflow as tf
from tensorflow.keras import layers
import numpy as np

# =========================
# 1. LOAD DATASET
# =========================
train_data = tf.keras.utils.image_dataset_from_directory(
    "dataset/Train",   
    image_size=(128,128),
    batch_size=16,
    label_mode='binary'
)

# =========================
# 2. NORMALIZE DATA
# =========================
normalization_layer = layers.Rescaling(1./255)
train_data = train_data.map(lambda x, y: (normalization_layer(x), y))

# =========================
# 3. CNN MODEL
# =========================
model = tf.keras.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(128,128,3)),
    layers.MaxPooling2D(),

    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(),

    layers.Conv2D(128, (3,3), activation='relu'),
    layers.MaxPooling2D(),

    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

# =========================
# 4. COMPILE
# =========================
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.summary()

# =========================
# 5. TRAIN
# =========================
model.fit(train_data, epochs=10)

# =========================
# 6. TEST ON IMAGE
# =========================
from tensorflow.keras.preprocessing import image

img = image.load_img(r"C:\Users\shivs\College Project\DL PBL 2ND yr\test2.jpg", target_size=(128,128))
img_array = image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

prediction = model.predict(img_array)

if prediction[0][0] > 0.5:
    print("Camouflage Detected")
else:
    print("No Camouflage")