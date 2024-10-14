import os
import glob
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

# Load and preprocess images
//image_directory = 'C:\\Users\\alone\\AppData\\Local\\Programs\\Python\\Python312\\face'  # Image directory
image_directory = '\\workspaces\\celeba\\new_directory_name\\img_align_celeba'  # Image directory
image_size = (224, 224)  # Resize images
X_train = []
y_skin_color = []
y_hair_color = []
y_beard_shape = []
y_beard_color = []
y_mustache_shape = []
y_mustache_color = []
y_mustache_exists = []
y_glasses = []
y_eyebrow_shape = []
y_makeup = []
y_scar = []

# Find image files
image_paths = glob.glob(os.path.join(image_directory, '*.jpg')) + glob.glob(os.path.join(image_directory, '*.png'))

# Load images and labels
for image_path in image_paths:
    img = load_img(image_path, target_size=image_size)
    img_array = img_to_array(img) / 255.0  # Scale images
    X_train.append(img_array)

    # Randomly generate labels for diversity; adjust these according to your dataset
    skin_color = random.randint(0, 2)  # 3 classes for skin color
    hair_color = random.randint(0, 4)  # 5 classes for hair color
    beard_shape = random.randint(0, 2)  # 3 classes for beard shape
    beard_color = random.randint(0, 2)  # 3 classes for beard color
    mustache_shape = random.randint(0, 2)  # 3 classes for mustache shape
    mustache_color = random.randint(0, 2)  # 3 classes for mustache color
    mustache_exists = random.randint(0, 1)  # 1 class (binary)
    glasses = random.randint(0, 1)  # 1 class (binary)
    eyebrow_shape = random.randint(0, 2)  # 3 classes for eyebrow shape
    makeup = random.randint(0, 1)  # 1 class (binary)
    scar = random.randint(0, 1)  # 1 class (binary)

    # Append labels to lists
    y_skin_color.append(skin_color)
    y_hair_color.append(hair_color)
    y_beard_shape.append(beard_shape)
    y_beard_color.append(beard_color)
    y_mustache_shape.append(mustache_shape)
    y_mustache_color.append(mustache_color)
    y_mustache_exists.append(mustache_exists)
    y_glasses.append(glasses)
    y_eyebrow_shape.append(eyebrow_shape)
    y_makeup.append(makeup)
    y_scar.append(scar)

# Convert X_train to a NumPy array
X_train = np.array(X_train, dtype=np.float32)

# Convert labels to one-hot encoding and numpy arrays
y_skin_color = np.array(to_categorical(y_skin_color, num_classes=3), dtype=np.float32)
y_hair_color = np.array(to_categorical(y_hair_color, num_classes=5), dtype=np.float32)
y_beard_shape = np.array(to_categorical(y_beard_shape, num_classes=3), dtype=np.float32)
y_beard_color = np.array(to_categorical(y_beard_color, num_classes=3), dtype=np.float32)
y_mustache_shape = np.array(to_categorical(y_mustache_shape, num_classes=3), dtype=np.float32)
y_mustache_color = np.array(to_categorical(y_mustache_color, num_classes=3), dtype=np.float32)
y_eyebrow_shape = np.array(to_categorical(y_eyebrow_shape, num_classes=3), dtype=np.float32)

# Convert binary labels to numpy arrays
y_mustache_exists = np.array(y_mustache_exists, dtype=np.float32)
y_glasses = np.array(y_glasses, dtype=np.float32)
y_makeup = np.array(y_makeup, dtype=np.float32)
y_scar = np.array(y_scar, dtype=np.float32)

# Split into training and test sets for each label
X_train, X_test, y_skin_color_train, y_skin_color_test = train_test_split(X_train, y_skin_color, test_size=0.2, random_state=42)
y_hair_color_train, y_hair_color_test = train_test_split(y_hair_color, test_size=0.2, random_state=42)
y_beard_shape_train, y_beard_shape_test = train_test_split(y_beard_shape, test_size=0.2, random_state=42)
y_beard_color_train, y_beard_color_test = train_test_split(y_beard_color, test_size=0.2, random_state=42)
y_mustache_shape_train, y_mustache_shape_test = train_test_split(y_mustache_shape, test_size=0.2, random_state=42)
y_mustache_color_train, y_mustache_color_test = train_test_split(y_mustache_color, test_size=0.2, random_state=42)
y_mustache_exists_train, y_mustache_exists_test = train_test_split(y_mustache_exists, test_size=0.2, random_state=42)
y_glasses_train, y_glasses_test = train_test_split(y_glasses, test_size=0.2, random_state=42)
y_eyebrow_shape_train, y_eyebrow_shape_test = train_test_split(y_eyebrow_shape, test_size=0.2, random_state=42)
y_makeup_train, y_makeup_test = train_test_split(y_makeup, test_size=0.2, random_state=42)
y_scar_train, y_scar_test = train_test_split(y_scar, test_size=0.2, random_state=42)

# After splitting, check shapes of training labels
print("Shapes after splitting:")
print(f"y_skin_color_train shape: {y_skin_color_train.shape}")
print(f"y_hair_color_train shape: {y_hair_color_train.shape}")
print(f"y_beard_shape_train shape: {y_beard_shape_train.shape}")
print(f"y_beard_color_train shape: {y_beard_color_train.shape}")
print(f"y_mustache_shape_train shape: {y_mustache_shape_train.shape}")
print(f"y_mustache_color_train shape: {y_mustache_color_train.shape}")
print(f"y_eyebrow_shape_train shape: {y_eyebrow_shape_train.shape}")

# Model Architecture
input_layer = Input(shape=(224, 224, 3))

# Convolutional layers
x = Conv2D(32, (3, 3), activation='relu')(input_layer)
x = MaxPooling2D(pool_size=(2, 2))(x)
x = Conv2D(64, (3, 3), activation='relu')(x)
x = MaxPooling2D(pool_size=(2, 2))(x)
x = Conv2D(128, (3, 3), activation='relu')(x)
x = MaxPooling2D(pool_size=(2, 2))(x)

# Flatten and dense layers
x = Flatten()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)

# Output layers
skin_color_output = Dense(3, activation='softmax', name='skin_color_output')(x)
hair_color_output = Dense(5, activation='softmax', name='hair_color_output')(x)
beard_shape_output = Dense(3, activation='softmax', name='beard_shape_output')(x)
beard_color_output = Dense(3, activation='softmax', name='beard_color_output')(x)
mustache_shape_output = Dense(3, activation='softmax', name='mustache_shape_output')(x)
mustache_color_output = Dense(3, activation='softmax', name='mustache_color_output')(x)
eyebrow_shape_output = Dense(3, activation='softmax', name='eyebrow_shape_output')(x)
mustache_exists_output = Dense(1, activation='sigmoid', name='mustache_exists_output')(x)
glasses_output = Dense(1, activation='sigmoid', name='glasses_output')(x)
makeup_output = Dense(1, activation='sigmoid', name='makeup_output')(x)
scar_output = Dense(1, activation='sigmoid', name='scar_output')(x)

# Create model
model = Model(inputs=input_layer, outputs=[
    skin_color_output,
    hair_color_output,
    beard_shape_output,
    beard_color_output,
    mustache_shape_output,
    mustache_color_output,
    mustache_exists_output,
    glasses_output,
    eyebrow_shape_output,
    makeup_output,
    scar_output
])

# Compile model
model.compile(optimizer='adam',
              loss={
                  'skin_color_output': 'categorical_crossentropy',
                  'hair_color_output': 'categorical_crossentropy',
                  'beard_shape_output': 'categorical_crossentropy',
                  'beard_color_output': 'categorical_crossentropy',
                  'mustache_shape_output': 'categorical_crossentropy',
                  'mustache_color_output': 'categorical_crossentropy',
                  'mustache_exists_output': 'binary_crossentropy',
                  'glasses_output': 'binary_crossentropy',
                  'eyebrow_shape_output': 'categorical_crossentropy',
                  'makeup_output': 'binary_crossentropy',
                  'scar_output': 'binary_crossentropy',
              },
              metrics={
                  'skin_color_output': 'accuracy',
                  'hair_color_output': 'accuracy',
                  'beard_shape_output': 'accuracy',
                  'beard_color_output': 'accuracy',
                  'mustache_shape_output': 'accuracy',
                  'mustache_color_output': 'accuracy',
                  'mustache_exists_output': 'accuracy',
                  'glasses_output': 'accuracy',
                  'eyebrow_shape_output': 'accuracy',
                  'makeup_output': 'accuracy',
                  'scar_output': 'accuracy'
              })

# Train the model
history = model.fit(X_train,
                    [y_skin_color_train, y_hair_color_train, y_beard_shape_train, y_beard_color_train,
                     y_mustache_shape_train, y_mustache_color_train, y_mustache_exists_train, 
                     y_glasses_train, y_eyebrow_shape_train, y_makeup_train, y_scar_train],
                    epochs=100,  # Burayı 100 olarak değiştirin
                    batch_size=32,
                    validation_split=0.2,
                    steps_per_epoch=len(X_train) // 32)  # Burada adım sayısını belirtiyoruz


# Save the model
try:
    model.save('multi_class_model.keras')
    print("Model successfully saved.")
except Exception as e:
    print(f"Error saving model: {e}")

# TensorFlow Lite conversion
try:
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    with open('model.tflite', 'wb') as f:
        f.write(tflite_model)
    print("Model successfully converted to TensorFlow Lite format.")
except Exception as e:
    print(f"Error during conversion: {e}")

# TensorFlow Lite model inference
try:
    interpreter = tf.lite.Interpreter(model_path="model.tflite")
    interpreter.allocate_tensors()

    # Get input and output tensors
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Use a sample input for testing
    sample_image = np.random.rand(224, 224, 3).astype(np.float32)  # Create a sample input
    sample_image = np.expand_dims(sample_image, axis=0)  # Add batch dimension
    interpreter.set_tensor(input_details[0]['index'], sample_image)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    print("Model output:", output_data)
except Exception as e:
    print(f"Error during model inference: {e}")
