import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Configuration
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
NUM_EPOCHS = 10

def create_model():
    model = tf.keras.Sequential([
            tf.keras.applications.MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet'),
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(64, activation='relu'),  # New hidden layer with ReLU activation
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
    return model

def prepare_dataset():
    train_dir = './data/train'
    val_dir = './data/test'
    train_filenames = os.listdir(train_dir)
    val_filenames = os.listdir(val_dir)

    # Data preprocessing
    train_images = []
    for filename in train_filenames:
        img = tf.keras.preprocessing.image.load_img(os.path.join(train_dir, filename), target_size=IMAGE_SIZE)
        img = tf.keras.preprocessing.image.img_to_array(img)
        img = img / 255.0  # Normalize to [0, 1]
        train_images.append(img)

    val_images = []
    for filename in val_filenames:
        img = tf.keras.preprocessing.image.load_img(os.path.join(val_dir, filename), target_size=IMAGE_SIZE)
        img = tf.keras.preprocessing.image.img_to_array(img)
        img = img / 255.0  # Normalize to [0, 1]
        val_images.append(img)

    train_labels = [0] * len(train_images)  # Replace 0 with appropriate labels
    val_labels = [0] * len(val_images)      # Replace 0 with appropriate labels

    return (tf.convert_to_tensor(train_images), tf.convert_to_tensor(train_labels)), (tf.convert_to_tensor(val_images), tf.convert_to_tensor(val_labels))

def train_model():
    (train_images, train_labels), (val_images, val_labels) = prepare_dataset()

    model = create_model()  # Define your model architecture here
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Training the model
    model.fit(
        train_images,
        train_labels,
        batch_size=BATCH_SIZE,
        epochs=NUM_EPOCHS,
        validation_data=(val_images, val_labels)
    )
    
    model.save('trained_model.h5')

if __name__ == "__main__":
    train_model()
