import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Configuration
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
NUM_EPOCHS = 10
SAVED_MODEL_DIR = "models/saved_model"

def create_model():
    # Define your model architecture here
    model = tf.keras.Sequential([
        tf.keras.applications.MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet'),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    return model

def prepare_dataset(data_dir):
    train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
    train_generator = train_datagen.flow_from_directory(
        data_dir,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='binary',
        subset='training'
    )
    validation_generator = train_datagen.flow_from_directory(
        data_dir,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='binary',
        subset='validation'
    )
    return train_generator, validation_generator

def train_model():
    data_dir = "data"
    train_generator, validation_generator = prepare_dataset(data_dir)

    model = create_model()  # Define your model architecture here
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Training the model
    model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // BATCH_SIZE,
        epochs=NUM_EPOCHS,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // BATCH_SIZE
    )

    # Save the trained model
    # if not os.path.exists(SAVED_MODEL_DIR):
    #     os.makedirs(SAVED_MODEL_DIR)
    # model.save(SAVED_MODEL_DIR)

if __name__ == "__main__":
    train_model()
