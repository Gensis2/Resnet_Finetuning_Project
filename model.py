import tensorflow as tf
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.applications import ResNet50V2
from tensorflow.keras.optimizers import Adam

class ResNetClassifier:
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.resnet_model = ResNet50V2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        unfreeze_index = len(self.resnet_model.layers) - 3
        for layer in self.resnet_model.layers[unfreeze_index:]:
            layer.trainable = False
        self.add_classification_layers()

    def add_classification_layers(self):
        x = GlobalAveragePooling2D()(self.resnet_model.output)
        x = Dense(64, activation='relu')(x)
        x = Dropout(0.5)(x)
        x = Dense(64, activation='relu')(x)
        predictions = Dense(self.num_classes, activation='softmax')(x)
        self.model = Model(inputs=self.resnet_model.input, outputs=predictions)

    def compile_model(self):
        self.model.compile(optimizer=Adam(learning_rate = 0.002), loss='categorical_crossentropy', metrics=['accuracy'])

    def fit(self, training_data, batch_size, epochs, validation_data=None):
        if validation_data is not None:
            self.model.fit(training_data, batch_size=batch_size, epochs=epochs, validation_data=validation_data)
        else:
            self.model.fit(training_data, batch_size=batch_size, epochs=epochs)


    def evaluate(self, training_data):
        return self.model.evaluate(training_data)

    def predict(self, x):
        return self.model.predict(x)

    def summary(self):
        self.model.summary()