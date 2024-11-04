from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

class MultitaskResNet:
    def __init__(self, input_shape=(128, 128, 3)):
        self.input_shape = input_shape
        self.model = None

    def build_model(self):
        # Load the ResNet50 model without the top layer (pretrained on ImageNet)
        base_model = ResNet50(include_top=False, input_shape=self.input_shape, weights='imagenet')
        base_model.trainable = False  # Freeze the base model

        # Shared layers (common backbone)
        x = base_model.output
        x = GlobalAveragePooling2D()(x)

        # Task 1: Face/No-Face Classification (Binary Classification)
        face_output = Dense(1, activation='sigmoid', name='face_output')(x)

        # Task 2: Age Prediction (Regression) with additional dense layers
        age_output = Dense(64, activation='relu')(x)
        age_output = Dense(32, activation='relu')(age_output)
        age_output = Dense(1, activation='linear', name='age_output')(age_output)

        # Task 3: Gender Classification (Binary Classification)
        gender_output = Dense(64, activation='relu')(x)  # First dense layer
        gender_output = Dense(32, activation='relu')(gender_output)  # Second dense layer
        gender_output = Dense(3, activation='softmax', name='gender_output')(gender_output)

        

        # Build and assign the model
        self.model = Model(inputs=base_model.input, outputs=[face_output, age_output, gender_output])

    def compile_model(self):
        if self.model is None:
            raise Exception("Model not built. Call 'build_model()' first.")
        
        # Compile the model with task-specific losses and metrics
        self.model.compile(optimizer=Adam(),
                           loss={
                               'face_output': 'binary_crossentropy',
                               'age_output': 'mean_squared_error',
                               'gender_output': 'sparse_categorical_crossentropy'
                           },
                           metrics={
                               'face_output': 'accuracy',
                               'age_output': 'mae',
                               'gender_output': 'accuracy'
                           })
    
    def train(self, train_dataset, val_dataset, epochs=10, callbacks=None):
        if self.model is None:
            raise Exception("Model not built. Call 'build_model()' first.")
        
        # Train the model
        history = self.model.fit(train_dataset,
                                 validation_data=val_dataset,
                                 epochs=epochs,
                                 callbacks=callbacks)
        return history
    
    def evaluate(self, val_dataset):
        if self.model is None:
            raise Exception("Model not built. Call 'build_model()' first.")
        
        # Evaluate the model
        results = self.model.evaluate(val_dataset)
        return results
    
    def save_model(self, filepath):
        if self.model is None:
            raise Exception("Model not built. Call 'build_model()' first.")
        
        # Save the model to a file
        self.model.save(filepath)
