from keras import applications, layers, models, optimizers, Optimizer


class MultitaskResNet:
    def __init__(self, input_shape=(128, 128, 3)):
        self.input_shape = input_shape
        self.model = None

    def build_model(self):
        # Load the ResNet50 model without the top layer (pretrained on ImageNet)
        base_model = applications.ResNet50(include_top=False, input_shape=self.input_shape, weights='imagenet')
        base_model.trainable = False  # Freeze the base model

        # Shared layers (common backbone)
        x = base_model.output
        x = layers.GlobalAveragePooling2D()(x)

        # Task 1: Face/No-Face Classification (Binary Classification)
        face_output = layers.Dense(1, activation='sigmoid', name='face_output')(x)

        # Task 2: Age Prediction (Regression) with additional dense layers
        age_output = layers.Dense(64, activation='relu')(x)
        age_output = layers.Dense(32, activation='relu')(age_output)
        age_output = layers.Dense(1, activation='linear', name='age_output')(age_output)

        # Task 3: Gender Classification (Binary Classification)
        gender_output = layers.Dense(64, activation='relu')(x)  # First dense layer
        gender_output = layers.Dense(32, activation='relu')(gender_output)  # Second dense layer
        gender_output = layers.Dense(3, activation='softmax', name='gender_output')(gender_output)

        

        # Build and assign the model
        self.model = models.Model(inputs=base_model.input, outputs=[face_output, age_output, gender_output])

    def compile_model(self):
        if self.model is None:
            raise Exception("Model not built. Call 'build_model()' first.")
        
        # Compile the model with task-specific losses and metrics
        self.model.compile(optimizer=optimizers.Adam(),
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

    def train(self, x, y, validation_data, epochs, batch_size, callbacks):
        if self.model is None:
            raise Exception("Model not built. Call 'build_model()' first.")

        return self.model.fit(x, y, validation_data=validation_data, epochs=epochs, batch_size=batch_size, callbacks=callbacks)
    
    def evaluate(self, val_dataset):
        if self.model is None:
            raise Exception("Model not built. Call 'build_model()' first.")
        
        # Evaluate the model
        results = self.model.evaluate(val_dataset)
        return results

    def evaluate(self, x, y, batch_size, return_dict=False):
        if self.model is None:
            raise Exception("Model not built. Call 'build_model()' first.")

        self.model.evaluate(x, y, batch_size=batch_size, verbose=2, return_dict=return_dict)
    
    def save_model(self, filepath):
        if self.model is None:
            raise Exception("Model not built. Call 'build_model()' first.")
        
        # Save the model to a file
        self.model.save(filepath)

    def summary(self):
        if self.model is None:
            raise Exception("Model not built. Call 'build_model()' first.")

        self.model.summary()