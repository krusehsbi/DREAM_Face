from keras import applications, layers, models, optimizers, saving


class MultitaskResNet(models.Model):
    def __init__(self, input_shape=(128, 128, 3)):
        super(MultitaskResNet, self).__init__()
        self.input_shape = input_shape

        # Data Augmentation Layers
        self.random_flip = layers.RandomFlip("horizontal")  # Random horizontal flip
        self.random_rotation = layers.RandomRotation(0.1)  # Random rotation within 10 degrees
        self.random_zoom = layers.RandomZoom(0.05)  # Random zoom within 10%

        # Load the ResNet50 model without the top layer (pretrained on ImageNet)
        self.base_model = applications.ResNet50(include_top=False, input_shape=self.input_shape, weights='imagenet')
        self.base_model.trainable = False  # Freeze the base model

        # Shared layers (common backbone)
        self.pooling_1 = layers.GlobalAveragePooling2D()

        # Task 1: Face/No-Face Classification (Binary Classification)
        self.face_output = layers.Dense(1, activation='sigmoid', name='face_output')

        # Task 2: Age Prediction (Regression) with additional dense layers
        self.age_1 = layers.Dense(64, activation='relu', name='age_1')
        self.age_2 = layers.Dense(32, activation='relu', name='age_2')
        self.age_output = layers.Dense(1, activation='linear', name='age_output')

        # Task 3: Gender Classification (Binary Classification)
        self.gender_1 = layers.Dense(64, activation='relu', name='gender_1')  # First dense layer
        self.gender_2 = layers.Dense(32, activation='relu', name='gender_2')  # Second dense layer
        self.gender_output = layers.Dense(3, activation='softmax', name='gender_output')

    def build(self, *kwargs):
        self.base_model.build(self.input_shape)
        x = self.base_model.output_shape
        self.pooling_1.build(x)
        x = self.pooling_1.compute_output_shape(x)

        self.face_output.build(x)

        self.age_1.build(x)
        age_shape = self.age_1.compute_output_shape(x)
        self.age_2.build(age_shape)
        age_shape = self.age_2.compute_output_shape(age_shape)
        self.age_output.build(age_shape)

        self.gender_1.build(x)
        gender_shape = self.gender_1.compute_output_shape(x)
        self.gender_2.build(gender_shape)
        gender_shape = self.gender_2.compute_output_shape(gender_shape)
        self.gender_output.build(gender_shape)
        self.built = True

    def call(self, inputs):
        # Apply data augmentation
        x = self.random_flip(inputs)
        x = self.random_rotation(x)
        x = self.random_zoom(x)

        # Shared layers (common backbone)
        x = self.base_model(x)
        x = self.pooling_1(x)

        # Task 1: Face/No-Face Classification (Binary Classification)
        face_out = self.face_output(x)

        # Task 2: Age Prediction (Regression) with additional dense layers
        age_out = self.age_1(x)
        age_out = self.age_2(age_out)
        age_out = self.age_output(age_out)

        # Task 3: Gender Classification (Binary Classification)
        gender_out = self.gender_1(x)  # First dense layer
        gender_out = self.gender_2(gender_out)  # Second dense layer
        gender_out = self.gender_output(gender_out)

        return {'face_output': face_out, 'age_output': age_out, 'gender_output': gender_out}

    def compile_default(self):
        # Compile the model with task-specific losses and metrics
        super().compile(optimizer=optimizers.Adam(),
                        loss={
                            'face_output': 'binary_crossentropy',
                            'age_output': "mean_squared_error",
                            'gender_output': "sparse_categorical_crossentropy",
                        },
                        metrics={
                            'face_output': 'accuracy',
                            'age_output': 'mae',
                            'gender_output': 'accuracy'
                        })


class MultitaskResNetDropout(models.Model):
    def __init__(self, input_shape=(128, 128, 3), dropout_rate=0.3):
        super(MultitaskResNetDropout, self).__init__()
        self.input_shape = input_shape
        self.dropout_rate = dropout_rate

        # Masking for outputs
        self.masking = layers.Masking(mask_value=500)

        # Data Augmentation Layers
        self.random_flip = layers.RandomFlip("horizontal")
        self.random_rotation = layers.RandomRotation(0.1)
        self.random_zoom = layers.RandomZoom(0.1)

        # Load the ResNet50 model without the top layer (pretrained on ImageNet)
        self.base_model = applications.ResNet50(include_top=False, input_shape=self.input_shape, weights='imagenet')
        self.base_model.trainable = False  # Freeze the base model

        # Shared layers (common backbone)
        self.dropout = layers.Dropout(dropout_rate)
        self.pooling_1 = layers.GlobalAveragePooling2D()

        # Task 1: Face/No-Face Classification (Binary Classification) with dropout before output
        self.face_dropout = layers.Dropout(self.dropout_rate)
        self.face_output = layers.Dense(1, activation='sigmoid', name='face_output')

        # Task 2: Age Prediction (Regression) with additional dense layers (no dropout)
        self.age_1 = layers.Dense(64, activation='relu', name='age_1')
        self.age_1_norm = layers.BatchNormalization()
        self.age_2 = layers.Dense(32, activation='relu', name='age_2')
        self.age_2_norm = layers.BatchNormalization()
        self.age_output = layers.Dense(1, activation='linear', name='age_output')

        # Task 3: Gender Classification with additional dense layers and dropout before output
        self.gender_1 = layers.Dense(64, activation='relu', name='gender_1')
        self.gender_1_norm = layers.BatchNormalization()
        self.gender_2 = layers.Dense(32, activation='relu', name='gender_2')
        self.gender_2_norm = layers.BatchNormalization()
        self.gender_dropout = layers.Dropout(self.dropout_rate)
        self.gender_output = layers.Dense(3, activation='softmax', name='gender_output')

    def build(self, *kwargs):
        # Build shared and task-specific layers
        self.random_flip.build(self.input_shape)
        x = self.random_flip.compute_output_shape(self.input_shape)
        self.random_rotation.build(x)
        x = self.random_rotation.compute_output_shape(x)
        self.random_zoom.build(x)
        x = self.random_zoom.compute_output_shape(x)

        # Build base model
        self.base_model.build(x)
        x = self.base_model.output_shape

        self.dropout.build(x)
        x = self.dropout.compute_output_shape(x)
        self.pooling_1.build(x)
        x = self.pooling_1.compute_output_shape(x)

        # Face
        self.face_dropout.build(x)
        face_shape = self.face_dropout.compute_output_shape(x)
        self.face_output.build(face_shape)
        face_shape = self.face_output.compute_output_shape(x)

        # Age
        self.age_1.build(x)
        age_shape = self.age_1.compute_output_shape(x)
        self.age_2.build(age_shape)
        age_shape = self.age_2.compute_output_shape(age_shape)
        self.age_output.build(age_shape)
        age_shape = self.age_output.compute_output_shape(age_shape)

        # Gender
        self.gender_1.build(x)
        gender_shape = self.gender_1.compute_output_shape(x)
        self.gender_2.build(gender_shape)
        gender_shape = self.gender_2.compute_output_shape(gender_shape)
        self.gender_dropout.build(gender_shape)
        gender_shape = self.gender_dropout.compute_output_shape(gender_shape)
        self.gender_output.build(gender_shape)
        gender_shape = self.gender_output.compute_output_shape(gender_shape)

        self.built = True

    def call(self, inputs):
        # Apply data augmentation
        x = self.random_flip(inputs)
        x = self.random_rotation(x)
        x = self.random_zoom(x)

        # Shared layers (common backbone)
        x = self.base_model(x)

        x = self.pooling_1(x)
        x = self.dropout(x)

        # Task 1: Face/No-Face Classification with dropout before output
        face_out = self.face_dropout(x)
        face_out = self.face_output(face_out)

        # Task 2: Age Prediction (no dropout)
        age_out = self.age_1(x)
        age_out = self.age_1_norm(age_out)
        age_out = self.age_2(age_out)
        age_out = self.age_2_norm(age_out)
        age_out = self.age_output(age_out)
        # age_out = self.masking(age_out)

        # Task 3: Gender Classification with dropout before output
        gender_out = self.gender_1(x)
        gender_out = self.gender_1_norm(gender_out)
        gender_out = self.gender_2(gender_out)
        gender_out = self.gender_2_norm(gender_out)
        gender_out = self.gender_dropout(gender_out)
        gender_out = self.gender_output(gender_out)
        # gender_out = self.masking(gender_out)

        return {'face_output': face_out, 'age_output': age_out, 'gender_output': gender_out}

    def compile_default(self):
        # Compile the model with task-specific losses and metrics
        super().compile(optimizer=optimizers.Adam(),
                        loss={
                            'face_output': 'binary_crossentropy',
                            'age_output': "mean_squared_error",
                            'gender_output': "sparse_categorical_crossentropy",
                        },
                        metrics={
                            'face_output': 'accuracy',
                            'age_output': 'mae',
                            'gender_output': 'accuracy'
                        })
