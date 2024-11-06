from keras import applications, layers, models, optimizers

class MultitaskResNet(models.Model):
    def __init__(self, input_shape=(128, 128, 3)):
        super(MultitaskResNet, self).__init__()
        self.input_shape = input_shape

        # Data Augmentation Layers
        self.random_flip = layers.RandomFlip("horizontal")  # Random horizontal flip
        self.random_rotation = layers.RandomRotation(0.1)    # Random rotation within 10 degrees
        self.random_zoom = layers.RandomZoom(0.05)            # Random zoom within 10%

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
                               'age_output': 'mean_squared_error',
                               'gender_output': 'sparse_categorical_crossentropy'
                           },
                           metrics={
                               'face_output': 'accuracy',
                               'age_output': 'mae',
                               'gender_output': 'accuracy'
                           })
