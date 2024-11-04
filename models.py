from keras import applications, layers, models, optimizers
import keras

class SingletaskResNet(keras.Model):
    def __init__(self, input_shape=(128, 128, 3)):
        super().__init__()

        self.input_shape = input_shape
        # Load the ResNet50 model without the top layer (pretrained on ImageNet)
        self.base_model = applications.ResNet50(
            include_top=False,
            input_shape=input_shape,
            weights='imagenet')
        self.base_model.trainable = False  # Freeze the base model

        self.face_output = layers.Dense(1, activation='sigmoid', name='face_output')

    def build(self, input_shape):
        self.base_model.build(input_shape)
        input_shape = self.base_model.output_shape

        self.face_output.build(input_shape)
        self.built = True

    def call(self, inputs):
        # Forward pass through ResNet50
        x = self.base_model(inputs)
        x = layers.GlobalAveragePooling2D()(x)

        # Face presence
        face_out = self.face_output(x)

        # Return all three outputs
        return {"face_output": face_out}

    def compile(self,
                optimizer=optimizers.Adam(),
                loss=None,
                loss_weights=None,
                metrics=None,
                weighted_metrics=None,
                run_eagerly=False,
                steps_per_execution=1,
                jit_compile="auto",
                auto_scale_loss=True):
        # Init default values
        if loss is None:
            loss = {"face_output": "binary_crossentropy"}
        if metrics is None:
            metrics = {
                'face_output': 'accuracy'}

        # Compile Model
        super().compile(optimizer, loss, loss_weights, metrics, weighted_metrics, run_eagerly, steps_per_execution,
                        jit_compile, auto_scale_loss)

class MultitaskResNet(keras.Model):
    def __init__(self, input_shape=(128, 128, 3)):
        super().__init__()

        self.input_shape = input_shape
        # Load the ResNet50 model without the top layer (pretrained on ImageNet)
        self.base_model = applications.ResNet50(
            include_top=False,
            input_shape=input_shape,
            weights='imagenet')
        self.base_model.trainable = False  # Freeze the base model

        self.face_output = layers.Dense(1, activation='sigmoid', name='face_output')

        # Task 2: Age Prediction (Regression) with additional dense layers
        self.age_1 = layers.Dense(64, activation='relu')
        self.age_2 = layers.Dense(32, activation='relu')
        self.age_output = layers.Dense(1, activation='linear', name='age_output')

        # Task 3: Gender Classification (Binary Classification)
        self.gender_1 = layers.Dense(64, activation='relu')
        self.gender_2 = layers.Dense(32, activation='relu')
        self.gender_output = layers.Dense(1, activation='sigmoid', name='gender_output')

    def build(self, input_shape):
        self.base_model.build(input_shape)
        input_shape = self.base_model.output_shape

        self.face_output.build(input_shape)

        self.age_1.build(input_shape)
        age_shape = self.age_1.compute_output_shape(input_shape)
        self.age_2.build(age_shape)
        age_shape = self.age_2.compute_output_shape(age_shape)
        self.age_output.build(age_shape)

        self.gender_1.build(input_shape)
        gender_shape = self.gender_1.compute_output_shape(input_shape)
        self.gender_2.build(gender_shape)
        gender_shape = self.gender_2.compute_output_shape(gender_shape)
        self.gender_output.build(gender_shape)
        self.built = True

    def call(self, inputs):
        # Forward pass through ResNet50
        x = self.base_model(inputs)
        x = layers.GlobalAveragePooling2D()(x)

        # Face presence
        face_out = self.face_output(x)

        # Age Prediction
        age_x = self.age_1(x)
        age_x = self.age_2(age_x)
        age_out = self.age_output(age_x)

        # Gender Prediction
        gender_x = self.gender_1(x)
        gender_x = self.gender_2(gender_x)
        gender_out = self.gender_output(gender_x)

        # Return all three outputs
        return {"face_output": face_out,
                "age_output": age_out,
                "gender_output": gender_out}

    def compile(self,
                optimizer=optimizers.Adam(),
                loss=None,
                loss_weights=None,
                metrics=None,
                weighted_metrics=None,
                run_eagerly=False,
                steps_per_execution=1,
                jit_compile="auto",
                auto_scale_loss=True):
        # Init default values
        if loss is None:
            loss = {"face_output": "binary_crossentropy",
                    "age_output": "mean_squared_error",
                    "gender_output": "binary_crossentropy"}
        if metrics is None:
            metrics = {
                'face_output': 'accuracy',
                'age_output': 'mae',
                'gender_output': 'accuracy'}

        # Compile Model
        super().compile(optimizer, loss, loss_weights, metrics, weighted_metrics, run_eagerly, steps_per_execution,
                        jit_compile, auto_scale_loss)