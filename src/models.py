
from os import makedirs, path
import requests
import bz2
from src.logging import get_logger
import tensorflow as tf

log = get_logger('models.py')
class ShapePredictor:

    # URL do arquivo shape_predictor_68_face_landmarks.dat.bz2
    __url: str = "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
    __file_name: str = 'shape_predictor_68_face_landmarks'
    __path: str = "./content/model/"
    __dat_path: str = path.join(__path, __file_name + '.dat')
    __compressed_dat_path = __dat_path + '.dat.bz2'


    def __new__(cls):
        ShapePredictor.download()
        return ShapePredictor.__dat_path

    @staticmethod
    def download():

        makedirs(ShapePredictor.__path, exist_ok=True)

        if not path.exists(ShapePredictor.__compressed_dat_path) and not path.exists(ShapePredictor.__dat_path):    
            with open(ShapePredictor.__compressed_dat_path, 'wb') as compressed_file:
                log.debug(f"Downloading file: {ShapePredictor.__compressed_dat_path}")
                response = requests.get(ShapePredictor.__url, stream=True)
                for chunk in response.iter_content(chunk_size=1024): 
                    if chunk: # filter out keep-alive new chunks
                        compressed_file.write(chunk)


        # Descompactar o arquivo
        if not path.exists(ShapePredictor.__dat_path):
            with bz2.BZ2File(ShapePredictor.__compressed_dat_path) as f_in:
                with open(ShapePredictor.__dat_path, 'wb') as f_out:
                    f_out.write(f_in.read())

            log.debug(f"Saving file: {ShapePredictor.__dat_path}")

        # Verificar se o arquivo foi descompactado corretamente
        if path.exists(ShapePredictor.__dat_path):
            log.info(f"Loaded model: {ShapePredictor.__dat_path}")
        else:
            log.error(f"Failed to load model: {ShapePredictor.__dat_path}")
            raise FileNotFoundError(f"Could not find file: {ShapePredictor.__dat_path}")

import tensorflow as tf

# Ensure TensorFlow 2.17.0 is used
assert tf.__version__ == '2.17.0', f"Expected TensorFlow version 2.17.0 but got {tf.__version__}"

# Function to create the Learnable Multiplication Kernel (LMK) layer
class LMKLayer(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size, **kwargs):
        super(LMKLayer, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size

    def build(self, input_shape):
        self.kernel = self.add_weight(
            shape=self.kernel_size,
            initializer='truncated_normal',
            trainable=True,
            name='kernel'
        )
        self.bias = self.add_weight(
            shape=[self.filters],
            initializer='zeros',
            trainable=True,
            name='bias'
        )

    def call(self, inputs):
        conv = tf.nn.conv2d(inputs, self.kernel, strides=[1, 1, 1, 1], padding='SAME')
        conv = tf.nn.bias_add(conv, self.bias)
        return conv

# Define the Basic-FreNet model using tf.keras.Model
class BasicFreNetModel(tf.keras.Model):
    def __init__(self, n_layers, k_values, conv_kernel_size, filters_summary, **kwargs):
        super(BasicFreNetModel, self).__init__(**kwargs)
        self.n_layers = n_layers
        self.lmk_layers = [LMKLayer(filters=f, kernel_size=conv_kernel_size) for f in k_values]
        self.summary_lmk_layer = LMKLayer(filters=filters_summary, kernel_size=conv_kernel_size)
        self.pool = tf.keras.layers.MaxPooling2D(pool_size=[2, 2], strides=2, padding='SAME')
        self.flatten = tf.keras.layers.Flatten()
        self.fc1 = tf.keras.layers.Dense(units=2048, activation='relu')
        self.dropout1 = tf.keras.layers.Dropout(0.5)
        self.fc2 = tf.keras.layers.Dense(units=512, activation='relu')
        self.dropout2 = tf.keras.layers.Dropout(0.5)
        self.output_layer = tf.keras.layers.Dense(units=6, activation='softmax')

    def call(self, inputs):
        x = inputs
        for lmk_layer in self.lmk_layers:
            x = lmk_layer(x)
        x = self.summary_lmk_layer(x)
        x = self.pool(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.dropout2(x)
        x = self.output_layer(x)
        return x