import os
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Layer
from keras.utils import to_categorical
from scipy.fftpack import dct
import tensorflow as tf

# Definindo as categorias de expressões
expressions = ['raiva', 'medo', 'nojo', 'surpresa', 'alegria', 'tristeza']

# Função para pré-processar a imagem
def preprocess_image(image):
    image = cv2.resize(image, (128, 128))
    return image

# Função para dividir a imagem em blocos e aplicar DCT
def dct_transform_blocks(image, block_size):
    blocks = [dct(dct(image[x:x+block_size, y:y+block_size].T, norm='ortho').T, norm='ortho')
              for x in range(0, image.shape[0], block_size)
              for y in range(0, image.shape[1], block_size)]
    return np.array(blocks)

# Função para carregar o dataset do arquivo CSV
def load_dataset_from_csv(csv_path):
    data = pd.read_csv(csv_path)
    images = []
    labels = []
    for index, row in data.iterrows():
        pixels = np.array(row['pixels'].split(), dtype='float32').reshape(128, 128)
        image = preprocess_image(pixels)
        image_dct_blocks = dct_transform_blocks(image, block_size=8)
        images.append(image_dct_blocks)
        labels.append(row['label'] - 1)  # Assumindo que os labels são de 1 a 6 e precisamos de 0 a 5
    return np.array(images), np.array(labels)

# Camada de multiplicação com pesos compartilhados
class BlockMultiplicationLayer(Layer):
    def _init_(self, block_size, k, **kwargs):
        self.block_size = block_size
        self.k = k
        super(BlockMultiplicationLayer, self)._init_(**kwargs)
    
    def build(self, input_shape):
        self.weights = [self.add_weight(name=f'weight_{i}', shape=(self.block_size, self.block_size), initializer='random_normal', trainable=True) for i in range(self.k)]
        self.biases = [self.add_weight(name=f'bias_{i}', shape=(1,), initializer='random_normal', trainable=True) for i in range(self.k)]
        super(BlockMultiplicationLayer, self).build(input_shape)
    
    def call(self, x):
        outputs = []
        for i in range(self.k):
            output_blocks = []
            for j in range(tf.shape(x)[1]):
                block = x[:, j, :, :] * self.weights[i] + self.biases[i]
                output_blocks.append(block)
            outputs.append(tf.stack(output_blocks, axis=1))
        return tf.stack(outputs, axis=-1)

# Definindo a rede neural Block-FreNet
def create_block_frenet(input_shape, block_size, k_list):
    model = Sequential()
    # Multiplication Layers
    num_blocks = (input_shape[0] // block_size) * (input_shape[1] // block_size)
    current_shape = (num_blocks, block_size, block_size)
    for k in k_list:
        model.add(BlockMultiplicationLayer(block_size, k, input_shape=current_shape))
        current_shape = (num_blocks, block_size, block_size, k)
    # Summarization Layer
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=current_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    # Fully Connected Layers
    model.add(Dense(2048, activation='relu'))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(len(expressions), activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Caminho do arquivo CSV
csv_path = 'caminho/para/seu/arquivo.csv'

# Carregar e dividir o dataset
X, y = load_dataset_from_csv(csv_path)
X = X.reshape(X.shape[0], -1, 8, 8)
y = to_categorical(y, num_classes=len(expressions))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Parâmetros da Block-FreNet
k_list = [48, 2, 1, 1]  # Correspondente aos valores de (k1, k2, k3, k4)

# Criar e treinar o modelo
model = create_block_frenet((128, 128), 8, k_list)
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Fazer previsões no conjunto de teste
predictions = model.predict(X_test)
predicted_labels = np.argmax(predictions, axis=1)

# Salvar as classificações em uma planilha
output_dir = 'output'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

output_data = []
for i, image_path in enumerate(os.listdir(os.path.join(data_dir, 'test'))):
    output_data.append([image_path, expressions[predicted_labels[i]]])

df = pd.DataFrame(output_data, columns=['Imagem', 'Classificação'])
df.to_csv(os.path.join(output_dir, 'classificacoes.csv'), index=False)

# Calcular a acurácia no conjunto de teste
accuracy = np.mean(predicted_labels == np.argmax(y_test, axis=1))
print(f"Acurácia: {accuracy * 100:.2f}%")

print("Classificações salvas em classificacoes.csv")