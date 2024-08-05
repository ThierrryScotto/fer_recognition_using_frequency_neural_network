import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import dct, idct
from PIL import Image

# Função para aplicar DCT 2D
def apply_dct(image):
    return dct(dct(image.T, norm='ortho').T, norm='ortho')

# Função para realizar a CIE
def critical_information_extraction(dct_image, reduction_factor=0.25):
    rows, cols = dct_image.shape
    crow, ccol = int(rows * reduction_factor), int(cols * reduction_factor)
    cie_image = np.zeros_like(dct_image)
    cie_image[:crow, :ccol] = dct_image[:crow, :ccol]
    return cie_image

# Função para aplicar IDCT 2D
def apply_idct(dct_image):
    return idct(idct(dct_image.T, norm='ortho').T, norm='ortho')

# Carregar a imagem e converter para escala de cinza
image_path = './teste/output_face.png'
image = Image.open(image_path).convert('L')
image = np.array(image)

# Aplicar DCT
dct_image = apply_dct(image)

# Aplicar CIE com γ = 0.25
gamma = 0.25
cie_image = critical_information_extraction(dct_image, gamma)

# Aplicar IDCT para reconverter para o domínio espacial
reconstructed_image = apply_idct(cie_image)

# Exibir a imagem original
plt.figure()
plt.imshow(image, cmap='gray')
plt.title('Imagem Original')
plt.axis('off')
plt.show()

# Exibir a imagem após DCT
plt.figure()
plt.imshow(np.log1p(np.abs(dct_image)), cmap='gray')
plt.title('Imagem após DCT')
plt.axis('off')
plt.show()

# Exibir a imagem após CIE
plt.figure()
plt.imshow(np.log1p(np.abs(cie_image)), cmap='gray')
plt.title('Imagem após CIE')
plt.axis('off')
plt.show()

# Exibir a imagem reconstruída após IDCT
plt.figure()
plt.imshow(reconstructed_image, cmap='gray')
plt.title('Imagem Reconstruída após IDCT')
plt.axis('off')
plt.show()
