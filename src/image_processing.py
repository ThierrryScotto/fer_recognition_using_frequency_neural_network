import dlib
from PIL import Image
import numpy as np
import os
from src.models import ShapePredictor
from src.logging import get_logger

log = get_logger('image_processing.py')

class ImagePreProcessor:

    __detector = dlib.get_frontal_face_detector()
    __predictor = dlib.shape_predictor(ShapePredictor())

    def __init__(self) -> None:
        pass

    @staticmethod
    def adjust_and_crop_face(image_path: str, output_folder: str):

        # Carregar a imagem
        log.debug('Loading image in dlib')
        image = dlib.load_rgb_image(image_path)

        # Detectar os rostos na imagem
        log.debug('Detecting faces in image')
        faces = ImagePreProcessor.__detector(image, 1)

        # Processar cada rosto detectado
        log.debug('Processing faces in image')
        for i, face in enumerate(faces):

            
            # Obter os pontos chave do rosto
            landmarks = ImagePreProcessor.__predictor(image, face)


            # Converter os pontos chave para um formato utilizável
            face_landmarks = {
                "left_eye": [(landmarks.part(j).x, landmarks.part(j).y) for j in range(36, 42)],
                "right_eye": [(landmarks.part(j).x, landmarks.part(j).y) for j in range(42, 48)],
                "left_eyebrow": [(landmarks.part(j).x, landmarks.part(j).y) for j in range(17, 22)],
                "right_eyebrow": [(landmarks.part(j).x, landmarks.part(j).y) for j in range(22, 27)],
                "chin": [(landmarks.part(j).x, landmarks.part(j).y) for j in range(0, 17)],
            }

            # Obter os pontos dos olhos, sobrancelhas e queixo
            left_eye = np.array(face_landmarks['left_eye'])
            right_eye = np.array(face_landmarks['right_eye'])
            left_eyebrow = np.array(face_landmarks['left_eyebrow'])
            right_eyebrow = np.array(face_landmarks['right_eyebrow'])
            chin = np.array(face_landmarks['chin'])

            # Calcular o ponto médio entre os olhos
            left_eye_center = left_eye.mean(axis=0)
            right_eye_center = right_eye.mean(axis=0)
            eye_center = ((left_eye_center + right_eye_center) / 2).astype(int)

            # Calcular o ângulo para horizontalizar os olhos
            dy = right_eye_center[1] - left_eye_center[1]
            dx = right_eye_center[0] - left_eye_center[0]
            angle = np.degrees(np.arctan2(dy, dx))

            # Rotacionar a imagem para horizontalizar os olhos
            pil_image = Image.fromarray(image)
            pil_image = pil_image.rotate(angle, center=tuple(eye_center), expand=True)

            # Recalcular os pontos chave após rotação
            rotated_image = np.array(pil_image)
            rotated_faces = ImagePreProcessor.__detector(rotated_image, 1)
            rotated_landmarks = ImagePreProcessor.__predictor(rotated_image, rotated_faces[0])

            # Obter os novos pontos chave para sobrancelhas e queixo
            left_eyebrow = np.array([(rotated_landmarks.part(j).x, rotated_landmarks.part(j).y) for j in range(17, 22)])
            right_eyebrow = np.array([(rotated_landmarks.part(j).x, rotated_landmarks.part(j).y) for j in range(22, 27)])
            chin = np.array([(rotated_landmarks.part(j).x, rotated_landmarks.part(j).y) for j in range(0, 17)])

            # Calcular o topo das sobrancelhas e o fundo do queixo
            top_of_eyebrows = min(left_eyebrow[:,1].min(), right_eyebrow[:,1].min())
            bottom_of_chin = chin[:,1].max()

            # Calcular as coordenadas do recorte
            top = top_of_eyebrows
            bottom = bottom_of_chin
            left = min(chin[:,0].min(), left_eyebrow[:,0].min(), right_eyebrow[:,0].min())
            right = max(chin[:,0].max(), left_eyebrow[:,0].max(), right_eyebrow[:,0].max())

            # Recortar a imagem
            cropped_face = pil_image.crop((left, top, right, bottom))
            image = cropped_face.resize((32, 32))
            # Salvar a imagem recortada com o nome original acrescido de "face"
            base_filename, ext = os.path.splitext(os.path.basename(image_path))
            face_image_path = os.path.join(output_folder, f'{base_filename}_face{ext}')
            image.save(face_image_path)
            log.info(f'Face {i+1} saved in file: {face_image_path}')

