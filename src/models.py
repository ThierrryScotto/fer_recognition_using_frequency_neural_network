
from os import makedirs, path
import requests
import bz2
from src.logging import get_logger

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
