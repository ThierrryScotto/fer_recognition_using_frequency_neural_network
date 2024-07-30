
from typing import Optional
from os import makedirs, path
import requests
import bz2

class ShapePredictor:

    # URL do arquivo shape_predictor_68_face_landmarks.dat.bz2
    __url: str = "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
    __file_name: str = 'shape_predictor_68_face_landmarks'
    __path: str = "./content/model/"


    def __init__(self, output: Optional[str] = None):
        self.download(output)

    def download(self, output: Optional[str] = None):

        if not output:
            output = self.__path
        makedirs(output, exist_ok=True)
        compressed_dat_path = path.join(output, self.__file_name + '.dat.bz2')
        dat_path = path.join(output, self.__file_name + '.bz2')
        

        if not path.exists(compressed_dat_path) and not path.exists(dat_path):    
            with open(compressed_dat_path, 'wb') as compressed_file:
                response = requests.get(self.__url, stream=True)
                for chunk in response.iter_content(chunk_size=1024): 
                    if chunk: # filter out keep-alive new chunks
                        compressed_file.write(chunk)


        # Descompactar o arquivo
        if not path.exists(dat_path):
            with bz2.BZ2File(compressed_dat_path) as f_in:
                with open(dat_path, 'wb') as f_out:
                    f_out.write(f_in.read())

        print(f"Arquivo descompactado salvo em: {dat_path}")

        # Verificar se o arquivo foi descompactado corretamente
        if path.exists(dat_path):
            print("Arquivo descompactado com sucesso!")
        else:
            print("Falha ao descompactar o arquivo.")
