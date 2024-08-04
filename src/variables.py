from os import environ, path

LOG_LEVEL: str = environ.get('LOG_LEVEL', 'INFO')
HEIGHT = 48
WIDTH = 48
OUTPUT_FOLDER = './content/images/input_images'
OUTPUT_PATH = path.join(OUTPUT_FOLDER, 'reconstructed_image')