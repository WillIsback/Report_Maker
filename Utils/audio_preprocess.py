# audio_preprocess.py
"""
@brief This module contains the functions to preprocess the audio file before the speech recognition model can process it.
"""
import pydub
import hashlib
from pathlib import Path
# Get the absolute path of the root directory of the project
root_dir = Path(__file__).resolve().parent.parent

def get_file_hash(file_path):
    with open(file_path, 'rb') as f:
        file_hash = hashlib.md5(usedforsecurity=False)
        while chunk := f.read(8192):
            file_hash.update(chunk)
    return file_hash.hexdigest()



def preprocess_audio(file_path, build_dataset= False):
    # Convert file_path to a pathlib.Path object
    file_path = Path(file_path)
    # Load the audio file
    audio = pydub.AudioSegment.from_file(file_path)

    # Convert stereo to mono
    audio = audio.set_channels(1)

    # Resample to 16kHz
    audio = audio.set_frame_rate(16000)

    if(build_dataset):
        preprocessed_file_path = root_dir / 'report' / 'dataset' / f'{file_path.stem}_preprocessed_mono_16khz.wav'
        # Create the directory if it doesn't exist
        preprocessed_file_path.parent.mkdir(parents=True, exist_ok=True)
        audio.export(preprocessed_file_path, format='wav')
    else:
        preprocessed_file_path = Path(str(file_path).replace('.mp3', '_preprocessed.wav'))
        audio.export(preprocessed_file_path, format='wav')

    return preprocessed_file_path


