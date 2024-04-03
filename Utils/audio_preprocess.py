# audio_preprocess.py
"""
@brief This module contains the functions to preprocess the audio file before the speech recognition model can process it.
"""
import pydub




def preprocess_audio(file_path):
    # Load the audio file
    audio = pydub.AudioSegment.from_file(file_path)

    # Convert stereo to mono
    audio = audio.set_channels(1)

    # Resample to 16kHz
    audio = audio.set_frame_rate(16000)

    # Export the preprocessed audio
    preprocessed_file_path = file_path.replace('.mp3', '_preprocessed.wav')
    audio.export(preprocessed_file_path, format='wav')

    return preprocessed_file_path


