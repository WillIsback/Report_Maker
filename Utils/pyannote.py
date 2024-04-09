import warnings
from dotenv import load_dotenv
import os
import torch
import torchaudio
from pyannote.audio import Pipeline
from pathlib import Path

# Suppress the UserWarning from torchaudio
warnings.filterwarnings("ignore", category=UserWarning)


load_dotenv()

# Get the Hugging Face API key from the environment variables
HUGGING_FACE = os.getenv('HUGGING_FACE')

class Pyannote:
    def __init__(self, audio_file, model_id="pyannote/speaker-diarization-3.1", use_auth_token=HUGGING_FACE):
        self.model_id = model_id
        self.pipeline = Pipeline.from_pretrained(
            model_id,
            use_auth_token=use_auth_token)
        self.pipeline.to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
        self.audio_file = Path(audio_file)
        # Load your audio file
        self.waveform, self.sample_rate = torchaudio.load(self.audio_file)
                # Ensure waveform is 2D tensor with shape (channel, time)
        if len(self.waveform.shape) == 1:
            self.waveform = self.waveform.unsqueeze(0)

    def diarization(self, DataSet_builder=False):
        # Perform speaker diarization
        print("\n-------------------------------------------------------------------------------------\n")
        print("\nPerforming speaker diarization on audio file ...")
        diarization = self.pipeline({"waveform": self.waveform, "sample_rate": self.sample_rate})
        # Get the absolute path of the root directory of the project
        root_dir = Path(__file__).resolve().parent.parent

        # Construct the absolute path of the diarization file
        diarization_file_path = root_dir / 'report' / 'log' / 'diarization.rttm'
        # Write the processed transcription to a file
        with diarization_file_path.open('w') as f:
            diarization.write_rttm(f)

        if DataSet_builder:
            # Construct the absolute path of the diarization file
            diarization_file_path = root_dir / 'report' / 'dataset' / f'{self.audio_file.stem}_diarization.rttm'
            # Write the processed transcription to a file
            with diarization_file_path.open('w') as f:
                diarization.write_rttm(f)
        print("\n-------------------------------end---------------------------------------------------\n")