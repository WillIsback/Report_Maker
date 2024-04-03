from dotenv import load_dotenv
import os
import torch
import torchaudio
from pyannote.audio import Pipeline


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
        # Load your audio file
        self.waveform, self.sample_rate = torchaudio.load(audio_file)
                # Ensure waveform is 2D tensor with shape (channel, time)
        if len(self.waveform.shape) == 1:
            self.waveform = self.waveform.unsqueeze(0)
        
    def diarization(self):
        # Perform speaker diarization
        print("\nPerforming speaker diarization on audio file ...")
        diarization = self.pipeline({"waveform": self.waveform, "sample_rate": self.sample_rate})
        
        # Save the diarization result to an RTTM file
        with open('report/log/diarization.rttm', 'w') as f:
            diarization.write_rttm(f)
