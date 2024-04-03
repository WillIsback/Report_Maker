
import torch

from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import json

class Whisper:
    def __init__(self, model_id="openai/whisper-large-v3", dtype=torch.float16):
        self.model_id = model_id
        self.dtype = dtype
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            use_safetensors=True
        )
        self.device = "cuda:0" if torch.cuda.is_available() else torch.device("cpu")
        self.model.to(self.device)
        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=self.model,
            tokenizer=self.processor.tokenizer,
            feature_extractor=self.processor.feature_extractor,
            max_new_tokens=400,
            chunk_length_s=30,
            batch_size=32,
            return_timestamps=True,
            torch_dtype=self.dtype,
            device=self.device)
        
    def transcription(self, file_path):
        # Perform speech recognition
        print(f"\nPerforming speech recognition and transcription on audio file ...")
        transcription = self.pipe(file_path, return_timestamps=True, generate_kwargs={"language": "french"})

        # Write the transcription to a file
        with open('report/log/transcription.json', 'w') as f:
            json.dump(transcription["chunks"], f)