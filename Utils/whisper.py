
import torch
from pathlib import Path
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import json
import librosa
import numpy as np

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
            max_new_tokens=128,
            chunk_length_s=30,
            batch_size=16,
            return_timestamps=True,
            torch_dtype=self.dtype,
            device=self.device)

    def transcription(self, file_path, lang='fr', DataSet_builder=False):
        file_path = Path(file_path)
        # Load the audio file into a numpy array
        audio, _ = librosa.load(file_path, sr=None)
        audio = np.asarray(audio)

        # Perform speech recognition
        print("\n-------------------------------------------------------------------------------------\n")
        Info_message = f"Performing speech recognition and transcription on audio file: {file_path} ..."
        print(Info_message)
        if lang == 'fr':
            transcription = self.pipe(audio, return_timestamps=True, generate_kwargs={"language": "french"})
        elif lang == 'en':
            transcription = self.pipe(audio, return_timestamps=True, generate_kwargs={"language": "english"})


        # Post-process the transcription to ensure that it doesn't exceed 128 tokens
        processed_transcription_chunks = []
        for chunk in transcription["chunks"]:
            chunk_text = chunk["text"]
            chunk_tokens = self.processor.tokenizer(chunk_text, return_tensors='pt', truncation=False)['input_ids'][0]

            # Split long transcription into chunks of no more than 128 tokens
            if len(chunk_tokens) > 128:
                chunk_tokens_chunks = [chunk_tokens[i:i + 128] for i in range(0, len(chunk_tokens), 128)]

                # Interpolate the timestamps for the new chunks
                start_time, end_time = chunk["timestamp"]
                total_duration = end_time - start_time
                token_duration = total_duration / len(chunk_tokens)

                for i, chunk_tokens_chunk in enumerate(chunk_tokens_chunks):
                    chunk_start_time = start_time + i * 128 * token_duration
                    chunk_end_time = min(chunk_start_time + 128 * token_duration, end_time)
                    chunk_text_chunk = self.processor.tokenizer.decode(chunk_tokens_chunk)
                    processed_transcription_chunks.append({"timestamp": [chunk_start_time, chunk_end_time], "text": chunk_text_chunk})
            else:
                processed_transcription_chunks.append(chunk)

        # Get the absolute path of the root directory of the project
        root_dir = Path(__file__).resolve().parent.parent

        # Construct the absolute path of the transcription file
        transcription_file_path = root_dir / 'report' / 'log' / 'transcription.json'

        # Write the processed transcription to a file
        with transcription_file_path.open('w') as f:
            json.dump(processed_transcription_chunks, f)

        if DataSet_builder:
            # Construct the absolute path of the diarization file
            transcription_file_path = root_dir / 'report' / 'dataset' / f'{file_path.stem}_transcription.json'

            # Write the processed transcription to a file
            with transcription_file_path.open('w') as f:
                json.dump(processed_transcription_chunks, f)

        print("\n-------------------------------end---------------------------------------------------\n")