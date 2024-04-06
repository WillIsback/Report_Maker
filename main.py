"""
@file: main.py
@author: WillIsback
@date: 2024-04-03
@brief This is the main module of the AI Report Maker program. It is a program that use automatic speech recognition to transcribe the record of a meeting and generate a report out of it.
    It first use whisper large v3 model to transcribe the audio file, then use speaker diarization to identify the speakers in the audio file.
    Then it combine the transcription and diarization to generate a full annoted transcription with timestamps speakers and text.
    Process the dialogue to generate sub-summary using a LLM of the user choice (openai-GPT3.5, GEMMA, MISTRAL).
    Process the sub-summary in the same LLM to generate a conclusion.
    Finally generate a report in markdown format.
"""
import os
import argparse
import torch
import platform
import time
import yaml
from Utils import preprocess_audio, get_file_hash, Process_text, generate_report, Whisper, Pyannote, plot
import pandas as pd

class ReportMaker:
    def __init__(self, file_path, mode, llm_model_name, lang='fr'):
        self.file_path = file_path
        self.mode = mode
        self.llm_model_name = llm_model_name
        self.lang = lang
        # Load the configuration file
        with open('Utils/config/config.yaml', 'r') as f:
            self.config = yaml.safe_load(f)

        self.llm_model_id = self.config['large_language_models'][f'{self.llm_model_name}']
        self.device = 'GPU' if torch.cuda.is_available() else 'CPU'
        if self.device == 'GPU':
            self.gpu_info = torch.cuda.get_device_properties(0)
            self.device_info = f"{self.gpu_info}, DEVICE: {platform.uname()}"
        else:
            self.gpu_info = torch.cuda.get_device_properties(0)
            self.device_info = f"DEVICE: {platform.uname()}"

        self.ASR_model_id = self.config['audio_processing_models']['whisper_model_id']
        self.diarization_model_id = self.config['audio_processing_models']['pyannote_model_id']

        self.whisper_time = 0
        self.pyannote_time = 0
        self.process_time = 0
        self.total_time = 0
        self.index = 0

        filename_with_ext = os.path.basename(self.file_path)
        self.filename, _ = os.path.splitext(filename_with_ext)
        self.audio_file = preprocess_audio(self.file_path)
        self.transcription_json = 'report/log/transcription.json'
        self.diarization_rttm = 'report/log/diarization.rttm'
        self.output_json = 'report/log/output.json'
        self.log_audio_file = f"audio_file: {self.filename}"
        self.log_file_path = 'logs/benchmark.csv'
        self.current_file_hash = get_file_hash(self.audio_file)

    def check_audio_file_change(self):
        if os.path.exists(self.log_file_path):
            df = pd.read_csv(self.log_file_path)
            self.index = df.iloc[-1, 0] + 1  # Increment the index
            if df.iloc[-1, 1] == self.current_file_hash:  # If the file hash is the same
                return False  # No change in the file
            else:  # If the file hash is different
                return True  # File has changed
        else:  # If the file does not exist
            self.index  = 0
            return True  # File has changed

    def log(self):
        if self.mode == 'dev':
            print(f'Index: {self.index}')  # Print the index

            # Prepare the new data
            log_data = pd.DataFrame({
                'index': [self.index],
                'file_hash': [self.current_file_hash],
                'whisper_model_id' : [self.ASR_model_id],
                'pyannote_model_id' : [self.diarization_model_id],
                'llm_model_id' : [self.llm_model_id],
                'log_audio_file': [self.log_audio_file],
                'device': [self.device],
                'device_info': [str(self.device_info)],
                'whisper_time': [self.whisper_time],
                'pyannote_time': [self.pyannote_time],
                'process_time': [self.process_time],
                'total_time': [self.total_time]
            })


            log_data.to_csv(self.log_file_path, mode='a', header=not os.path.exists(self.log_file_path), index=False)

            plot(self.log_file_path)

    def preprocess_audio(self):
        # Preprocess the audio file:
        print(f"\nPreprocessing audio file: {self.file_path}")
        return preprocess_audio(self.file_path)

    def run_ASR(self):
        # Perform speech recognition and transcription
        whisper = Whisper(self.ASR_model_id)
        whisper_start_time = time.time()
        if self.lang == 'fr':
            whisper.transcription(self.audio_file, lang='fr')
        elif self.lang == 'en':
            whisper.transcription(self.audio_file, lang='en')

        whisper_end_time = time.time()
        self.whisper_time = whisper_end_time - whisper_start_time

    def run_Diarization(self):
        #Perform speaker diarization
        pyannote = Pyannote(self.audio_file, self.diarization_model_id)
        pyannote_start_time = time.time()
        pyannote.diarization()
        pyannote_end_time = time.time()
        self.pyannote_time = pyannote_end_time - pyannote_start_time

    def run_preprocess_text(self):
        # combine transcription and diarization
        print("\nProcessing, combining transcription and diarization")
        process_start_time = time.time()
        Process_text(self.transcription_json, self.diarization_rttm, self.output_json, self.llm_model_name)
        process_end_time = time.time()
        self.process_time = process_end_time - process_start_time

    def generate_report(self):
        # Generate the report
        markdown_files = generate_report(self.output_json, f'report/markdown/{self.llm_model_name}-{self.filename}_report_output_{self.index}.md')
        return markdown_files
    def PrintHeader(self):
        print("\033[1;34m\n-------------------------------------------------------------------------------------\n\033[0m")
        print("""
        \033[1;32m
        ██████╗ ███████╗██████╗  ██████╗ ██████╗ ████████╗    ███╗   ███╗ █████╗ ██╗  ██╗███████╗██████╗
        ██╔══██╗██╔════╝██╔══██╗██╔═══██╗██╔══██╗╚══██╔══╝    ████╗ ████║██╔══██╗██║ ██╔╝██╔════╝██╔══██╗
        ██████╔╝█████╗  ██████╔╝██║   ██║██████╔╝   ██║       ██╔████╔██║███████║█████╔╝ █████╗  ██████╔╝
        ██╔══██╗██╔══╝  ██╔═══╝ ██║   ██║██╔══██╗   ██║       ██║╚██╔╝██║██╔══██║██╔═██╗ ██╔══╝  ██╔══██╗
        ██║  ██║███████╗██║     ╚██████╔╝██║  ██║   ██║       ██║ ╚═╝ ██║██║  ██║██║  ██╗███████╗██║  ██║
        ╚═╝  ╚═╝╚══════╝╚═╝      ╚═════╝ ╚═╝  ╚═╝   ╚═╝       ╚═╝     ╚═╝╚═╝  ╚═╝╚═╝  ╚═╝╚══════╝╚═╝  ╚
        \033[0m
        """)
        print(f"\033[1;32mRunning AI Report Maker in \033[1;34m{self.mode}\033[1;32m mode with \033[1;34m{self.device}\033[1;32m device to process the record \033[1;34m{self.filename}\033[1;32m with \033[1;34m{self.llm_model_id}\033[1;32m Large Language Model and \033[1;34m{self.ASR_model_id}\033[1;32m ASR model and \033[1;34m{self.diarization_model_id}\033[1;32m Diarization model.\033[0m")
        print("\033[1;34m\n-------------------------------------------------------------------------------------\n\n\033[0m")

    def run(self):
        self.PrintHeader()
        start_time= time.time()
        if self.check_audio_file_change():
            self.run_ASR()
            self.run_Diarization()
        self.run_preprocess_text()
        end_time = time.time()
        self.total_time = end_time - start_time

        print(f"\nProcessing time: \033[1;34m{self.total_time}\033[1;32m seconds\n\033[0m")


        if self.mode == 'dev':
            # Log the results
            self.log()
            # Generate the report
            markdown_files = self.generate_report()
            for file in markdown_files:
                print(f"\033[1;32m\nReport generated: \033[1;34m{file}\033[1;32m\n\033[0m")

        elif self.mode == 'prod':
            # Generate the report
            markdown_files = self.generate_report()
            for file in markdown_files:
                print(f"\033[1;32m\nReport generated: \033[1;34m{file}\033[1;32m\n\033[0m")

        print("\n------------------------------------end run----------------------------------------------\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process an audio file and generate a report.')
    parser.add_argument('file_path', type=str, help='The path to the audio file to process')
    parser.add_argument('--mode', type=str, default='prod', help='The mode to run the script in (dev or prod)')
    parser.add_argument('--llm', type=str, required=True, help='The Large Language Model to use(gpt, gemma-7b, gemma-2b)')
    parser.add_argument('--lang', type=str, default='fr', help='The language of the audio file (fr or en)')

    args = parser.parse_args()

    report_maker = ReportMaker(args.file_path, args.mode, args.llm)
    report_maker.run()