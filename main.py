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
import json
from pdb import run
import torch
import platform
import time
import csv
import yaml
from Utils import preprocess_audio, Process_transcription_and_diarization, generate_report, Whisper, Pyannote, plot

class ReportMaker:
    def __init__(self, file_path, mode, llm_model_name):
        self.file_path = file_path
        self.mode = mode
        self.llm_model_name = llm_model_name
        # Load the configuration file
        with open('Utils/config/config.yaml', 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.llm_model_id = self.config['large_language_models'][self.llm_model_name]
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
        self.log_entry_label = f"{self.ASR_model_id},{self.diarization_model_id},{self.llm_model_id}"
        self.log_audio_file = f"audio_file: {self.filename}"
        self.log_file_path = 'logs/benchmark.csv'
        
    def preprocess_audio(self):
        # Preprocess the audio file:
        print(f"\nPreprocessing audio file: {self.file_path}")
        return preprocess_audio(self.file_path)
    
    def run_ASR(self):
        # Perform speech recognition and transcription
        whisper = Whisper(self.config['audio_processing_models']['whisper_model_id'])
        whisper_start_time = time.time()
        whisper.transcription(self.audio_file)
        whisper_end_time = time.time()
        self.whisper_time = whisper_end_time - whisper_start_time
        
    def run_Diarization(self):
        #Perform speaker diarization
        pyannote = Pyannote(self.audio_file, self.config['audio_processing_models']['pyannote_model_id'])
        pyannote_start_time = time.time()
        pyannote.diarization()
        pyannote_end_time = time.time()
        self.pyannote_time = pyannote_end_time - pyannote_start_time
        
    def run_preprocess_text(self):
        # combine transcription and diarization
        print("\nProcessing, combining transcription and diarization")
        process_start_time = time.time()
        Process_transcription_and_diarization(self.transcription_json, self.diarization_rttm, self.output_json, self.llm_model_name)
        process_end_time = time.time()
        self.process_time = process_end_time - process_start_time
        
    def generate_report(self):
        # Generate the report
        generate_report(self.output_json, f'report/{self.llm_model_name}-{self.filename}_report_output_{self.index}.md')
    
    def run(self):
        start_time= time.time()
        self.run_ASR()
        self.run_Diarization()
        self.run_preprocess_text()
        end_time = time.time()
        self.total_time = end_time - start_time

        print(f"\nProcessing time: {self.total_time} seconds\n")
        

        if self.mode == 'dev':
            # Determine the index of the next entry
            if os.path.exists(self.log_file_path):
                with open(self.log_file_path, 'r') as f:
                    reader = csv.reader(f)
                    self.index = max(sum(1 for row in reader) - 1, 0)
            else:
                self.index = 0
                
            # Write the new entry
            with open(self.log_file_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([self.index, self.log_entry_label, self.log_audio_file , self.device, str(self.device_info), self.whisper_time, self.pyannote_time, self.process_time, self.total_time])

                # Generate the report
                self.generate_report()
                
                plot(self.log_file_path)
            
        elif self.mode == 'prod':         
            # Generate the report
            self.generate_report()
            

            
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process an audio file and generate a report.')
    parser.add_argument('file_path', type=str, help='The path to the audio file to process')
    parser.add_argument('--mode', type=str, default='prod', help='The mode to run the script in (dev or prod)')
    parser.add_argument('--llm', type=str, required=True, help='The Large Language Model to use(gpt, gemma, mistral, bert)')

    args = parser.parse_args()

    report_maker = ReportMaker(args.file_path, args.mode, args.llm)
    report_maker.run()