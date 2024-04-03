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
import argparse
import json
import torch
import platform
import time
import csv
import yaml
from Utils import preprocess_audio, Process_transcription_and_diarization, generate_report, Whisper, Pyannote, plot

def main(file_path, mode, llm_model_name):
    # Load the configuration file
    with open('Utils/config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # Check if the provided LLM model name is valid
    if llm_model_name not in config['large_language_models']:
        raise ValueError(f"Invalid LLM model name. Available options are: {', '.join(config['large_language_models'].keys())}")

    # Get the actual model ID
    llm_model_id = config['large_language_models'][llm_model_name]

    start_time = time.time()

    # Preprocess the audio file:
    print(f"\nPreprocessing audio file: {file_path}")
    audio_file = preprocess_audio(file_path)

    # Perform speech recognition and transcription
    whisper = Whisper(config['audio_processing_models']['whisper_model_id'])
    whisper_start_time = time.time()
    whisper.transcription(audio_file)
    whisper_end_time = time.time()
    whisper_time = whisper_end_time - whisper_start_time

    #Perform speaker diarization
    pyannote = Pyannote(audio_file, config['audio_processing_models']['pyannote_model_id'])
    pyannote_start_time = time.time()
    pyannote.diarization()
    pyannote_end_time = time.time()
    pyannote_time = pyannote_end_time - pyannote_start_time

    # combine transcription and diarization
    print("\nProcessing, combining transcription and diarization")
    process_start_time = time.time()
    Process_transcription_and_diarization('report/log/transcription.json', 'report/log/diarization.rttm', 'report/log/output.json', llm_model_name)
    process_end_time = time.time()
    process_time = process_end_time - process_start_time

    # Load the JSON data from the file
    with open('report/log/output.json', 'r') as f:
        json_output = json.load(f)
        
    # Generate a report from the JSON data
    print("\nGenerating report from the JSON data")
    report_start_time = time.time()
    generate_report(json_output, f'report/{llm_model_name}_report_output.md')
    report_end_time = time.time()
    report_time = report_end_time - report_start_time

    end_time = time.time()
    total_time = end_time - start_time

    if mode == 'dev':
        device = 'GPU' if torch.cuda.is_available() else 'CPU'
        device_info = platform.uname()
        log_entry_label = f"{config['audio_processing_models']['whisper_model_id']},{config['audio_processing_models']['pyannote_model_id']},{llm_model_id}"
        log_audio_file = f"audio_file: {file_path}"
        with open('logs/benchmark.csv', 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([log_entry_label, log_audio_file , device, device_info, whisper_time, pyannote_time, process_time, report_time, total_time])
        plot('logs/benchmark.csv')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process an audio file and generate a report.')
    parser.add_argument('file_path', type=str, help='The path to the audio file to process')
    parser.add_argument('--mode', type=str, default='prod', help='The mode to run the script in (dev or prod)')
    parser.add_argument('--llm', type=str, required=True, help='The Large Language Model to use(gpt, gemma, mistral)')

    args = parser.parse_args()

    main(args.file_path, args.mode, args.llm)