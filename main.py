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
from Utils import preprocess_audio, Process_transcription_and_diarization, generate_report, Whisper, Pyannote
import json

def main(file_path):
    # Preprocess the audio file:
    print(f"\nPreprocessing audio file: {file_path}")
    audio_file = preprocess_audio(file_path)

    # Perform speech recognition and transcription
    whisper = Whisper()
    whisper.transcription(audio_file)

    #Perform speaker diarization
    pyannote = Pyannote(audio_file)
    pyannote.diarization()

    # combine transcription and diarization
    print("\nProcessing, combining transcription and diarization")
    Process_transcription_and_diarization('report/log/transcription.json', 'report/log/diarization.rttm', 'report/log/output.json')

    # Load the JSON data from the file
    with open('report/log/output.json', 'r') as f:
        json_output = json.load(f)
        
    # Generate a report from the JSON data
    print("\nGenerating report from the JSON data")
    generate_report(json_output, 'report/basic_report_output.md')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process an audio file and generate a report.')
    parser.add_argument('file_path', type=str, help='The path to the audio file to process')

    args = parser.parse_args()

    main(args.file_path)