# main.py

"""
@brief This is the main module of the AI Report Maker program. It is a program that use automatic speech recognition to transcribe the record of a meeting and generate a report out of it.
    It first use whisper large v3 model to transcribe the audio file, then use speaker diarization to identify the speakers in the audio file.
    Then it combine the transcription and diarization to generate a full annoted transcription with timestamps speakers and text.
    Process the dialogue to generate sub-summary using a LLM of the user choice (openai-GPT3.5, GEMMA, MISTRAL).
    Process the sub-summary in the same LLM to generate a conclusion.
    
    the actual features are:
    - Language supported : french only at the moment
    - Automatic Speech Recognition with timestamps
    - Speaker Diarization with timestamps
    - Generate an annotated transcription with timestamps, speakers and text
    - Generate sub-summary for each speaker
    - Generate a conclusion for the meeting
    - Generate a report in markdown format with the annotated transcription, sub-summary and conclusion
    
    future features are:
    - Multilanguage support
    - Add more LLM to generate sub-summary and conclusion
    - Add a GUI to interact with the program
    - Automatic speaker naming and identification (could be based on behavior such as name called question/answer, if the meeting is drived by someone giving the right talk to a speaker by calling their name, etc.)
    - Extraction of important moments of the meeting from the audio file (could be based on speaker intonation, volume, etc.)
    - Extration of Topics discussed in the meeting
    - Sentiment analysis of the meeting
    - Generate a report in different format (pdf, docx, etc.)
    

"""
from dotenv import load_dotenv
import os
from pyannote.audio import Pipeline
import torch
import torchaudio
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from Utils import preprocess_audio, Process_transcription_and_diarization, generate_report
import json
from openai import OpenAI

load_dotenv()

# Get the Hugging Face API key from the environment variables
HUGGING_FACE = os.getenv('HUGGING_FACE')


# Get the OpenAI API key from the environment variables
client = OpenAI(api_key=os.getenv('OPENAI_KEY'))

if torch.cuda.is_available():
    device = "cuda:0"
    print("Using GPU")
else:
    device = torch.device("cpu")
    print("Using CPU")
    
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model_id = "openai/whisper-large-v3"

model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
)
model.to(device)


processor = AutoProcessor.from_pretrained(model_id)

# Automatic Speech Recognition whisper large v3
pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    max_new_tokens=400,
    chunk_length_s=30,
    batch_size=32,
    return_timestamps=True,
    torch_dtype=torch_dtype,
    device=device,
)

# Preprocess the audio file:
file_path = 'audio/interview_5mn.mp3'
print(f"\nPreprocessing audio file: {file_path}")
audio_file = preprocess_audio(file_path)

# Load your audio file
waveform, sample_rate = torchaudio.load(audio_file)
waveform = waveform.squeeze(dim=0)


# Perform speech recognition
print(f"\nPerforming speech recognition and transcription on audio file: {audio_file}")
transcription = pipe(file_path, return_timestamps=True, generate_kwargs={"language": "french"})

# Write the transcription to a file
with open('report/log/transcription.json', 'w') as f:
    json.dump(transcription["chunks"], f)


pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1",
    use_auth_token=HUGGING_FACE)

pipeline.to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
waveform, sample_rate = torchaudio.load(audio_file)
            
# Pyannote speaker diarization
print("\nPerforming speaker diarization on audio file: {file_path}")
diarization = pipeline({"waveform": waveform, "sample_rate": sample_rate})

# Save diarization result to RTTM file
with open('report/log/diarization.rttm', 'w') as f:
    diarization.write_rttm(f)
    
# combine transcription and diarization
print("\nProcessing, combining transcription and diarization")
Process_transcription_and_diarization('report/log/transcription.json', 'report/log/diarization.rttm', 'report/log/output.json')

# Load the JSON data from the file
with open('report/log/output.json', 'r') as f:
    json_output = json.load(f)
    
# Generate a report from the JSON data
print("\nGenerating report from the JSON data")
generate_report(json_output, 'report/basic_report_output.md')


    
    
# # Split the data into chunks
# max_tokens = 4096
# chunk_size = max_tokens if len(data) <= max_tokens else max_tokens // 2
# chunks = [data[i:i + chunk_size] for i in range(0, len(data), chunk_size)]

# # Define the system message
# system_msg = 'You are a helpful assistant who generate a markdown formated with style meeting report in french.'

# # Initialize an empty string to hold the final report
# report_md = ""

# # Get the current date
# current_date = datetime.datetime.now().strftime("%d-%m-%Y")

# # Generate a summary for each chunk
# for chunk in chunks:
#     # Format the chunk into a prompt
#     prompt = f"Genere un rapport et son sommaire des points clefs avec les elements suivants pour la date {current_date}:\n\n"
#     for item in chunk:
#         prompt += f"Le sujet de la réunion\n"
#         prompt += f"à la fin du rapport une conclusion de la réunion avec les points clés.\n"
#         prompt += f"dans le rapport résumé de chaque intervenant {item['speaker']}**: {item['text']}\n"

        

#     # Use the language model to generate the summary
#     response = client.chat.completions.create(model="gpt-3.5-turbo",
#                                               messages=[{"role": "system", "content": system_msg},
#                                                       {"role": "user", "content": prompt}],
#                                               temperature=0)
#     # Extract the generated text from the response
#     generated_text = response.choices[0].message.content

#     # Add the generated text to the final report if it's not None
#     if generated_text is not None:
#         report_md += generated_text

# # Write the final report to a Markdown file
# with open('report_output.md', 'w') as f:
#     f.write(report_md)