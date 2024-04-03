# text_preprocess.py
"""
@brief This module contains the functions to preprocess the text data before the report generation model can process it.
"""
from tracemalloc import start
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
from .mistral import Mistral
from .gemma import Gemma
import json
import spacy
import datetime
from collections import defaultdict

from tqdm import tqdm


def summarize_text(text, max_length):
    print(f"\nGenerating summary for the following text: \n")
    model_name = "facebook/mbart-large-50-many-to-many-mmt"
    tokenizer = MBart50TokenizerFast.from_pretrained(model_name)
    model = MBartForConditionalGeneration.from_pretrained(model_name)

    # encode the text
    inputs = tokenizer(text, return_tensors="pt", max_length=max_length, truncation=True, padding='longest')

    # generate summary
    summary_ids = model.generate(inputs['input_ids'], num_beams=4, max_length=max_length, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    return summary

def load_transcription(transcription_file):
    with open(transcription_file, 'r') as f:
        data = f.read()
        try:
            data = json.loads(data)  # parse the JSON data
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON: {e}")
            return []
    sentences = [(item['text'], item['timestamp']) for item in data if 'text' in item and 'timestamp' in item]
    return sentences

def load_diarization(rttm_file):
    diarization = []
    with open(rttm_file, 'r') as f:
        for line in f:
            fields = line.strip().split()
            speaker = fields[7]
            start_timestamp = float(fields[3])
            end_timestamp = start_timestamp + float(fields[4])
            diarization.append([speaker, start_timestamp, end_timestamp])
    return diarization

def initialize_nlp():
    nlp = spacy.load('fr_core_news_sm')
    return nlp


def process_paragraph(paragraph, current_speaker, nlp, timestamp):
    # print(f"\nProcessing paragraph\n")
    doc = nlp(paragraph)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    return {
        'text': paragraph,
        'speaker': current_speaker,
        'entities': entities,
        'timestamp': timestamp
    }
    
def find_speaker(timestamp, diarization):
    start_time, end_time = timestamp
    if start_time is None or end_time is None:
        return None  # Return None if either timestamp is None
    for speaker, start_timestamp, end_timestamp in diarization:
        if start_timestamp <= start_time <= end_timestamp or start_timestamp <= end_time <= end_timestamp:
            return speaker
    return None

MAX_TOKENS = 2048
TOKENS_PADDING = 256
TOKENS_CHUNK = 512

def process_dialogue(all_sentences, gemma):
    sections = []
    overall_transcription = ''
    speaker_transcription = ''

    for sentence in all_sentences:
        new_transcription = f"{speaker_transcription}\n{sentence['speaker']}: {sentence['text']}"
        inputs = (gemma.encoder(new_transcription))
        encoded_length = len(inputs['input_ids'][0])
        # print(f"Encoded length: {encoded_length}")
        if encoded_length > MAX_TOKENS - TOKENS_PADDING:
            # print(f"\nGenerating summary for transcriptions: {speaker_transcription}")
            sub_summary = gemma.request(speaker_transcription)
            sections.append({'summary': sub_summary})
            overall_transcription += ' ' + sub_summary
            speaker_transcription = f"speaker: {sentence['speaker']}: {sentence['text']}"
        else:
            speaker_transcription = new_transcription

    if speaker_transcription:
        # print(f"\nGenerating summary for transcriptions: {speaker_transcription}")
        sub_summary = gemma.request(speaker_transcription)
        sections.append({'summary': sub_summary})
        overall_transcription += ' ' + sub_summary

    return sections, overall_transcription

def process_conclusion(overall_transcription, gemma):
    overall_summary = ''
    chunks = [overall_transcription]

    # Split chunks until each chunk is small enough
    while any(len(gemma.encoder(chunk)['input_ids'][0]) > MAX_TOKENS - TOKENS_CHUNK for chunk in chunks):
        new_chunks = []
        for chunk in chunks:
            if len(gemma.encoder(chunk)['input_ids'][0]) > MAX_TOKENS - TOKENS_CHUNK:
                half = len(chunk) // 2
                new_chunks.append(chunk[:half])
                new_chunks.append(chunk[half:])
            else:
                new_chunks.append(chunk)
        chunks = new_chunks

    # Summarize each chunk and concatenate the summaries
    for chunk in chunks:
        overall_summary += ' ' + gemma.summarize(chunk)

    return overall_summary.strip()

def Process_transcription_and_diarization(transcription_file, rttm_file, output_file):
    sentences = load_transcription(transcription_file)
    diarization = load_diarization(rttm_file)
    nlp = initialize_nlp()
    gemma = Gemma()

    all_sentences = [{'text': sentence, 'timestamp': timestamp, 'speaker': find_speaker(timestamp, diarization)} 
                     for sentence, timestamp in tqdm(sentences, desc="Processing sentences: ...") if find_speaker(timestamp, diarization) is not None]

    all_sentences.sort(key=lambda s: s['timestamp'])

    sections, overall_transcription = process_dialogue(all_sentences, gemma)
    overall_summary = process_conclusion(overall_transcription, gemma)

    output = {'conclusion': overall_summary, 'sections': sections, 'details': all_sentences}

    with open(output_file, 'w') as f:
        json.dump(output, f)
        
        
def generate_report(json_output, markdown_file):
    with open(markdown_file, 'w') as f:
        # Write the title
        date = datetime.date.today().strftime("%d/%m/%Y")
        f.write(f"# Compte-rendu réunion {date}\n\n")

        # Write the table of contents
        f.write("## Table des matières\n\n")
        f.write("1. [Transcription de la réunion](#Transcription-de-la-réunion)\n")
        f.write("2. [Résumé de la réunion](#Résumé-de-la-réunion)\n")
        f.write("3. [Conclusion](#conclusion)\n\n")

        # Write the complete transcription section
        f.write("## Transcription de la réunion\n\n")
        f.write("<details>\n<summary>View Full Transcription</summary>\n\n")
        for sentence in json_output['details']:
            f.write(f"{sentence['timestamp']} - {sentence['speaker']}: {sentence['text']} <br> \n\n")
        f.write("</details>\n\n")

        # Write the content summary section
        f.write("## Résumé de la réunion\n\n")
        for section in json_output['sections']:
            f.write(f"{section['summary']} <br> \n\n")

        # Write the conclusion
        f.write("## Conclusion\n\n")
        for conclusion in json_output['conclusion']:
            f.write(f"- {conclusion}\n\n")