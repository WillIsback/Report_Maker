# text_preprocess.py
"""
@brief This module contains the functions to preprocess the text data before the report generation model can process it.
"""
from .mistral import Mistral
from .gemma import Gemma
from .gpt import GPT
from .bert import Camembert, BART
import json
import spacy
import datetime
import yaml
from tqdm import tqdm


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

MAX_TOKENS = 512
MARGINS = 64
TOKENS_PADDING = 0 + MARGINS
TOKENS_CHUNK = 0 + MARGINS

def process_dialogue(all_sentences, llm):
    sections = []
    speaker_transcription = ''

    for sentence in tqdm(all_sentences, desc="Processing dialogue"):
        new_transcription = f"{speaker_transcription}\n{sentence['speaker']}: {sentence['text']}"
        encoded_length = llm.tokenlen(new_transcription)
        if encoded_length > MAX_TOKENS - TOKENS_PADDING:
            sub_summary = llm.request(speaker_transcription)
            sections.append({'summary': sub_summary})
            speaker_transcription = f"speaker: {sentence['speaker']}: {sentence['text']}"
        else:
            speaker_transcription = new_transcription

    if speaker_transcription:
        sub_summary = llm.request(speaker_transcription)
        sections.append({'summary': sub_summary})
        
    return sections 

def process_conclusion(overall_transcription, llm):
    overall_summary = ''
    chunks = ''
    for section in tqdm(overall_transcription, desc="Processing conclusion"):
        new_summary = f"{chunks} \n {section['summary']}"
        encoded_length = llm.tokenlen(new_summary)
        if encoded_length > MAX_TOKENS - TOKENS_CHUNK:
            overall_summary += llm.summarize(chunks)
            chunks = section['summary']
        else:
            chunks = new_summary
    if chunks:
        overall_summary += llm.summarize(chunks)
    return overall_summary

def Process_transcription_and_diarization(transcription_file, rttm_file, output_file, llm_model_name):
    sentences = load_transcription(transcription_file)
    diarization = load_diarization(rttm_file)
    nlp = initialize_nlp()
    
    # Load the configuration file and initialize the LLM model
    with open('Utils/config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    if llm_model_name == 'gpt':
        llm = GPT(config['large_language_models']['gpt'])
    elif llm_model_name == 'mistral':
        llm = Mistral(config['large_language_models']['mistral'])
    elif llm_model_name == 'gemma':
        llm = Gemma(config['large_language_models']['gemma'])
    elif llm_model_name == 'camembert':
        llm = Camembert(config['large_language_models']['camembert'])
    elif llm_model_name == 'bart':
        llm = BART(config['large_language_models']['bart'])
    else:
        llm = Gemma(config['large_language_models']['gemma'])
        
    all_sentences = [{'text': sentence, 'timestamp': timestamp, 'speaker': find_speaker(timestamp, diarization)} 
                     for sentence, timestamp in tqdm(sentences, desc="Processing all sentences") if find_speaker(timestamp, diarization) is not None]

    all_sentences.sort(key=lambda s: s['timestamp'])

    sections = process_dialogue(all_sentences, llm)
    overall_summary = process_conclusion(sections, llm)

    output = {'conclusion': overall_summary, 'sections': sections, 'details': all_sentences}

    with open(output_file, 'w') as f:
        json.dump(output, f)
        
        
def generate_report(json_output, markdown_file):
    with open(json_output, 'r') as f:
        json_output = json.load(f)
        
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
        f.write(f"- {json_output['conclusion']}\n")