# text_preprocess.py
"""
@brief This module contains the functions to preprocess the text data before the report generation model can process it.
"""
from .anr import ANR
from .mistral import Mistral
from .gemma import Gemma
from .gpt import GPT
from .bert import Camembert, BART
import json
import datetime
import yaml
from tqdm import tqdm


MAX_TOKENS = 512
MARGINS = 64
TOKENS_PADDING = 0 + MARGINS
TOKENS_CHUNK = 0 + MARGINS


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

def find_speaker(timestamp, diarization):
    start_time, end_time = timestamp
    if start_time is None or end_time is None:
        return None  # Return None if either timestamp is None
    for speaker, start_timestamp, end_timestamp in diarization:
        if start_timestamp <= start_time <= end_timestamp or start_timestamp <= end_time <= end_timestamp:
            return speaker
    return None

def process_all_sentences(sentences, diarization):
    all_sentences = [{'index': i, 'sentence': {'text': sentence, 'timestamp': timestamp, 'speaker': find_speaker(timestamp, diarization)}}
                for i, (sentence, timestamp) in enumerate(tqdm(sentences, desc="Processing all sentences")) if find_speaker(timestamp, diarization) is not None]
    all_sentences.sort(key=lambda s: s['sentence']['timestamp'])
    return all_sentences

def process_paragraph(all_sentences, llm):
    paragraphs = []
    current_paragraph = {'id': 0, 'speaker': [all_sentences[0]['sentence']['speaker']], 'text': all_sentences[0]['sentence']['speaker'] + ': ' + all_sentences[0]['sentence']['text'], 'tokens_length': llm.tokenlen(all_sentences[0]['sentence']['text']), 'timestamp': all_sentences[0]['sentence']['timestamp']}

    for sentence in tqdm(all_sentences[1:], desc="Processing Paragraph"):
        sentence_tokens = llm.tokenlen(sentence['sentence']['text'])

        if current_paragraph['tokens_length'] + sentence_tokens <= 400:
            # If adding the sentence won't exceed the limit, add it to the paragraph
            if sentence['sentence']['speaker'] == current_paragraph['speaker'][-1]:
                # If the speaker is the same, just append the text
                current_paragraph['text'] += ' ' + sentence['sentence']['text']
                current_paragraph['timestamp'][1] = sentence['sentence']['timestamp'][1]
                current_paragraph['tokens_length'] += sentence_tokens
            else:
                # If the speaker changes, add new speaker identification to the text and update the speaker
                current_paragraph['text'] += '\n' + sentence['sentence']['speaker'] + ': ' + sentence['sentence']['text']
                current_paragraph['speaker'].append(sentence['sentence']['speaker'])
                current_paragraph['tokens_length'] += sentence_tokens
        else:
            # If adding the sentence would exceed the limit, save the current paragraph and start a new one
            paragraphs.append({'paragraph' : current_paragraph})
            current_paragraph = {'id': len(paragraphs), 'speaker': [sentence['sentence']['speaker']], 'text': sentence['sentence']['text'], 'tokens_length': sentence_tokens, 'timestamp': sentence['sentence']['timestamp']}

    # Don't forget to add the last paragraph
    if current_paragraph['text']:
        paragraphs.append({'paragraph' : current_paragraph})

    return paragraphs

def process_dialogue(paragraphs, llm):
    sections = []
    for i, paragraph in enumerate(tqdm(paragraphs, desc="Processing dialogue")):
        try:
            sub_summary = llm.request(paragraph['paragraph']['text'])
            sections.append({'summary': sub_summary})
        except IndexError:
            print(f"IndexError occurred at index {i} with paragraph")
    return sections

def process_conclusion(overall_transcription, llm):
    overall_summary = ''
    chunks = ''
    for section in tqdm(overall_transcription, desc="Processing conclusion"):
        new_summary = f"{chunks} \n {section['summary']}"
        encoded_length = llm.tokenlen(new_summary)
        if encoded_length > MAX_TOKENS - TOKENS_CHUNK:
            summary = llm.summarize(str(chunks))
            if summary is not None:
                overall_summary += summary
            chunks = section['summary']
        else:
            chunks = new_summary
    if chunks:
        summary = llm.summarize(str(chunks))
        if summary is not None:
            overall_summary += summary
    return overall_summary

def Process_transcription_and_diarization(transcription_file, rttm_file, output_file, llm_model_name):
    print("\n-------------------------------------------------------------------------------------\n")
    print("\nPerforming text process ...")
    sentences = load_transcription(transcription_file)
    diarization = load_diarization(rttm_file)

    # Load the configuration file and initialize the LLM model
    with open('Utils/config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    if llm_model_name == 'gpt':
        llm = GPT(config['large_language_models']['gpt'])
    elif llm_model_name == 'mistral':
        llm = Mistral(config['large_language_models']['mistral'])
    elif llm_model_name == 'gemma-7b':
        llm = Gemma(config['large_language_models']['gemma-7b'])
    elif llm_model_name == 'gemma-2b':
        llm = Gemma(config['large_language_models']['gemma-2b'])
    elif llm_model_name == 'camembert':
        llm = Camembert(config['large_language_models']['camembert'])
    elif llm_model_name == 'bart':
        llm = BART(config['large_language_models']['bart'])
    else:
        llm = Gemma(config['large_language_models']['gemma-2b'])

    all_sentences = process_all_sentences(sentences, diarization)

    anr = ANR(all_sentences)
    anr.Full()
    speaker_names = anr.GetSpeakerName()

    paragraphs = process_paragraph(all_sentences, llm)

    sections = process_dialogue(paragraphs, llm)

    overall_summary = process_conclusion(sections, llm)

    output = {'conclusion': overall_summary, 'sections': sections, 'details': paragraphs, 'speaker_names': speaker_names}

    print("\n-------------------------------end---------------------------------------------------\n")
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
            f.write(f"Timestamp : {sentence['paragraph']['timestamp']} / {sentence['paragraph']['speaker']}: <br> {sentence['paragraph']['text']} <br> \n\n")
        f.write("</details>\n\n")

        # Write the content summary section
        f.write("## Résumé de la réunion\n\n")
        f.write("### Noms des participants\n\n")
        for speaker_id, name in json_output['speaker_names'].items():
            f.write(f"-{speaker_id}: {name}\n")
        f.write("### Sections\n\n")
        for section in json_output['sections']:
            f.write(f"{section['summary']} <br> \n\n")

        # Write the conclusion
        f.write("## Conclusion\n\n")
        f.write(f"- {json_output['conclusion']}\n")