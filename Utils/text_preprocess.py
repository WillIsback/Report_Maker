# text_preprocess.py
from .anr import ANR
from pathlib import Path
from .gemma import Gemma, FastGemma
from .gpt import GPT
from .summarize_dataset_builder import summarize_dataset_builder
import json
import yaml
from tqdm import tqdm

# Get the absolute path of the root directory of the project
root_dir = Path(__file__).resolve().parent.parent

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

    all_sentences_file_path = root_dir / 'report' / 'log' / 'all_sentences.json'
    with open(all_sentences_file_path, 'w') as f:
        json.dump(all_sentences, f, indent=4)
    return all_sentences

def process_paragraph(key_sentences, llm, max_tokens, max_token_output, overlap=3):
    paragraphs = []
    sentence_buffer = []
    current_paragraph = {'id': 0, 'speaker': [key_sentences[0]['sentence']['speaker']], 'text': key_sentences[0]['sentence']['speaker'] + ': ' + key_sentences[0]['sentence']['text'], 'tokens_length': llm.tokenlen(key_sentences[0]['sentence']['text']), 'timestamp': key_sentences[0]['sentence']['timestamp'],'named_entities': key_sentences[0]['sentence']['named_entities']}
    sentence_buffer.append(key_sentences[0])

    for sentence in tqdm(key_sentences[1:], desc="Processing Paragraph"):
        sentence_tokens = llm.tokenlen(sentence['sentence']['text'])

        if current_paragraph['tokens_length'] + sentence_tokens <= (max_tokens-max_token_output+150): # 50 is a buffer for the instruction to summarize tokens
            # If adding the sentence won't exceed the limit, add it to the paragraph
            if sentence['sentence']['speaker'] == current_paragraph['speaker'][-1]:
                # If the speaker is the same, just append the text
                current_paragraph['text'] += ' ' + sentence['sentence']['text']
                current_paragraph['timestamp'][1] = sentence['sentence']['timestamp'][1]
                current_paragraph['tokens_length'] += sentence_tokens
                current_paragraph['named_entities'] += sentence['sentence']['named_entities']
            else:
                # If the speaker changes, add new speaker identification to the text and update the speaker
                current_paragraph['text'] += '\n\n' + sentence['sentence']['speaker'] + ': ' + sentence['sentence']['text']
                current_paragraph['speaker'].append(sentence['sentence']['speaker'])
                current_paragraph['tokens_length'] += sentence_tokens
                current_paragraph['named_entities'] += sentence['sentence']['named_entities']

            sentence_buffer.append(sentence)
            if len(sentence_buffer) > overlap:
                sentence_buffer.pop(0)
        else:
            # If adding the sentence would exceed the limit, save the current paragraph and start a new one
            paragraphs.append({'paragraph' : current_paragraph})
            current_paragraph = {'id': len(paragraphs), 'speaker': [sentence_buffer[0]['sentence']['speaker']], 'text': sentence_buffer[0]['sentence']['speaker'] + ': ' + sentence_buffer[0]['sentence']['text'], 'tokens_length': llm.tokenlen(sentence_buffer[0]['sentence']['text']), 'timestamp': sentence_buffer[0]['sentence']['timestamp'], 'named_entities': sentence_buffer[0]['sentence']['named_entities']}
            for buffered_sentence in sentence_buffer[1:]:
                current_paragraph['text'] += ' ' + buffered_sentence['sentence']['text']
                current_paragraph['timestamp'][1] = buffered_sentence['sentence']['timestamp'][1]
                current_paragraph['tokens_length'] += llm.tokenlen(buffered_sentence['sentence']['text'])
                current_paragraph['named_entities'] += buffered_sentence['sentence']['named_entities']
            sentence_buffer = [sentence]
    # Don't forget to add the last paragraph
    if current_paragraph['text']:
        paragraphs.append({'paragraph' : current_paragraph})

    paragraphs_file_path = root_dir / 'report' / 'log' / 'paragraphs.json'
    with open(paragraphs_file_path, 'w') as f:
        json.dump(paragraphs, f, indent=4)

    return paragraphs

def Process_text(transcription_file, rttm_file, output_file, llm_model_name, DataSet_builder=False, verbose=False):
    print("\n-------------------------------------------------------------------------------------\n")
    print("\nPerforming text process ...")
    sentences = load_transcription(transcription_file)
    diarization = load_diarization(rttm_file)
    verbose = verbose
    if not DataSet_builder:
        # Load the configuration file and initialize the LLM model
        with open('Utils/config/config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        if llm_model_name == 'gpt':
            llm = GPT(config['large_language_models']['gpt'])
        elif llm_model_name == 'gemma':
            llm = Gemma(config['large_language_models']['gemma'])
        elif llm_model_name == 'fast-gemma':
            try:
                llm = FastGemma()
            except Exception as e:
                print(f"\n\nFailed to load FastGemma wit error {e}, falling back to Gemma\n\n")
                llm = Gemma(config['large_language_models']['gemma'])
        else:
            llm = Gemma(config['large_language_models']['gemma'])

        all_sentences = process_all_sentences(sentences, diarization)

        # Calculate the total token size of all sentences
        total_token_size = sum(llm.tokenlen(sentence['sentence']['text']) for sentence in all_sentences)

        # Calculate the maximum token output size
        max_token_size = llm.max_tokens
        max_token_output = llm.max_output_tokens
        # Calculate the threshold_percentile based on the total token size
        threshold_percentile = (total_token_size / (((max_token_size / max_token_output)-1)*(max_token_size-max_token_output))) * 10
        threshold_percentile = max(15, min(threshold_percentile, 70))
        threshold_percentile = 0

        print(f"\033[1;32m\nfor llm: \033[1;34m{config['large_language_models'][llm_model_name]}\033[1;32m, Total token size: \033[1;34m{total_token_size}\033[1;32m, Max Token size : \033[1;34m{max_token_size}\033[1;32m, Max token output: \033[1;34m{max_token_output}\033[1;32m, NER Threshold percentile: \033[1;34m{threshold_percentile}\033[1;32m\n\033[0m")
        # process ANR
        anr = ANR(all_sentences)
        speaker_names = anr.GetSpeakerName()
        key_sentences = anr.summarize_text(threshold_percentile)
        all_sentences_with_key_elements = anr.add_key_elements()
        # process chunks of paragraphs sized by max_token_size - max_token_output
        paragraphs = process_paragraph(key_sentences, llm, max_token_size, max_token_output)
        transcription = {'transcription': process_paragraph(all_sentences_with_key_elements, llm , max_token_size, max_token_output)}

        if total_token_size < max_token_size - max_token_output:
            llm_report = llm.summarize(all_sentences, verbose=verbose)

        else:
            # Generate the report with strategy [return MapReduce, Refine]  combined return -> MapReduce, Refine and combined and produce 3 reports
            llm_report = llm.MapReduce(paragraphs, verbose=verbose)
        print("\n-------------------------------end---------------------------------------------------\n")
        return llm_report, speaker_names, transcription

    else:
        with open('Utils/config/config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        llm = summarize_dataset_builder(config['large_language_models']['gpt'])

        all_sentences = process_all_sentences(sentences, diarization)

        # Calculate the total token size of all sentences
        total_token_size = sum(llm.tokenlen(sentence['sentence']['text']) for sentence in all_sentences)

        # Calculate the maximum token output size
        max_token_size = llm.max_tokens
        max_token_output = llm.max_output_tokens
        # Calculate the threshold_percentile based on the total token size
        threshold_percentile = (total_token_size / (((max_token_size / max_token_output)-1)*(max_token_size-max_token_output))) * 10
        threshold_percentile = max(15, min(threshold_percentile, 70))
        threshold_percentile = 0
        print(f"\033[1;32m\nBuilding dataset with llm: \033[1;34m{config['large_language_models']['gpt']}\033[1;32m, with config such as --> Total token size: \033[1;34m{total_token_size}\033[1;32m, Max Token size : \033[1;34m{max_token_size}\033[1;32m, Max token output: \033[1;34m{max_token_output}\033[1;32m, NER Threshold percentile: \033[1;34m{threshold_percentile}\033[1;32m\n\033[0m")
        # process ANR
        anr = ANR(all_sentences)
        speaker_names = anr.GetSpeakerName()
        key_sentences = anr.summarize_text(threshold_percentile)
        # process chunks of paragraphs sized by max_token_size - max_token_output
        paragraphs = process_paragraph(key_sentences, llm, max_token_size, max_token_output)

        if total_token_size < max_token_size - max_token_output:
            llm.summarize_dataset(all_sentences, verbose=verbose)
        else:
            llm.MapReduce_dataset(paragraphs, verbose=verbose)
        print("\n-------------------------------end---------------------------------------------------\n")