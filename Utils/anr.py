# anr.py for automatic name recognition is a program that find the names of the speakers in the audio file by using behaviour analysis. It part of the AI Report Maker program.
from pathlib import Path
import spacy
import json
import numpy as np

class ANR:
    def __init__(self, text):
        self.text = text
        self.nlp = spacy.load("fr_core_news_lg")
        self.trf = spacy.load("fr_dep_news_trf")
        self.entities = {}
        self.organizations = {}
        self.locations = {}
        self.first_names = []
        self.key_elements_counts = []
        self.all_labels = {}
        self.root_dir = Path(__file__).resolve().parent.parent        # Get the absolute path of the root directory of the project
        self.threshold = 0
        self.trf_results = self.TRF()


    def TRF(self):
        trf_results = []

        for sentences_dict in self.text:
            paragraph = sentences_dict['sentence']['text']
            doc = self.nlp(paragraph)
            named_entities = [(ent.text, ent.label_) for ent in doc.ents]
            doc = self.trf(paragraph)
            main_verbs = [token.lemma_ for token in doc if token.dep_ in ('ROOT', 'relcl')]
            subjects_objects = [(token.lemma_, token.dep_) for token in doc if token.dep_ in ('nsubj', 'dobj')]
            sentences = [sent.text for sent in doc.sents]
            timestamp = sentences_dict['sentence']['timestamp']
            speaker_id = sentences_dict['sentence']['speaker']
            trf_results.append({
                'named_entities': named_entities,
                'main_verbs': main_verbs,
                'subjects_objects': subjects_objects,
                'sentences': sentences,
                'timestamp': timestamp,
                'speaker': speaker_id
            })
            self.key_elements_counts.append(len(named_entities) + len(main_verbs) + len(subjects_objects))

        file_path_trf = self.root_dir / 'report' / 'log' / 'trf.json'
        with open(file_path_trf, 'w') as f:
            json.dump(trf_results, f, indent=4)
        return trf_results

    def summarize_text(self,threshold_percentile=50):
        # Identify key sentences
        key_sentences = []
        self.threshold = np.percentile(self.key_elements_counts, threshold_percentile)

        for index, result in enumerate(self.trf_results):
            # Count the number of key elements in the sentence
            num_key_elements = len(result['named_entities']) + len(result['main_verbs']) + len(result['subjects_objects'])
            # If the sentence contains more than a certain number of key elements, consider it a key sentence
            if num_key_elements > self.threshold:
                key_sentences.append({
                        'index': index,
                        'sentence': {
                            'text': result['sentences'][0],
                            'timestamp': result['timestamp'],
                            'speaker': result['speaker'],
                            'named_entities': result['named_entities']
                        }
                    })  # Assuming each result corresponds to one sentence
        file_path_summary = self.root_dir / 'report' / 'log' / 'nlp_summary.json'
        with open(file_path_summary, 'w') as f:
            json.dump(key_sentences, f, indent=4)
        return key_sentences

    def add_key_elements(self):
        # Identify key sentences
        all_sentences_with_key_elements = []
        for index, result in enumerate(self.trf_results):
            all_sentences_with_key_elements.append({
                    'index': index,
                    'sentence': {
                        'text': result['sentences'][0],
                        'timestamp': result['timestamp'],
                        'speaker': result['speaker'],
                        'named_entities': result['named_entities']
                    }
                })  # Assuming each result corresponds to one sentence
        return all_sentences_with_key_elements

    def GetSpeakerName(self):
        speakers_names = {}  # Initialize a dictionary to store speaker IDs and names
        all_speaker_ids = {sentence_dict['speaker'] for sentence_dict in self.trf_results}  # Create a set of all speaker IDs

        for i in range(len(self.trf_results) - 1):  # Iterate over sentences, excluding the last one
            sentence_dict = self.trf_results[i]
            sentence = sentence_dict['sentences'][0]  # Extract the text of the sentence

            # Rule 2: Question Rule
            '''
            If a name is mentioned in a question, it's likely that the name belongs to the speaker who answers the question.
            This is based on the assumption that questions are often directed to the person being asked about
            '''
            if '?' in sentence:
                next_sentence_dict = self.text[i + 1]
                next_speaker_id = next_sentence_dict['sentence']['speaker']  # Extract the speaker ID of the next sentence
                named_entities = sentence_dict['named_entities']
                for entity in named_entities:
                    if entity[1] == 'PER':
                        name = entity[0]
                        speakers_names[next_speaker_id] = name
                        print(f"Rule 2 used to assign {name} to {next_speaker_id}")

        # After all rules have been applied, assign '...' to the remaining speaker IDs
        remaining_speaker_ids = all_speaker_ids - set(speakers_names.keys())
        for speaker_id in remaining_speaker_ids:
            speakers_names[speaker_id] = '...'

        # Replace the speaker IDs with their names
        for speaker_id, name in speakers_names.items():
            print(f"Speaker_names dic: {speaker_id}: {name}")

        return speakers_names



# Rule based idea for speaker identification:
# Rule 3: Introduction Rule
'''
If a name is mentioned in the beginning of a conversation, it's likely that the name belongs to the speaker who is being introduced.
This is based on the assumption that people often introduce themselves at the start of a conversation.
'''
# if i == 0:
#     for name in self.first_names:
#         if name.lower() in sentence.lower():
#             speaker_names[speaker_id] = name
#             print(f"Rule 3 used to assign {name} to {speaker_id}")

# Rule 1: Last Speaker Rule
'''
If a name is mentioned right after a speaker's turn, it's likely that the name belongs to the last speaker.
This is based on the assumption that people often refer to themselves in the third person when speaking.
'''
# for name in self.first_names:
#     if name.lower() in sentence.lower():
#         speaker_names[speaker_id] = name
#         print(f"Rule 1 used to assign {name} to {speaker_id}")