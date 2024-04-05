# anr.py for automatic name recognition is a program that find the names of the speakers in the audio file by using behaviour analysis. It part of the AI Report Maker program.
from pathlib import Path
import spacy
import json

class ANR:
    def __init__(self, text):
        self.text = text
        self.nlp = spacy.load("fr_core_news_lg")
        self.entities = {}
        self.organizations = {}
        self.locations = {}
        self.first_names = []
        self.all_labels = {}
    def Full(self):
        onomatopoeia = ['ah', 'eh', 'uh', 'oh', 'um', 'hmm', 'ha', 'hee', 'bah',
                        'poof', 'phew', 'ugh', 'sheesh', 'whew', 'yikes', 'eek', 'hm',
                        'huh', 'whoa', 'oops', 'ouch', 'aw', 'yo', 'wow', 'wham', 'zap',
                        'bang', 'boom', 'crash', 'ding', 'knock', 'tap', 'clang', 'clank',
                        'sizzle', 'splash', 'squish', 'thud', 'thump', 'tick', 'tock',
                        'vroom', 'whir', 'buzz', 'crackle', 'giggle', 'groan', 'grunt',
                        'guffaw', 'ha', 'hoot', 'howl', 'roar', 'scream', 'shriek', 'snicker',
                        'snort', 'squeal', 'yawn', 'yelp', 'yowl', 'yuck']

        # Get the absolute path of the root directory of the project
        root_dir = Path(__file__).resolve().parent.parent

        # Construct the absolute path of the transcription file
        first_names_file_path = root_dir / 'Utils' / 'database' / 'first_names.json'

        # Load your database of first names
        with open(first_names_file_path, 'r') as f:
            first_names_data = json.load(f)

        # Use the French names
        first_names_dict = first_names_data['french']

        for paragraph_dict in self.text:
            paragraph = paragraph_dict['sentence']['text']
            # Remove onomatopoeia
            paragraph = ' '.join(word for word in paragraph.split() if word.lower() not in onomatopoeia)
            doc = self.nlp(paragraph)
            for ent in doc.ents:
                if ent.label_ == 'PER':
                    self.entities[ent.text] = ent.label_
                if ent.label_ == 'ORG':
                    self.organizations[ent.text] = ent.label_
                if ent.label_ == 'LOC':
                    self.locations[ent.text] = ent.label_


        self.all_labels = {**self.entities, **self.organizations, **self.locations}


        ent_first_names = [name.split()[0] for name in self.entities.keys()]
        for name in ent_first_names:
            if name.lower() in first_names_dict:
                self.first_names.append(name)

        # Remove duplicates
        self.first_names = list(set(self.first_names))

        # Construct the absolute path of the transcription file
        log_anr_file_path = root_dir/'report/log/log_anr.json'

        # Load your database of first names
        with open(log_anr_file_path, 'w') as f:
            json.dump(self.all_labels, f, indent=4)

        return self.first_names, self.organizations, self.locations, self.entities

    def GetSpeakerName(self):
        speakers_names = {}  # Initialize a dictionary to store speaker IDs and names
        all_speaker_ids = {sentence_dict['sentence']['speaker'] for sentence_dict in self.text}  # Create a set of all speaker IDs

        for i in range(len(self.text) - 1):  # Iterate over sentences, excluding the last one
            sentence_dict = self.text[i]
            sentence = sentence_dict['sentence']['text']  # Extract the text of the sentence
            # Rule 1: Last Speaker Rule
            '''
            If a name is mentioned right after a speaker's turn, it's likely that the name belongs to the last speaker.
            This is based on the assumption that people often refer to themselves in the third person when speaking.
            '''
            # for name in self.first_names:
            #     if name.lower() in sentence.lower():
            #         speaker_names[speaker_id] = name
            #         print(f"Rule 1 used to assign {name} to {speaker_id}")

            # Rule 2: Question Rule
            '''
            If a name is mentioned in a question, it's likely that the name belongs to the speaker who answers the question.
            This is based on the assumption that questions are often directed to the person being asked about
            '''
            if '?' in sentence:
                next_sentence_dict = self.text[i + 1]
                next_speaker_id = next_sentence_dict['sentence']['speaker']  # Extract the speaker ID of the next sentence
                for name in self.first_names:
                    if name.lower() in sentence.lower():
                        speakers_names[next_speaker_id] = name
                        print(f"Rule 2 used to assign {name} to {next_speaker_id}")

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

        # After all rules have been applied, assign '...' to the remaining speaker IDs
        remaining_speaker_ids = all_speaker_ids - set(speakers_names.keys())
        for speaker_id in remaining_speaker_ids:
            speakers_names[speaker_id] = '...'

        # Replace the speaker IDs with their names
        for speaker_id, name in speakers_names.items():
            print(f"Speaker_names dic: {speaker_id}: {name}")
        return speakers_names

