# trunk-ignore-all(isort)
# trunk-ignore-all(black)
from pathlib import Path
import subprocess
import math
import argparse
import random
from dotenv import load_dotenv
import os
from datasets import DatasetDict
from datasets import load_dataset
from tqdm import tqdm
import re
# trunk-ignore(ruff/F401)
from huggingface_hub import HfApi, create_repo
from huggingface_hub.hf_api import HfHubHTTPError

load_dotenv()

# Get the Hugging Face API key from the environment variables
HUGGING_FACE = os.getenv('HUGGING_FACE')

def main(audio_dir):
    # Directory containing the audio files
    dir = Path(audio_dir)

    # Find all audio files in the directory and its subdirectories, excluding files that end with "_preprocessed.wav"
    audio_files = [file for file in dir.rglob('*.mp3') if not file.name.endswith("_preprocessed.wav")]

    # Shuffle the audio files
    random.shuffle(audio_files)

    # Count the total number of audio files
    total_files = len(audio_files)

    # Print the total number of audio files
    print(f"\033[1;32m\nTotal number of audio files: \033[1;34m{total_files}\033[1;32m\n\n\033[0m")

    # Calculate 20% and 80% of the total number of files
    val_files = math.ceil(total_files / 5)
    train_files = math.floor(total_files * 0.80)

    # Ensure that val_files and train_files are at least 1
    val_files = max(1, val_files)
    train_files = max(1, train_files)

    # Check if the training_data.json file exists before deleting it
    training_data_file = Path('report/log/training_data.json')
    if training_data_file.exists():
        confirmation = input("Are you sure you want to delete the file? (yes/no): ")
        if confirmation.lower() == 'yes':
            training_data_file.unlink()

    # Generate the validation dataset
    for file in tqdm(audio_files[:val_files], desc="Processing validation files"):
        print(f"\033[1;32m\nProcessing file:  \033[1;34m{file}\033[1;32m\n\033[0m")
        subprocess.run(['python', 'main.py', str(file), '--mode', 'build_dataset'])

    # Check if the training_data.json file exists before Rename it
    if training_data_file.exists():
        # Rename the training_data.json file to validation_data.json
        training_data_file.rename('report/log/validation_data.json')
        print("\033\n[1;32Finished generating the validation set and renamed the file\n\033[0m")

    # Generate the training dataset
    for file in tqdm(audio_files[val_files:val_files+train_files], desc="Processing training files"):
        print(f"\033[1;32m\nProcessing file:  \033[1;34m{file}\033[1;32m\n\033[0m")
        subprocess.run(['python', 'main.py', str(file), '--mode', 'build_dataset'])

    print("\033\n[1;32Finished generating the training set\n\033[0m")
    # The training_data.json file is now ready



    # Load the RTTM, transcription, and WAV files
    rttm_files = [str(file) for file in Path('report/dataset').rglob('*.rttm')]
    transcription_files = [str(file) for file in Path('report/dataset').rglob('*.json')]
    wav_files = [str(file) for file in Path('report/dataset').rglob('*.wav')]

    # Get the list of all files
    all_files = rttm_files + transcription_files + wav_files

    # Extract the index from the file name
    file_indices = set()
    for file in all_files:
        match = re.search(r'podcast_fr_(\d+)', file)
        if match is not None:
            file_indices.add(match.group(1))

    # Rename files
    for file_index in file_indices:
        for file in all_files:
            if f'podcast_fr_{file_index}' in file:
                extension = os.path.splitext(file)[1]
                os.rename(file, f'report/dataset/{file_index}{extension}')

    # Create tar file
    subprocess.run(['tar', '--sort=name', '-cf', 'report/dataset/dataset.tar', 'report/dataset/'])

    api = HfApi()
    # Define the name of your Hugging Face account and the name of the repository
    repository_name = 'Labagaite/fr-transcription-diarization-dataset'
    # Try to create the repository
    try:
        api.create_repo(repository_name, repo_type='dataset',token=HUGGING_FACE)
    except HfHubHTTPError as e:
        if 'You already created this dataset repo' in str(e):
            print(f'The repository {repository_name} already exists.')
        else:
            raise

    api.upload_file(
        path_or_fileobj='report/dataset/dataset.tar',
        path_in_repo='dataset.tar',
        repo_id=repository_name,
        repo_type="dataset",
    )



    # Create a dataset
    # Load the validation and training data from the JSON files
    # chatml data format : {"role" : "user", "content" : "", "role" : "assistant", "content" : ""}
    training_data = load_dataset('json', data_files='report/log/training_data.json')
    validation_data = load_dataset('json', data_files='report/log/validation_data.json')
    # Create a dataset
    dataset = DatasetDict({
        'train': training_data['train'],
        'validation': validation_data['train'],
    })
    # Save the dataset to disk
    dataset.save_to_disk('report/dataset')
    # Push the dataset to your Hugging Face account
    dataset.push_to_hub('Labagaite/fr-summarizer-dataset', token=HUGGING_FACE)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process audio files.')
    parser.add_argument('audio_dir', type=str, help='The directory containing the audio files')

    args = parser.parse_args()

    main(args.audio_dir)