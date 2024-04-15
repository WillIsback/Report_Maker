# post process is the program tha generate the report from the summarize text and the paragraps done by the AI
from pathlib import Path
import datetime

# Get the absolute path of the root directory of the project
root_dir = Path(__file__).resolve().parent.parent

def generate_report(transcription, output_file, audio_file, resume):
    if not isinstance(transcription, dict):
        raise ValueError("transcription must be a dictionary")
    if 'transcription' not in transcription:
        raise ValueError("transcription must have a 'transcription' key")

    paragraph_str = ''
    # Ajoutez l'extension appropriée au nom du fichier
    for paragraph in transcription['transcription']:
        text = ' '.join(word if not word.startswith('SPEAKER_') else f'<br>**{word}**' for word in paragraph['paragraph']['text'].split())
        timestamp = ' / '.join(str(t) for t in paragraph['paragraph']['timestamp'])
        speaker = ', '.join(set(paragraph['paragraph']['speaker']))
        paragraph_str += f"Timestamp : {timestamp} / {speaker}:<br> {text} <br> \n\n"

    markdown_file_with_extension = output_file
    with open(markdown_file_with_extension, 'w') as f:
        # Écrivez le titre
        date = datetime.date.today().strftime("%d/%m/%Y")
        report = REPORT_TEMPLATE.format(
            audio_file=audio_file,
            date=date,
            trancription=paragraph_str,
            resume=resume
        )
        f.write(report)
    # Return the list of file paths
    return markdown_file_with_extension




REPORT_TEMPLATE = """
# Compte-rendu du la transcription {audio_file} du {date}
## Table des matières
- [Transcription de la réunion](#Transcription-de-la-réunion)
- [Résumé de la réunion](#Résumé-de-la-réunion)
- [Conclusion](#conclusion)

## Transcription de la réunion
<details>
<summary>Voir la transcription en entier</summary>
\n\n{trancription}\n\n
</details>

## Résumé de la transcription
{resume}



"""