# post process is the program tha generate the report from the summarize text and the paragraps done by the AI
from pathlib import Path
import json
import datetime

# Get the absolute path of the root directory of the project
root_dir = Path(__file__).resolve().parent.parent

def generate_report(json_output, markdown_file):
    with open(json_output, 'r') as f:
        json_output = json.load(f)
    trf_file_path = root_dir/'report'/'log'/'trf.json'
    with open(trf_file_path, 'r') as f:
        trf_results = json.load(f)

    # Create a set of all named entities for quick lookup
    named_entities = set()
    for result in trf_results:
        for entity in result['named_entities']:
            named_entities.add(entity[0])

    # Vérifiez si un ou trois rapports sont présents
    if 'llm_report' in json_output:
        llm_reports = {'llm_report': json_output['llm_report']}
    else:
        llm_reports = {key: value for key, value in json_output.items() if key.startswith('llm_report')}

    # Create an empty list to store the file paths
    markdown_files = []

    # Pour chaque rapport, créez un fichier Markdown
    for report_name, report_content in llm_reports.items():
        # Ajoutez l'extension appropriée au nom du fichier
        markdown_file_with_extension = f"{markdown_file}_{report_name}.md"
        with open(markdown_file_with_extension, 'w') as f:
            # Écrivez le titre
            date = datetime.date.today().strftime("%d/%m/%Y")
            f.write(f"# Compte-rendu réunion {date}\n\n")

            # Écrivez la table des matières
            f.write("## Table des matières\n\n")
            f.write("1. [Transcription de la réunion](#Transcription-de-la-réunion)\n")
            f.write("2. [Résumé de la réunion](#Résumé-de-la-réunion)\n")
            f.write("3. [Conclusion](#conclusion)\n\n")

            # Écrivez la section de transcription complète
            f.write("## Transcription de la réunion\n\n")
            f.write("<details>\n<summary>View Full Transcription</summary>\n\n")
            for sentence in json_output['details']:
                # Vérifiez chaque mot dans le texte, s'il s'agit d'une entité nommée ou d'un interlocuteur, mettez-le en gras
                text = ' '.join(word if word not in named_entities else f'**{word}**' for word in sentence['paragraph']['text'].split())
                text = ' '.join(word if not word.startswith('SPEAKER_') else f'<br>**{word}**' for word in sentence['paragraph']['text'].split())
                f.write(f"Timestamp : {sentence['paragraph']['timestamp']} / {sentence['paragraph']['speaker']}:<br> {text} <br> \n\n")
            f.write("</details>\n\n")

            # Écrivez la section de résumé du contenu
            f.write("## Résumé de la réunion\n\n")
            f.write("### Noms des participants\n\n")
            for speaker_id, name in json_output['speaker_names'].items():
                f.write(f"-{speaker_id}: {name}\n")
            f.write("### Sections\n\n")
            f.write(f"{report_content}\n\n")

        # Append the file path to the list
        markdown_files.append(markdown_file_with_extension)

    # Return the list of file paths
    return markdown_files