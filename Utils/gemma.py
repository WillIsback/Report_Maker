# trunk-ignore-all(isort)
# trunk-ignore-all(black)
# gemma.py is a utility file that contains functions that are used to interact with the GEMMA API
from transformers import AutoTokenizer, pipeline
import torch
import os
from dotenv import load_dotenv
from pathlib import Path
from .anr import CleanNamedEntities
from tqdm import tqdm
load_dotenv()

# Get the Hugging Face API key from the environment variables
HUGGING_FACE = os.getenv('HUGGING_FACE')

class Gemma:
    def __init__(self, model_id="google/gemma-2b-it", dtype=torch.float16):
        self.model_id = model_id
        self.dtype = dtype
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, token=HUGGING_FACE)
        self.max_tokens = 8192
        self.max_output_tokens = 2048
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.pipeline = pipeline(
                "text-generation",
                model=self.model_id,
                model_kwargs={"torch_dtype": torch.float16},
                device=self.device,
                token=HUGGING_FACE
            )
        self.root_dir = Path(__file__).resolve().parent.parent
        self.isNameEntitiesCleaned = False

    def tokenlen(self, text):
        # Activate truncation and padding
        return len(self.tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512).to(self.device)['input_ids'][0])

    def summarize(self, text, verbose=False):
        # Extract the 'text' field from each sentence and join them into a single string
        text = ' '.join(sentence['sentence']['text'] for sentence in text)
        subprompt = [
                {
                    "role": "user",
                    "content": f"""Vous avez plusieurs segments d'une transcription audio sur divers sujets.
                                    Votre tâche est de créer un rapport détaillé à partir de ces informations. Veuillez faire ce qui suit :
                                    - Organisez les sujets en chapitres distincts.
                                    - Fournissez une description détaillée pour chaque sujet, y compris ses points clés, ses implications, son contexte et ses perspectives.
                                    - Mettez en évidence les aspects uniques de chaque sujet et comment ils contribuent à leur domaine respectif.
                                    - Générez le texte dans un ton professionnel et formel, adapté à un rapport d'entreprise.
                                    - Formatez le texte en Markdown.
                                    Voici les résumés : \n\n{text}\n"""
                },
                {
                    "role": "assistant",
                    "content": """# {Titre du Sujet}

                                    ## Aperçu
                                    {Brève description du sujet}

                                    ## Points Clés
                                    {Description détaillée des points clés du sujet}

                                    ## Implications
                                    {Explication des implications du sujet}

                                    ## Contexte
                                    {Description du contexte du sujet}

                                    ## Perspectives
                                    {Discussion sur les perspectives du sujet}

                                    ## Contribution
                                    {Explication de la manière dont le sujet contribue à son domaine respectif}"""
                }
            ]

        prompt = self.pipeline.tokenizer.apply_chat_template(subprompt, tokenize=False, add_generation_prompt=True)
        outputs = self.pipeline(
            prompt,
            do_sample=True,
            temperature=1,
            top_k=50,
            top_p=0.5,
            add_special_tokens=True,
            max_new_tokens=self.max_output_tokens,
        )
        generated_response = (outputs[0]["generated_text"][len(prompt):])
        reponse_token_size = self.tokenlen(generated_response)
        if verbose:
            print(f"\nResponse generated \033[1;34m{reponse_token_size}\033[1;32m token : \n\n\033[0m")
            print(f"{generated_response}\n")

        return generated_response

    def MapReduce(self, texts, verbose=False):
        try:
            Num_paragraph = len(texts)
            sub_summary = []
            list_named_entities = []
            # Calculate max tokens for each sub-summary
            max_tokens_per_summary = self.max_tokens // Num_paragraph
            print(f"\nGenerating MapReduce summary with gemma on a total of {Num_paragraph} paragraph ...\n")

            # Clean named entities for all texts before the loop
            for text in texts:
                text['paragraph']['named_entities'] = CleanNamedEntities(text['paragraph']['named_entities'])
                list_named_entities.append(text['paragraph']['named_entities'])
            self.isNameEntitiesCleaned = True

            # Loop through each text content
            for text in tqdm(texts, desc="Generating MapReduce summary"):
                paragraph = text['paragraph']
                named_entities  = paragraph['named_entities']

                # Skip the current iteration if named_entities is empty -> rule base filtering : if no named entities then skip because the text is not relevant
                if not named_entities:
                    continue

                text_content = paragraph['text']
                inputs_length = paragraph['tokens_length']
                input_id = paragraph['id']
                speaker_id = paragraph['speaker']
                speaker_id = list(set(speaker_id))
                speaker_id = ",".join(speaker_id)
                actual_index = f"{input_id+1}/{Num_paragraph}"

                if self.device == 'cuda':
                    torch.manual_seed(1)

                subprompt = [
                        {
                            "role": "user",
                            "content": f"""Vous avez plusieurs plusieurs segment d'une transcription audio avec les intervenants suivants : {speaker_id}.
                                        Votre tâche est de résumer ce dialogue.
                                        - Identifiez les sujets principaux à l'aide des entités nommées suivantes : \n\n{named_entities}.\n\n
                                        - Fournissez une description.
                                        Voici la transcription à résumer : \n\n{text_content}\n\n"""
                        },
                        {
                            "role": "assistant",
                            "content": """Sujet : {Sujet} \n\n
                                        Description : {Description}"""
                        }
                    ]

                max_length = max_tokens_per_summary
                min_tokens = max_tokens_per_summary
                max_new_tokens = int((inputs_length * 0.5) if (inputs_length < max_length) else min_tokens)
                if verbose:
                    print("\033[1;34m\n-------------------------------------------------------------------------------------\n\n\033[0m")
                    print(f"\033[1;34m\nParagraph: {actual_index}, Input token size: {inputs_length}, Max token length: {max_length},  Max new tokens: {max_new_tokens}\n\033[0m")
                    print(f"\nname_entities : {named_entities}\n")
                prompt = self.pipeline.tokenizer.apply_chat_template(subprompt, tokenize=False, add_generation_prompt=True)
                outputs = self.pipeline(
                    prompt,
                    do_sample=True,
                    temperature=1,
                    top_k=50,
                    top_p=0.5,
                    add_special_tokens=True,
                    max_new_tokens=self.max_output_tokens,
                )
                generated_response = (outputs[0]["generated_text"][len(prompt):])
                reponse_token_size = self.tokenlen(generated_response)
                if verbose:
                    print(f"\nResponse generated \033[1;34m{reponse_token_size}\033[1;32m token : \n\n\033[0m")
                    print(f"{generated_response}\n")
                sub_summary.append(f"sous-résumé {actual_index}: \n {generated_response}")

        except (Exception, SystemExit) as e:
            print(f"An unexpected error occurred: {e}")
            return None

        try:
            # Combine the sub-summary into a single document
            sub_summary = "\n".join(sub_summary)
            length_text_content = self.tokenlen(sub_summary)
            list_named_entities = ",".join(list_named_entities)
            if self.device == 'cuda':
                    torch.manual_seed(2)
            final_prompt = [
                    {
                        "role": "user",
                        "content": f"""Vous avez plusieurs résumés provenant de différentes parties d'une transcription audio.
                                        Votre tâche est de créer un rapport final à partir de ces informations.
                                        Veuillez faire ce qui suit :
                                        - Organisez les sujets en chapitres distincts.
                                        - Fournissez une description pour chaque partie.
                                        - Concentre-toi sur les faits et les entités nommées suivantes: \n\n{named_entities}.\n\n
                                        - Générez le texte en Markdown.
                                            Voici les résumés: \n\n{sub_summary}\n"""
                    },
                    {
                        "role": "assistant",
                        "content": ""
                    }
                ]

            max_new_tokens = self.max_output_tokens
            if verbose:
                print("\033[1;34m\n-------------------------------------------------------------------------------------\n\n\033[0m")
                print(f"\033[1;34m\n\nOverall summary input token size :{length_text_content},  Max length tokens: {self.max_tokens},  Max new tokens: {max_new_tokens}\n\033[0m")
                print(f"\nname_entities : {list_named_entities}\n")
            prompt_combined = self.pipeline.tokenizer.apply_chat_template(final_prompt, tokenize=False, add_generation_prompt=True)
            outputs = self.pipeline(
                prompt_combined,
                do_sample=True,
                temperature=1,
                top_k=50,
                top_p=0.5,
                add_special_tokens=True,
                max_new_tokens=self.max_output_tokens,
            )
            generated_response = (outputs[0]["generated_text"][len(prompt_combined):])
            reponse_token_size = self.tokenlen(generated_response)
            if verbose:
                print(f"\nResponse generated \033[1;34m{reponse_token_size}\033[1;32m token : \n\n\033[0m")
                print(f"{generated_response}\n")
            return generated_response

        except (Exception, SystemExit) as e:
            print(f"An unexpected error occurred: {e}")
            return None

    def Refine(self, texts, verbose=False):
        try:
            Num_paragraph = len(texts)
            # Calculate max tokens for each sub-summary
            max_tokens_per_summary = self.max_tokens
            print(f"\nGenerating Refine strategy summary with gemma on a total of {Num_paragraph} paragraph ...\n")
            # Clean named entities for all texts before the loop
            if not self.isNameEntitiesCleaned:
                for text in texts:
                    text['paragraph']['named_entities'] = CleanNamedEntities(text['paragraph']['named_entities'])

            # Loop through each text content
            for text in tqdm(texts, desc="Generating refined summary"):
                paragraph = text['paragraph']
                named_entities  = paragraph['named_entities']
                text_content = paragraph['text']
                inputs_length = paragraph['tokens_length']
                input_id = paragraph['id']
                speaker_id = paragraph['speaker']
                speaker_id = list(set(speaker_id))
                speaker_id = ",".join(speaker_id)
                actual_index = f"{input_id+1}/{Num_paragraph}"
                if self.device == 'cuda':
                        torch.manual_seed(1)
                # If first paragraph then summarize the first paragraph with an initial prompt
                if input_id == 0:
                    init_prompt = [
                            {
                                "role": "user",
                                "content": f"""Vous avez un segment d'une transcription audio avec les intervenants suivants : {speaker_id}.
                                            Votre tâche est de résumer ce dialogue.
                                            - Identifiez le sujets principaux à l'aide des entités nommées suivantes : \n\n{named_entities}.\n\n
                                            - Fournissez une description.
                                            Concentrez-vous sur les faits et les entités nommées.
                                            Voici la transcription à résumer : \n\n{text_content}\n\n"""
                            },
                            {
                                "role": "assistant",
                                "content": ""
                            }
                        ]
                    max_length = max_tokens_per_summary
                    min_tokens = max_tokens_per_summary * 0.5
                    max_new_tokens = int((inputs_length * 0.5) if (inputs_length < max_length) else min_tokens)
                    if verbose:
                        print("\033[1;34m\n-------------------------------------------------------------------------------------\n\n\033[0m")
                        print(f"\033[1;34m\nParagraph: {actual_index}, Input token size: {inputs_length}, Max token length: {max_length},  Max new tokens: {max_new_tokens}\n\033[0m")
                        print(f"\nname_entities : {named_entities}\n")
                    prompt = self.pipeline.tokenizer.apply_chat_template(init_prompt, tokenize=False, add_generation_prompt=True)
                    outputs = self.pipeline(
                        prompt,
                        do_sample=True,
                        temperature=1,
                        top_k=50,
                        top_p=0.5,
                        add_special_tokens=True,
                        max_new_tokens=self.max_output_tokens,
                    )
                    init_response = (outputs[0]["generated_text"][len(prompt):])
                    reponse_token_size = self.tokenlen(init_response)
                    if verbose:
                        print(f"\nResponse generated \033[1;34m{reponse_token_size}\033[1;32m token : \n\n\033[0m")
                        print(f"{init_response}\n")
                    existing_summary = init_response
                elif (input_id > 0): # If not the first paragraph then summarize the paragraph with a recursive prompt
                    recursive_prompt = [
                            {
                                "role": "user",
                                "content": f"""Vous avez déjà un résumé partiel d'une transcription audio.
                                            Votre mission est maintenant de compléter ce résumé avec une nouvelle transcriptions audio des intervenants suivants : {speaker_id}.
                                                - Organisez les sujets en chapitres distincts.
                                                    - Fournir une description détaillée pour chaque chapitre.
                                                - Concentrez-vous sur les faits et les entités nommées.
                                                    - Voici le résumé existant: \n\n{existing_summary}.\n\n
                                                - Affinez le résumé (si besoin) en fonction des nouvelles informations ci-dessous.
                                                    - Voici les nouvelles entités nommées : \n\n{named_entities}.\n\n
                                                    - Voici la nouvelle transcription : \n\n{text_content}.\n\n
                                            Avec ces nouvelles informations, affinez le résumé original et présentez le au format Markdown."""
                            },
                            {
                                "role": "assistant",
                                            "content": ""
                            }
                        ]
                    max_length = max_tokens_per_summary
                    min_tokens = max_tokens_per_summary * 0.3
                    max_new_tokens = int((inputs_length*0.5) if (inputs_length < max_length*0.5) else min_tokens)
                    if verbose:
                        print("\033[1;34m\n-------------------------------------------------------------------------------------\n\n\033[0m")
                        print(f"\033[1;34m\nParagraph: {actual_index}, Input token size: {inputs_length}, Max token length: {max_length},  Max new tokens: {max_new_tokens}\n\033[0m")
                        print(f"\nname_entities : {named_entities}\n")
                    prompt = self.pipeline.tokenizer.apply_chat_template(recursive_prompt, tokenize=False, add_generation_prompt=True)
                    outputs = self.pipeline(
                        prompt,
                        do_sample=True,
                        temperature=1,
                        top_k=50,
                        top_p=0.5,
                        add_special_tokens=True,
                        max_new_tokens=self.max_output_tokens,
                    )
                    recursive_response = (outputs[0]["generated_text"][len(prompt):])
                    reponse_token_size = self.tokenlen(recursive_response)
                    if verbose:
                        print(f"\nResponse generated \033[1;34m{reponse_token_size}\033[1;32m token : \n\n\033[0m")
                        print(f"{recursive_response}\n")
                    existing_summary += "\n"  + recursive_response

            return existing_summary

        except (Exception, SystemExit) as e:
            print(f"An unexpected error occurred: {e}")
            return None


    def Combined(self, texts, verbose=False):
        MapReduce = self.MapReduce(texts, verbose=verbose)
        Refine = self.Refine(texts, verbose=verbose)
        try:
            print("\nGenerating a combined report with gemma ...\n")
            combined_prompt = [
                    {
                        "role": "user",
                        "content": f"""Vous recevez deux résumés complet de la même transcription audio.
                                    Votre mission est maintenant de générez un compte rendu le plus juste est coherent en recoupant les informations des deux résumés.
                                        - Organisez les sujets en chapitres distincts.
                                            - Fournir une description détaillée pour chaque chapitre.
                                            - Concentrez-vous sur les faits et les entités nommées.
                                        - Réalisez un sommaire en tête de document.
                                            - Voici le résumé issu de la method "MapReduce" : \n\n{MapReduce}.\n\n
                                            - Voici le résumé issu de la method "Refine" : \n\n{Refine}.\n\n
                                    Avec ces deux résumés, générez le compte rendu et présentez le au format Markdown."""
                    },
                    {
                        "role": "assistant",
                                    "content": ""
                    }
                ]
            prompt = self.pipeline.tokenizer.apply_chat_template(combined_prompt, tokenize=False, add_generation_prompt=True)
            outputs = self.pipeline(
                prompt,
                do_sample=True,
                temperature=1,
                top_k=50,
                top_p=0.5,
                add_special_tokens=True,
                max_new_tokens=self.self.max_output_tokens,
            )
            combined_response = (outputs[0]["generated_text"][len(prompt):])
            reponse_token_size = self.tokenlen(combined_response)
            if verbose:
                print(f"\nResponse generated \033[1;34m{reponse_token_size}\033[1;32m token : \n\n\033[0m")
                print(f"{combined_response}\n")
            return combined_response

        except (Exception, SystemExit) as e:
            print(f"An unexpected error occurred: {e}")
            return None