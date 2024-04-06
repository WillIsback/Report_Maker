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
        self.max_output_tokens = self.max_tokens * 0.5
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.pipeline = pipeline(
                "text-generation",
                model=self.model_id,
                model_kwargs={"torch_dtype": torch.float16},
                device=self.device,
                token=HUGGING_FACE
            )
        self.root_dir = Path(__file__).resolve().parent.parent


    def tokenlen(self, text):
        # Activate truncation and padding
        return len(self.tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512).to(self.device)['input_ids'][0])

    def get_coef(self, Num_paragraph):
        # Define the minimum and maximum coefficients
        min_coef, max_coef = 0.1, 0.3

        # Normalize Num_paragraph to the range [0, 1]
        normalized_Num_paragraph = Num_paragraph / max(1, Num_paragraph)  # avoid division by zero

        # Calculate the coefficient
        coef = min_coef + (max_coef - min_coef) * normalized_Num_paragraph

        return coef

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
                    max_new_tokens=max_new_tokens,
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
            coef = self.get_coef(Num_paragraph)
            max_new_tokens = int((self.max_tokens - length_text_content) * coef)
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
                max_new_tokens=max_new_tokens,
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