# trunk-ignore-all(isort)
# trunk-ignore-all(black)
# gemma.py is a utility file that contains functions that are used to interact with the GEMMA API
from transformers import AutoTokenizer, pipeline
import torch
import os
from dotenv import load_dotenv
from pathlib import Path
from tqdm import tqdm
from .prompts import Prompts
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
        self.prompts = Prompts()


    def tokenlen(self, text):
        # Activate truncation and padding
        return len(self.tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512).to(self.device)['input_ids'][0])

    def summarize(self, text, verbose=False):
        # Check if 'text' is a list of dictionaries with a 'sentence' field that contains a 'text' field
        if isinstance(text, list) and all(isinstance(sentence, dict) and 'sentence' in sentence and 'text' in sentence['sentence'] for sentence in text):
            # Extract the 'text' field from each sentence and join them into a single string
            text = ' '.join(sentence['sentence']['text'] for sentence in text)
        # If 'text' is not a list of such dictionaries, use it as it is
        elif isinstance(text, str):
            pass
        else:
            raise ValueError("Invalid input: 'text' should be a string or a list of dictionaries with a 'sentence' field that contains a 'text' field")

        prompt = self.pipeline.tokenizer.apply_chat_template(self.prompts.summarize_prompt(text), tokenize=False, add_generation_prompt=True)
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


            print(f"\nGenerating MapReduce summary with gemma on a total of {Num_paragraph} paragraph ...\n")

            # Loop through each text content
            for text in tqdm(texts, desc="Generating MapReduce summary"):
                paragraph = text['paragraph']

                text_content = paragraph['text']
                inputs_length = paragraph['tokens_length']
                input_id = paragraph['id']
                speaker_id = paragraph['speaker']
                speaker_id = list(set(speaker_id))
                speaker_id = ",".join(speaker_id)
                actual_index = f"{input_id+1}/{Num_paragraph}"

                if self.device == 'cuda':
                    torch.manual_seed(1)
                if verbose:
                    print("\033[1;34m\n-------------------------------------------------------------------------------------\n\n\033[0m")
                    print(f"\033[1;34m\nParagraph: {actual_index}, Input token size: {inputs_length}, Max token length: {self.max_tokens},  Max new tokens: {self.max_output_tokens}\n\033[0m")

                generated_response = self.summarize(text_content, verbose=verbose)
                reponse_token_size = self.tokenlen(generated_response)
                sub_summary.append(f"sous-résumé {actual_index}: \n {generated_response}")

        except (Exception, SystemExit) as e:
            print(f"An unexpected error occurred: {e}")
            return None

        try:
            # Combine the sub-summary into a single document
            sub_summary = "\n".join(sub_summary)
            length_text_content = self.tokenlen(sub_summary)
            if self.device == 'cuda':
                    torch.manual_seed(2)
            if verbose:
                print("\033[1;34m\n-------------------------------------------------------------------------------------\n\n\033[0m")
                print(f"\033[1;34m\n\nOverall summary input token size :{length_text_content},  Max length tokens: {self.max_tokens},  Max new tokens: {self.max_output_tokens}\n\033[0m")
            final_prompt = self.pipeline.tokenizer.apply_chat_template(self.prompts.MapReduce_final_prompt(text_content), tokenize=False, add_generation_prompt=True)
            outputs = self.pipeline(
                final_prompt,
                do_sample=True,
                temperature=1,
                top_k=50,
                top_p=0.5,
                add_special_tokens=True,
                max_new_tokens=self.max_output_tokens,
            )
            generated_response = (outputs[0]["generated_text"][len(final_prompt):])
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
            print(f"\nGenerating Refine strategy summary with gemma on a total of {Num_paragraph} paragraph ...\n")
          # Loop through each text content
            for text in tqdm(texts, desc="Generating refined summary"):
                paragraph = text['paragraph']
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
                    init_response = self.summarize(text_content, verbose=verbose)
                    reponse_token_size = self.tokenlen(init_response)
                    existing_summary = init_response

                elif (input_id > 0): # If not the first paragraph then summarize the paragraph with a recursive prompt

                    if verbose:
                        print("\033[1;34m\n-------------------------------------------------------------------------------------\n\n\033[0m")
                        print(f"\033[1;34m\nParagraph: {actual_index}, Input token size: {inputs_length}, Max token length: {self.max_tokens},  Max new tokens: {self.max_output_tokens}\n\033[0m")
                    Refined_recursive_prompt = self.pipeline.tokenizer.apply_chat_template(self.prompts.Refined_recursive_prompt(existing_summary,text_content), tokenize=False, add_generation_prompt=True)
                    outputs = self.pipeline(
                        Refined_recursive_prompt,
                        do_sample=True,
                        temperature=1,
                        top_k=50,
                        top_p=0.5,
                        add_special_tokens=True,
                        max_new_tokens=self.max_output_tokens,
                    )
                    recursive_response = (outputs[0]["generated_text"][len(Refined_recursive_prompt):])
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

            Combined_prompt = self.pipeline.tokenizer.apply_chat_template(self.prompts.Combined_prompt(MapReduce,Refine), tokenize=False, add_generation_prompt=True)
            outputs = self.pipeline(
                Combined_prompt,
                do_sample=True,
                temperature=1,
                top_k=50,
                top_p=0.5,
                add_special_tokens=True,
                max_new_tokens=self.max_output_tokens,
            )
            combined_response = (outputs[0]["generated_text"][len(Combined_prompt):])
            reponse_token_size = self.tokenlen(combined_response)
            if verbose:
                print(f"\nResponse generated \033[1;34m{reponse_token_size}\033[1;32m token : \n\n\033[0m")
                print(f"{combined_response}\n")
            return MapReduce, Refine, combined_response

        except (Exception, SystemExit) as e:
            print(f"An unexpected error occurred: {e}")
            return None