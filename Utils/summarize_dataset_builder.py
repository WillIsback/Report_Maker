# gpt.py
from openai import OpenAI
from tiktoken import get_encoding
from dotenv import load_dotenv
from tqdm import tqdm
import torch
import os
from .prompts import Prompts
import json
from pathlib import Path

# Get the absolute path of the root directory of the project
root_dir = Path(__file__).resolve().parent.parent

class summarize_dataset_builder:
    def __init__(self, model="gpt-3.5-turbo"):
        self.model = model
        load_dotenv()
        # Get the OpenAI API key from the environment variables
        self.client = OpenAI(api_key=os.getenv('OPENAI_KEY'))
        self.tokenizer = get_encoding("cl100k_base")
        self.max_tokens = 4096
        self.max_output_tokens = 1024
        self.max_size = self.max_tokens - self.max_output_tokens
        self.isNameEntitiesCleaned = False
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.prompts = Prompts()
        self.training_data_json = root_dir/'report/log/training_data.json'

    def write_training_data(self, prompt, response, filename='report/log/training_data.json'):
        filename = self.training_data_json
        # Replace the assistant's content with the response
        for message in prompt:
            if message["role"] == "assistant":
                message["content"] = response

        # Save the prompt and response to a JSON file
        with open(filename, 'a') as f:
            json.dump({'messages': prompt}, f, indent=4)
            f.write('\n')

    def tokenlen(self, text):
        return len(self.tokenizer.encode(text, disallowed_special=()))

    def split_text(self, text, mark="sous-résumé"):
        max_size = self.max_size
        # Split the text by the "sous-résumé" mark
        chunks = text.split(mark)
        # Initialize the list of split texts and the current chunk
        split_texts = []
        current_chunk = ""
        # Iterate over the chunks
        for chunk in chunks:
            # If adding the next chunk does not exceed the maximum size

            if self.tokenlen(current_chunk) + self.tokenlen(chunk) <= max_size:
                # Add the chunk to the current chunk
                current_chunk += " {Suite du rapport} : " + chunk
            else:
                # If the current chunk is not empty, add it to the list of split texts
                if current_chunk:
                    split_texts.append(current_chunk)
                # Start a new chunk with the current chunk
                current_chunk = mark + chunk
        # If the last chunk is not empty, add it to the list of split texts
        if current_chunk:
            split_texts.append(current_chunk)
        return split_texts

    def summarize_dataset(self, text, verbose=False):
        # Check if 'text' is a list of dictionaries with a 'sentence' field that contains a 'text' field
        if isinstance(text, list) and all(isinstance(sentence, dict) and 'sentence' in sentence and 'text' in sentence['sentence'] for sentence in text):
            # Extract the 'text' field from each sentence and join them into a single string
            text = ' '.join(sentence['sentence']['text'] for sentence in text)
        # If 'text' is not a list of such dictionaries, use it as it is
        elif isinstance(text, str):
            pass
        else:
            raise ValueError("Invalid input: 'text' should be a string or a list of dictionaries with a 'sentence' field that contains a 'text' field")

        summarize_prompt = self.prompts.summarize_prompt(text)
        messages_length = self.tokenlen(f"{summarize_prompt[0]['content']} {summarize_prompt[1]['content']}")


        if verbose:
            print("\033[1;34m\n-------------------------------------------------------------------------------------\n\n\033[0m")
            print(f"\033[1;34m\nGenerating summary with GPT with input token size: {messages_length} and output: {self.max_output_tokens}\n\033[0m")
        # Use the language model to generate the summary
        response = self.client.chat.completions.create(model=self.model,
                                                messages=summarize_prompt,
                                                max_tokens=self.max_output_tokens,
                                                temperature=0)
        # Extract the generated text from the response
        generated_response = response.choices[0].message.content
        reponse_token_size = self.tokenlen(generated_response)
        if verbose:
            print(f"\nResponse generated \033[1;34m{reponse_token_size}\033[1;32m token : \n\n\033[0m")
            print(f"{generated_response}\n")
        # Write the prompt and response to the file
        self.write_training_data(summarize_prompt, generated_response)
        return generated_response


    def MapReduce_dataset(self, texts, verbose=False):
        try:
            Num_paragraph = len(texts)
            sub_summary = []
            for text in tqdm(texts, desc="Generating MapReduce summary"):
                paragraph = text['paragraph']
                text_content = paragraph['text']
                input_id = paragraph['id']
                speaker_id = paragraph['speaker']
                speaker_id = list(set(speaker_id))
                speaker_id = ",".join(speaker_id)
                actual_index = f"{input_id+1}/{Num_paragraph}"

                if self.device == 'cuda':
                    torch.manual_seed(1)

                # Extract the generated text from the response
                generated_response = self.summarize_dataset(text_content, verbose=verbose)
                sub_summary.append(f"sous-résumé {actual_index}: \n {generated_response}")

        except (Exception, SystemExit) as e:
            print(f"An unexpected error occurred: {e}")
            return None

        try:
            # Combine the sub-summary into a single document
            sub_summary_as_string = "\n".join(sub_summary)
            final_prompt = self.prompts.MapReduce_final_prompt(sub_summary_as_string)
            final_prompt_length = self.tokenlen(f"{final_prompt[0]['content']} {final_prompt[1]['content']}")
            sub_summary = []
            if final_prompt_length > (self.max_tokens - self.max_output_tokens):
                if verbose:
                    print(f"\033[1;34m\n\nOverall summary input token size :{final_prompt_length} >  Max length tokens: {self.max_tokens} - Max new tokens: {self.max_output_tokens}. Refine method called\n\033[0m")
                split_texts = self.split_text(sub_summary_as_string)
                for j, chunk in enumerate(split_texts):
                    new_paragraph = paragraph.copy()
                    new_paragraph['text'] = chunk
                    new_paragraph['id'] = j
                    sub_summary.append({'paragraph': new_paragraph})
                self.Refine_dataset(sub_summary, verbose=verbose)
            else:

                if verbose:
                    print("\033[1;34m\n-------------------------------------------------------------------------------------\n\n\033[0m")
                    print(f"\033[1;34m\n\nOverall summary input token size :{final_prompt_length},  Max length tokens: {self.max_tokens},  Max new tokens: {self.max_output_tokens}\n\033[0m")

                if self.device == 'cuda':
                        torch.manual_seed(2)



                # Use the language model to generate the summary
                response = self.client.chat.completions.create(model=self.model,
                                                        messages=final_prompt,
                                                        max_tokens=self.max_output_tokens,
                                                        temperature=0)

                # Extract the generated text from the response
                generated_response = response.choices[0].message.content
                reponse_token_size = self.tokenlen(generated_response)
                if verbose:
                    print(f"\nResponse generated \033[1;34m{reponse_token_size}\033[1;32m token : \n\n\033[0m")
                    print(f"{generated_response}\n")
                # Write the prompt and response to the file

                self.write_training_data(final_prompt, generated_response)
        except (Exception, SystemExit) as e:
            print(f"An unexpected error occurred: {e}")
            return None

    def Refine_dataset(self, texts, verbose=False):
        try:
            Num_paragraph = len(texts)
            # Initialize existing_summary with the text content of the first paragraph
            existing_summary = texts[0]['paragraph']['text']
            # Loop through each text content
            for i, text in enumerate(tqdm(texts[1:], desc="Generating refined summary")):
                paragraph = text['paragraph']
                text_content = paragraph['text']
                inputs_length = self.tokenlen(text_content)
                actual_index = f"{i+2}/{Num_paragraph}"
                if self.device == 'cuda':
                        torch.manual_seed(1)
                if verbose:
                    print("\033[1;34m\n-------------------------------------------------------------------------------------\n\n\033[0m")
                    print(f"\033[1;34m\nParagraph: {actual_index}, Input token size: {inputs_length}, Max token length: {self.max_tokens},  Max new tokens: {self.max_output_tokens}\n\033[0m")

                Refined_recursive_prompt = self.prompts.Refined_recursive_prompt(existing_summary, text_content)
                # Use the language model to generate the summary
                response = self.client.chat.completions.create(model=self.model,
                                                        messages=Refined_recursive_prompt,
                                                        max_tokens=self.max_output_tokens,
                                                        temperature=0)

                # Extract the generated text from the response
                recursive_response = response.choices[0].message.content
                reponse_token_size = self.tokenlen(recursive_response)
                if verbose:
                    # print(f"\033[1;34m\nPrompt recursif : {Refined_recursive_prompt} \n\n\033[0m")
                    print(f"\nResponse generated \033[1;34m{reponse_token_size}\033[1;32m token : \n\n\033[0m")
                    print(f"{recursive_response}\n")
                existing_summary = recursive_response
                self.write_training_data(Refined_recursive_prompt, recursive_response)
        except (Exception, SystemExit) as e:
            print(f"An unexpected error occurred: {e}")
            return None
