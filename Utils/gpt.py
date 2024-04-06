# gpt.py
from openai import OpenAI
from tiktoken import get_encoding
from dotenv import load_dotenv
from tqdm import tqdm
import torch
import os
from .prompts import Prompts

class GPT:
    def __init__(self, model="gpt-3.5-turbo"):
        self.model = model
        load_dotenv()
        # Get the OpenAI API key from the environment variables
        self.client = OpenAI(api_key=os.getenv('OPENAI_KEY'))
        self.tokenizer = get_encoding("cl100k_base")
        self.max_tokens = 16400
        self.max_output_tokens = 4096
        self.isNameEntitiesCleaned = False
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.prompts = Prompts()

    def tokenlen(self, text):
        return len(self.tokenizer.encode(text))


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
        return generated_response


    def MapReduce(self, texts, verbose=False):
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
                generated_response = self.summarize(text_content, verbose=verbose)
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
            final_prompt = self.prompts.MapReduce_final_prompt(sub_summary)
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
            return generated_response

        except (Exception, SystemExit) as e:
            print(f"An unexpected error occurred: {e}")
            return None

    def Refine(self, texts, verbose=False):
        try:
            Num_paragraph = len(texts)
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
                    existing_summary = self.summarize(text_content, verbose=verbose)
                elif (input_id > 0): # If not the first paragraph then summarize the paragraph with a recursive prompt

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
            print("\nGenerating a combined report with GPT ...\n")

            Combined_prompt = self.prompts.Combined_prompt(MapReduce, Refine)
            # Use the language model to generate the summary
            response = self.client.chat.completions.create(model=self.model,
                                                    messages=Combined_prompt,
                                                    max_tokens=self.max_output_tokens,
                                                    temperature=0)
            # Extract the generated text from the response
            combined_response = response.choices[0].message.content
            reponse_token_size = self.tokenlen(combined_response)
            if verbose:
                print(f"\nResponse generated \033[1;34m{reponse_token_size}\033[1;32m token : \n\n\033[0m")
                print(f"{combined_response}\n")
            return MapReduce, Refine, combined_response

        except (Exception, SystemExit) as e:
            print(f"An unexpected error occurred: {e}")
            return None