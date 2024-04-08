# trunk-ignore-all(isort)
# trunk-ignore-all(black)
# gemma.py is a utility file that contains functions that are used to interact with the GEMMA API
from transformers import AutoTokenizer, AutoConfig, pipeline
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
        self.config = AutoConfig.from_pretrained(model_id)
        print(self.config.max_position_embeddings)
        self.dtype = dtype
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.max_tokens = 4096
        self.max_output_tokens = 1024
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.root_dir = Path(__file__).resolve().parent.parent
        self.prompts = Prompts()

        try:
            self.pipeline = pipeline(
                "text-generation",
                model=self.model_id,
                model_kwargs={"torch_dtype": torch.float16},
                device=self.device
            )
        except Exception as e:
            print(f"Failed to load model {self.model_id}. Error: {e}")


    def tokenlen(self, text):
        # Activate truncation and padding
        return len(self.tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=8192).to(self.device)['input_ids'][0])

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
                current_chunk += "Suite du rapport : " + chunk
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
            temperature=0.1,
            top_k=20,
            top_p=0.3,
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
                sub_summary.append(f"Rapport détaillé {actual_index}: \n\n {generated_response}\n\n")

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
                    print(f"\033[1;34m\n\nOverall summary input token size :{final_prompt_length} >  Max length tokens: {self.max_tokens} - Max new tokens: {self.max_output_tokens}\n\033 Refine method called\n\033[0m")
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
                final_prompt = self.pipeline.tokenizer.apply_chat_template(final_prompt, tokenize=False, add_generation_prompt=True)
                outputs = self.pipeline(
                    final_prompt,
                    do_sample=True,
                    temperature=0.1,
                    top_k=20,
                    top_p=0.3,
                    add_special_tokens=True,
                    max_new_tokens=self.max_output_tokens,
                )
                length_prompt = self.tokenlen(final_prompt)
                if verbose:
                    print("\033[1;34m\n-------------------------------------------------------------------------------------\n\n\033[0m")
                    print(f"\033[1;34m\n\nOverall summary input token size :{length_prompt},  Max length tokens: {self.max_tokens},  Max new tokens: {self.max_output_tokens}\n\033[0m")
                    # print(f"\033[1;34m Final prompt : \n\n {final_prompt}\n\n\033[0m")
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
            # Initialize existing_summary with the text content of the first paragraph
            existing_summary = texts[0]['paragraph']['text']
          # Loop through each text content
            for i, text in enumerate(tqdm(texts, desc="Generating refined summary")):
                paragraph = text['paragraph']
                text_content = paragraph['text']
                actual_index = f"{i+2}/{Num_paragraph}"
                Refined_recursive_prompt = self.prompts.Refined_recursive_prompt(existing_summary,text_content)
                Refined_recursive_prompt_length = self.tokenlen(f"{Refined_recursive_prompt[0]['content']} {Refined_recursive_prompt[1]['content']}")
                Refined_recursive_prompt = self.pipeline.tokenizer.apply_chat_template(self.prompts.Refined_recursive_prompt(existing_summary,text_content), tokenize=False, add_generation_prompt=True)

                if self.device == 'cuda':
                        torch.manual_seed(1)
                if verbose:
                    print("\033[1;34m\n-------------------------------------------------------------------------------------\n\n\033[0m")
                    print(f"\033[1;34m\nParagraph: {actual_index}, Input token size: {Refined_recursive_prompt_length}, Max token length: {self.max_tokens},  Max new tokens: {self.max_output_tokens}\n\033[0m")

                outputs = self.pipeline(
                    Refined_recursive_prompt,
                    do_sample=True,
                    temperature=0.5,
                    top_k=40,
                    top_p=0.4,
                    add_special_tokens=True,
                    max_new_tokens=self.max_output_tokens,
                )
                recursive_response = (outputs[0]["generated_text"][len(Refined_recursive_prompt):])
                reponse_token_size = self.tokenlen(recursive_response)
                if verbose:
                    print(f"\nResponse generated \033[1;34m{reponse_token_size}\033[1;32m token : \n\n\033[0m")
                    print(f"{recursive_response}\n")
                existing_summary = recursive_response
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
                temperature=0.1,
                top_k=20,
                top_p=0.3,
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