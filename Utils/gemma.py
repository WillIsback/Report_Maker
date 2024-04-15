# gemma.py is a utility file that contains functions that are used to interact with the GEMMA API
from transformers import AutoTokenizer, AutoConfig, pipeline
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
import torch
import os
from dotenv import load_dotenv
from pathlib import Path
from tqdm import tqdm
from .prompts import Prompts
import gc
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
        self.max_tokens = 2048
        self.max_output_tokens = 1024
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.root_dir = Path(__file__).resolve().parent.parent
        self.prompts = Prompts()
        self.max_size = self.max_tokens - self.max_output_tokens
        try:
            self.pipeline = pipeline(
                "text-generation",
                model=self.model_id,
                model_kwargs={"torch_dtype": torch.float16},
            )
        except Exception as e:
            print(f"Failed to load model {self.model_id}. Error: {e}")


    def tokenlen(self, text):
        # Activate truncation and padding
        return len(self.tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=8192).to(self.device)['input_ids'][0])

    def split_text(self, text, mark="Rapport détaillé"):
        max_size = self.max_size
        # Split the text by the "Rapport détaillé" mark
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
            max_length=1024,
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
                self.Refine(sub_summary, verbose=verbose)
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
                    max_new_tokens=1024,
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
                    max_new_tokens=1024,
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


"""
FAST GEMMA USING UNSLOTH FASTLANGUAGEMODEL LIBRARY ------------------------------------------------------------------------------------
"""
class FastGemma:
    def __init__(self, model_id='Labagaite/gemma-Summarizer-2b-it-LORA-bnb-4bit', dtype=torch.float16):
        gc.collect()
        torch.cuda.empty_cache()
        self.model_id = model_id
        self.dtype = dtype
        self.max_tokens = 4096
        self.max_output_tokens = 2048
        self.max_size = self.max_tokens - self.max_output_tokens
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.root_dir = Path(__file__).resolve().parent.parent
        self.prompts = Prompts()
        try:
            self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.model_id,
            max_seq_length=1024,
            dtype=None,
            load_in_4bit=True,
            )
        except Exception as e:
            print(f"Failed to load model {self.model_id}. Error: {e}")
        try:
            self.tokenizer = get_chat_template(
            self.tokenizer,
            chat_template = 'gemma_chatml', # Supports zephyr, chatml, mistral, llama, alpaca, vicuna, vicuna_old, unsloth
            mapping = {"role": "role", "content": "content", "user": "user", "assistant": "assistant"}, # ShareGPT style
            map_eos_token = True, # Maps <|im_end|> to </s> instead
            )
        except Exception as e:
            print(f"Failed to load chat template. Error: {e}")
        self.tokenizer.paddding_side = "left"
        FastLanguageModel.for_inference(self.model)
        print(self.model.config.max_position_embeddings)



    def tokenlen(self, text):
        if isinstance(text, str):
            # If text is a string, use tokenizer.encode
            return self.tokenizer.encode(text, return_tensors='pt').shape[1]
        elif isinstance(text, list):
            # If text is a list, use apply_chat_template
            return self.tokenizer.apply_chat_template(text, return_tensors='pt').input_ids.shape[1]
        else:
            raise TypeError("Input should be either a string or a list.")

    def check_token_threshold_and_truncate(self, tokenizer, model, messages_chat, max_seq_length):
        try:
            # Check if the input token length is less than the max_seq_length
            input_token_length = len(
                tokenizer.encode(messages_chat)
            )
            if model.config.max_position_embeddings is not None:
                max_model_token_config = model.config.max_position_embeddings
            else:
                max_model_token_config = tokenizer.model_max_length

            MaxTokenCapacityThreshold = (
                max_model_token_config - (input_token_length + max_seq_length)
            ) < 0
            if MaxTokenCapacityThreshold:
                print(
                    "Warning: Maximum token threshold has been reached. Activating truncation to prevent crash. Rouge score will be affected."
                )
                truncation = True
            else:
                truncation = False
            return truncation
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            return None

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

        # check if the input token length is less than the max_seq_length, if it is set truncation to True
        truncation = self.check_token_threshold_and_truncate(
            self.tokenizer, self.model, text, 1024
        )

        prompt = self.tokenizer.apply_chat_template(self.prompts.summarize_prompt(text),
                                                    tokenize=True,
                                                    add_generation_prompt=True,  # Must add for generation
                                                    return_tensors="pt",
                                                    max_length=self.model.config.max_position_embeddings,
                                                    truncation=truncation,).to(self.device)
        summary_ids = self.model.generate(
            input_ids=prompt,
            max_new_tokens = prompt.shape[1] * 0.25,
            do_sample=False,
        )
         # Decode the summary
        summary_text = self.tokenizer.decode(
            summary_ids[0][prompt.shape[1]:], skip_special_tokens=True
        )
        generated_response = summary_text
        reponse_token_size = summary_ids.shape[1] - prompt.shape[1]
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
            final_prompt_length = self.tokenlen(f"{final_prompt[0]['content']}")
            if final_prompt_length > (self.max_tokens - self.max_output_tokens):
                print(f"\033[1;34m\n\nOverall summary input token size :{final_prompt_length} >  Max length tokens: {self.max_tokens} - Max new tokens: {self.max_output_tokens}\n\033 Refine method called\n\033[0m")
                sub_summary = self.Reform_text(sub_summary_as_string, paragraph, verbose=verbose)
                return (self.Refine(sub_summary, verbose=verbose))
            else:

                if verbose:
                    print("\033[1;34m\n-------------------------------------------------------------------------------------\n\n\033[0m")
                    print(f"\033[1;34m\n\nOverall summary input token size :{final_prompt_length},  Max length tokens: {self.max_tokens},  Max new tokens: {self.max_output_tokens}\n\033[0m")

                if self.device == 'cuda':
                        torch.manual_seed(2)
                final_prompt = self.tokenizer.apply_chat_template(final_prompt,
                                                    tokenize=True,
                                                    return_tensors="pt",
                                                    add_generation_prompt=True).to(self.device)
                summary_ids = self.model.generate(
                    input_ids=final_prompt,
                    max_new_tokens = 1024,
                    do_sample=False,
                    pad_token_id = self.tokenizer.pad_token_id,
                    temperature=1,
                    top_k=20,
                    top_p=0.95,
                    repetition_penalty=1.2,
                )
                # Decode the summary
                summary_text = self.tokenizer.decode(
                    summary_ids[0][final_prompt.shape[1]:], skip_special_tokens=True
                )
                generated_response = summary_text
                reponse_token_size = self.tokenlen(generated_response)
                if verbose:
                    print("\033[1;34m\n-------------------------------------------------------------------------------------\n\n\033[0m")
                    print(f"\033[1;34m\n\nOverall summary input token size :{reponse_token_size},  Max length tokens: {self.max_tokens},  Max new tokens: {self.max_output_tokens}\n\033[0m")
                    # print(f"\033[1;34m Final prompt : \n\n {final_prompt}\n\n\033[0m")
                if verbose:
                    print(f"\nResponse generated \033[1;34m{reponse_token_size}\033[1;32m token : \n\n\033[0m")
                    print(f"{generated_response}\n")
                return generated_response

        except (Exception, SystemExit) as e:
            print(f"An unexpected error occurred: {e}")
            return None

    def Reform_text(self, text, paragraph, mark="Rapport détaillé", verbose=False):
        # Split the text by the "sous-résumé" mark
        chunks = text.split(mark)
        # Initialize the list of split texts and the current chunk
        sub_summary = []
        Reformed_paragraph = []
        current_chunk = ""
        # Iterate over the chunks
        for chunk in chunks:
            if self.tokenlen(current_chunk) + self.tokenlen(chunk) <= 2048:
                current_chunk += " {Suite du rapport} : " + chunk
            else:
                # If the current chunk is not empty, add it to the list of split texts
                if current_chunk:
                    Reformed_paragraph.append(current_chunk)
                # Start a new chunk with the current chunk
                current_chunk = mark + chunk
        # If the last chunk is not empty, add it to the list of split texts
        if current_chunk:
            Reformed_paragraph.append(current_chunk)
        # loop through the Reformed_paragraph to format them into a list of dictionaries
        for j, chunk in enumerate(Reformed_paragraph):
            new_paragraph = paragraph.copy()
            new_paragraph['text'] = chunk
            new_paragraph['id'] = j
            sub_summary.append({'paragraph': new_paragraph})
        return sub_summary

    def Refine(self, texts, verbose=False):
        try:
            Num_paragraph = len(texts)
            # Calculate max tokens for each sub-summary
            print(f"\nGenerating Refine strategy summary with gemma on a total of {Num_paragraph} paragraph ...\n")
            # Initialize existing_summary with the text content of the first paragraph
            existing_summary = texts[0]['paragraph']['text']
            # Loop through each text content
            for i, text in enumerate(tqdm(texts[1:], desc="Generating refined summary"), start=1):
                paragraph = text['paragraph']
                text_content = paragraph['text']
                actual_index = f"{i+1}/{Num_paragraph}"
                Refined_recursive_prompt = self.prompts.Refined_recursive_prompt(existing_summary,text_content)
                Refined_recursive_prompt_length = self.tokenlen(f"{Refined_recursive_prompt[0]['content']}")
                Refined_recursive_prompt = self.tokenizer.apply_chat_template(Refined_recursive_prompt,
                                                    tokenize=True,
                                                    return_tensors="pt",
                                                    add_generation_prompt=True).to(self.device)
                if self.device == 'cuda':
                        torch.manual_seed(1)
                if verbose:
                    print("\033[1;34m\n-------------------------------------------------------------------------------------\n\n\033[0m")
                    print(f"\033[1;34m\nParagraph: {actual_index}, Input token size: {Refined_recursive_prompt_length}, Max token length: {self.max_tokens},  Max new tokens: {self.max_output_tokens}\n\033[0m")
                summary_ids = self.model.generate(
                    input_ids=Refined_recursive_prompt,
                    max_new_tokens = 1024,
                    do_sample=False,
                    pad_token_id = self.tokenizer.pad_token_id,
                    temperature=1,
                    top_k=20,
                    top_p=0.95,
                    repetition_penalty=1.2,
                )
                # Decode the summary
                summary_text = self.tokenizer.decode(
                    summary_ids[0][Refined_recursive_prompt.shape[1]:], skip_special_tokens=True
                )
                generated_response = summary_text
                reponse_token_size = self.tokenlen(generated_response)
                if verbose:
                    print(f"\nResponse generated \033[1;34m{reponse_token_size}\033[1;32m token : \n\n\033[0m")
                    print(f"{generated_response}\n")
                existing_summary = generated_response
            return existing_summary
        except (Exception, SystemExit) as e:
            print(f"An unexpected error occurred: {e}")
            return None
