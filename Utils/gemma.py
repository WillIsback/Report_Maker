# gemma.py is a utility file that contains functions that are used to interact with the GEMMA API
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os
from dotenv import load_dotenv

load_dotenv()

# Get the Hugging Face API key from the environment variables
HUGGING_FACE = os.getenv('HUGGING_FACE')

class Gemma:
    def __init__(self, model_id="google/gemma-2b-it", dtype=torch.float16):
        self.model_id = model_id
        self.dtype = dtype
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, token=HUGGING_FACE)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="cuda",
            torch_dtype=dtype,
            token=HUGGING_FACE
        )
    def encoder(self, text):
        return self.tokenizer(text, return_tensors='pt').to("cuda")

    def tokenlen(self, text):
        # Activate truncation and padding
        return len(self.tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512).to("cuda")['input_ids'][0])

    def request(self, text ,max_output_tokens=100):
        try:
            info_mess = "\nGenerating key points with gemma for the following text ...\n"
            print(info_mess)

            prompt = "Résume les dialogues suivants: "  + text + "\n"
            messages = f"{prompt} \n Le résumé des dialogues est: "

            # print(f"\nRequesting response from GEMMA API for the following prompt: {messages} \n")
            inputs = self.tokenizer(messages, return_tensors='pt').to("cuda")
            max_output_tokens = max_output_tokens + len(inputs['input_ids'][0])
            print(f"Input token size is {len(inputs['input_ids'][0])} Max output tokens: {max_output_tokens}\n")
            outputs = self.model.generate(**inputs, max_length=max_output_tokens, num_return_sequences=1, temperature=0.9)

            text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            split_text = text.split("Le résumé des dialogues est:")
            generated_response = split_text[1].strip() if len(split_text) > 1 else "No summary found"
            # print(f"\nAll generated: \n{text}\n")
            # print(f"\nResponse generated: \n {generated_response}\n")

            return generated_response

        except (Exception, SystemExit) as e:
            print(f"An unexpected error occurred: {e}")
            return None


    def summarize(self, text, max_output_tokens=512):
        try:
            info_mess = "\nGenerating conclusion with gemma for the following text ...\n"
            print(info_mess)

            prompt = "Résume les textes ci-dessous:\n\n" + text + "\n\n"
            messages = f"{prompt} Le résumé des textes est: "

            inputs = self.tokenizer(messages, return_tensors='pt').to("cuda")
            max_output_tokens = max_output_tokens + len(inputs['input_ids'][0])
            print(f"Input token size is {len(inputs['input_ids'][0])} Max output tokens: {max_output_tokens}\n")
            outputs = self.model.generate(**inputs, max_length=max_output_tokens, num_return_sequences=1, temperature=0.9)

            text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            split_text = text.split("Le résumé des textes est:")
            generated_response = split_text[1].strip() if len(split_text) > 1 else "No summary found"
            # print(f"\nAll generated: \n{text}\n")
            # print(f"Response generated: \n{generated_response}")

            return generated_response


        except (Exception, SystemExit) as e:
            print(f"An unexpected error occurred: {e}")
            return None