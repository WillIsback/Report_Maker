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
        return len(self.tokenizer(text, return_tensors='pt').to("cuda")['input_ids'][0])   
     
    def request(self, text ,max_output_tokens=256):
        try:  
            print(f"\nGenerating key points with gemma for the following text ...\n")

            assistant_msg = 'Vous êtes un assistant utile qui résume une transcription en français.'
            
            prompt = "Génère le résumé des dialogues suivants: "  + text + "\n"
            messages = f"System: {assistant_msg} \n User: {prompt} \n Le résumé de la transcription est: "
            
            # print(f"\nRequesting response from GEMMA API for the following prompt: {messages} \n")
            inputs = self.tokenizer(messages, return_tensors='pt').to("cuda")
            print(f"Input tokenize size is {len(inputs['input_ids'][0])}\n")
            max_output_tokens = max_output_tokens + len(inputs['input_ids'][0])
            print(f"Max output tokens: {max_output_tokens}\n")
            outputs = self.model.generate(**inputs, max_length=max_output_tokens, min_length=50, num_return_sequences=1)
            
            text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            generated_response = text.split("Le résumé de la transcription est:")[1]
            # print(f"\nAll generated: {text}\n")
            # print(f"\nResponse generated: {generated_response}\n")
            
            return generated_response
    
        except (Exception, SystemExit) as e:
            print(f"An unexpected error occurred: {e}")
            return None


    def summarize(self, text, max_output_tokens=256):
        try:  
            print(f"\nGenerating conclusion with gemma for the following text ...\n")

            assistant_msg = 'Vous êtes un assistant utile qui génère un résumé de rapport en français.'
            
            prompt = "Génère la conclusion des textes ci-dessous:\n\n" + text + "\n\n"
            messages = f"System: {assistant_msg} \n User: {prompt} \n La conclusion des textes est: "
            
            inputs = self.tokenizer(messages, return_tensors='pt').to("cuda")
            print(f"Input tokenize size is {len(inputs['input_ids'][0])}\n")
            max_output_tokens = max_output_tokens + len(inputs['input_ids'][0])
            print(f"Max output tokens: {max_output_tokens}\n")
            outputs = self.model.generate(**inputs, max_length=max_output_tokens, min_length=256, num_return_sequences=1)
            
            text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            generated_response = text.split("La conclusion des textes est:")[1]
            # print(f"\nAll generated: {text}\n")
            # print(f"Response generated: {generated_response}")
            
            return generated_response
        
    
        except (Exception, SystemExit) as e:
            print(f"An unexpected error occurred: {e}")
            return None