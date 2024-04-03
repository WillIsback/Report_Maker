from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

# Get the OpenAI API key from the environment variables
client = OpenAI(api_key=os.getenv('OPENAI_KEY'))

class GPT:
    def __init__(self, api_key, model="gpt-3.5-turbo"):
        self.api_key = api_key
        self.model = model
        
    def request(self, text, max_output_tokens=256):
        # Define the system message
        system_msg = 'Vous êtes un assistant utile qui génère un résumé.'
        prompt = f"Genere un résumé des dialogues ci-dessous:\n\n {text} \n\n"

        # Use the language model to generate the summary
        response = client.chat.completions.create(model=self.model,
                                                messages=[{"role": "system", "content": system_msg},
                                                        {"role": "user", "content": prompt}],
                                                temperature=0)
        # Extract the generated text from the response
        generated_text = response.choices[0].message.content
        return generated_text
    
    def summarize(self, text, max_output_tokens=256):
        # Define the system message
        system_msg = 'Vous êtes un assistant utile qui génère un résumé.'
        prompt = f"Genere un résumé des résumés ci-dessous:\n\n {text} \n\n"

        # Use the language model to generate the summary
        response = client.chat.completions.create(model=self.model,
                                                messages=[{"role": "system", "content": system_msg},
                                                        {"role": "user", "content": prompt}],
                                                temperature=0)
        # Extract the generated text from the response
        generated_text = response.choices[0].message.content
        return generated_text