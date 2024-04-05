from openai import OpenAI
from tiktoken import get_encoding
from dotenv import load_dotenv
import os


class GPT:
    def __init__(self, model="gpt-3.5-turbo"):
        self.model = model
        load_dotenv()
        # Get the OpenAI API key from the environment variables
        self.client = OpenAI(api_key=os.getenv('OPENAI_KEY'))
        self.tokenizer = get_encoding("cl100k_base")
        self.max_tokens = 2048
    def encoder(self, text):
        return self.tokenizer.encode(text)

    def tokenlen(self, text):
        return len(self.tokenizer.encode(text))

    def request(self, text, max_output_tokens=100):
        # Define the system message
        system_msg = 'Vous êtes un assistant utile qui génère un résumé.'
        prompt = f"Genere un résumé des dialogues ci-dessous:\n\n {text} \n\n"
        messages=[{"role": "system", "content": system_msg},
                {"role": "user", "content": prompt}]

        messages_length = self.tokenlen(f"{messages[0]['content']} {messages[1]['content']}")
        max_output_tokens = max_output_tokens + messages_length
        print(f"\nGenerating summary with GPT with input token size: {messages_length} and output: {max_output_tokens}\n")
        # Use the language model to generate the summary
        response = self.client.chat.completions.create(model=self.model,
                                                messages=messages,
                                                max_tokens=self.max_tokens,
                                                temperature=0)

        # Extract the generated text from the response
        generated_text = response.choices[0].message.content
        return generated_text

    def summarize(self, text, max_output_tokens=512):
        # Define the system message
        system_msg = 'Vous êtes un assistant utile qui génère un résumé.'
        prompt = f"Genere un résumé des résumés ci-dessous:\n\n {text} \n\n"
        messages=[{"role": "system", "content": system_msg},
                {"role": "user", "content": prompt}]

        messages_length = self.tokenlen(f"{messages[0]['content']} {messages[1]['content']}")
        max_output_tokens = max_output_tokens + messages_length
        print(f"\nGenerating conclusion with GPT with input token size: {messages_length} and output: {max_output_tokens}\n")
        # Use the language model to generate the summary
        response = self.client.chat.completions.create(model=self.model,
                                                messages=messages,
                                                max_tokens=self.max_tokens,
                                                temperature=0)
        # Extract the generated text from the response
        generated_text = response.choices[0].message.content
        return generated_text