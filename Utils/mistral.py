# mistral.py is a utility file that contains the Mistral class that is used to interact with the Mistral API.
from llama_cpp import Llama


class Mistral:
    def __init__(self, model_path="model/mistral-7b-instruct-v0.2.Q4_K_M.gguf", chat_format="mistral-instruct", n_threads=8, n_gpu_layers=35, verbose=False):
        self.model_path = model_path
        self.chat_format = chat_format
        self.n_threads = n_threads
        self.n_gpu_layers = n_gpu_layers
        self.verbose = verbose
        self.max_tokens = 4096
    def request(self, text, max_output_tokens=256):
        try:
            print(f"\nGenerating key points with mistral for the following text ...\n")

            assistant_msg = 'Vous êtes un assistant utile qui génère une réponse formatée avec un style markdown pour un rapport de réunion en français.'

            prompt = "Genere un résumé du discours suivant en français:\n\n" + text + "\n\n"
            messages = [
                        {"role": "assistant", "content": assistant_msg},
                        {"role": "user", "content": prompt}]

            # Calculate the total number of tokens in the messages
            total_message_tokens = sum([len(message['content'].split()) for message in messages])

            # Calculate n_ctx as the sum of the total message tokens and the maximum output tokens
            n_ctx = total_message_tokens + max_output_tokens

            if n_ctx >= 10000:
                print("n_ctx is too large. Please reduce the size of your input or output.")
                return None

            llm = Llama(model_path=self.model_path,
                        chat_format=self.chat_format,
                        n_ctx=n_ctx,
                        n_threads=self.n_threads,
                        n_gpu_layers=self.n_gpu_layers,
                        max_tokens=max_output_tokens,
                        verbose=self.verbose,  # Verbose is required to pass to the callback manager
                        )  # Set chat_format according to the model you are using

            response = llm.create_chat_completion(messages)
            generated_text = response['choices'][0]['message']['content']
            print(f"Response generated: {generated_text}")
            return generated_text

        except (Exception, SystemExit) as e:
            print(f"An unexpected error occurred: {e}")
            return None
