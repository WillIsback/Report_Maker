# bert.py
import torch
from transformers import RobertaTokenizerFast, EncoderDecoderModel, BartForConditionalGeneration, BartTokenizer, pipeline
import textwrap

class BART:
    def __init__(self, model_id="facebook/bart-large-cnn"):
        self.model_id = model_id
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.model = BartForConditionalGeneration.from_pretrained(model_id).to(self.device)
        self.tokenizer = BartTokenizer.from_pretrained(model_id)

        self.translator_en_to_fr = pipeline('translation', model='Helsinki-NLP/opus-mt-en-fr')
        self.translator_fr_to_en = pipeline('translation', model='Helsinki-NLP/opus-mt-fr-en')
        self.max_tokens = 512  # Set the max tokens for BART

    def tokenlen(self, text):

        return len(self.tokenizer(text)['input_ids'])

    def request(self, text, max_output_length=512):
        try:
            info_mess="\nGenerating key points with BART for the following text ...\n"
            print(info_mess)
            prompt_length = self.tokenlen(text)
            print(f"\nPrompt length: {prompt_length}\n")
            if prompt_length > self.max_tokens :
                print(f"Text too long, please provide a shorter text : {prompt_length}")
                return None

            # text_en = self.translator_fr_to_en(text, max_length=max_output_length)[0]['translation_text']
            inputs = self.tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=max_output_length, truncation=True).to(self.device)
            summary_ids = self.model.generate(inputs, max_length=150, min_length=50, length_penalty=2.0, num_beams=4, early_stopping=True)
            summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
            formatted_summary = "\n".join(textwrap.wrap(summary, width=80))
            text_fr = self.translator_en_to_fr(formatted_summary, max_length=512)[0]['translation_text']
            return text_fr


        except (Exception, SystemExit) as e:
            print(f"An unexpected error occurred: {e}")
            return None

    def summarize(self, text, max_output_length=512):
        try:
            info_mess="\nGenerating conclusion with BART for the following text ...\n"
            print(info_mess)
            prompt_length = self.tokenlen(text)
            print(f"\nPrompt length: {prompt_length}\n")
            if prompt_length > self.max_tokens :
                print(f"Text too long, please provide a shorter text : {prompt_length}")
                return None
            # text_en = self.translator_fr_to_en(text, max_length=max_output_length)[0]['translation_text']
            inputs = self.tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=self.max_tokens, truncation=True).to(self.device)
            summary_ids = self.model.generate(inputs, max_length=150, min_length=50, length_penalty=2.0, num_beams=4, early_stopping=True)
            summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
            formatted_summary = "\n".join(textwrap.wrap(summary, width=80))
            text_fr = self.translator_en_to_fr(formatted_summary, max_length=512)[0]['translation_text']
            return text_fr


        except (Exception, SystemExit) as e:
            print(f"An unexpected error occurred: {e}")
            return None

class Camembert:
    def __init__(self, model_id="mrm8488/camembert2camembert_shared-finetuned-french-summarization"):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model_id = model_id
        self.model = EncoderDecoderModel.from_pretrained(self.model_id).to(self.device)
        self.tokenizer = RobertaTokenizerFast.from_pretrained(self.model_id)
        self.max_tokens = 512  # Set the max tokens for Camembert
    def tokenlen(self, text):
        return len(self.tokenizer(text))

    def request(self, text, max_output_length=512):
        try:
            max_output_length = min(max_output_length + self.tokenlen(text), 512)

            if(self.tokenlen(text) > 512):
                print(f"Text too long, please provide a shorter text : {self.tokenlen(text)}")
                return None

            inputs = self.tokenizer([text], padding="max_length", truncation=True, max_length=512, return_tensors="pt")
            input_ids = inputs.input_ids.to(self.device)
            attention_mask = inputs.attention_mask.to(self.device)
            output = self.model.generate(input_ids, attention_mask=attention_mask)
            print(f"\nGenerating summary with camembert with max token size: {max_output_length}\n")
            print(f"\nOutput: {self.tokenizer.decode(output[0], skip_special_tokens=True)}")
            return self.tokenizer.decode(output[0], skip_special_tokens=True)

        except (Exception, SystemExit) as e:
            print(f"An unexpected error occurred: {e}")
            return None

    def summarize(self, text, max_output_length=512):
        try:
            max_output_length = min(max_output_length + self.tokenlen(text), 512)


            inputs = self.tokenizer([text], padding="max_length", truncation=True, max_length=512, return_tensors="pt")
            input_ids = inputs.input_ids.to(self.device)
            attention_mask = inputs.attention_mask.to(self.device)
            output = self.model.generate(input_ids, attention_mask=attention_mask)
            print(f"\nGenerating conclusion with camembert with max token size: {max_output_length}\n")
            print(f"\nOutput: {self.tokenizer.decode(output[0], skip_special_tokens=True)}")
            return self.tokenizer.decode(output[0], skip_special_tokens=True)

        except (Exception, SystemExit) as e:
            print(f"An unexpected error occurred: {e}")
            return None