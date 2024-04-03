![poster](images/Ai_report_maker_poster.jpg)
# AI Report Maker
AI Report Maker is a powerful tool designed to automate the process of transcribing and summarizing meetings. It leverages state-of-the-art machine learning models to provide detailed and accurate reports.

Here's a brief overview of how it works:

1. **Transcription**: AI Report Maker uses the [`Whisper Large v3`](https://huggingface.co/openai/whisper-large-v3) model for automatic speech recognition. It transcribes the audio recording of a meeting into text.

2. **Speaker Diarization**: The program identifies different speakers in the audio file, providing a clear context for the transcription. [`Pyannote.v3`](https://huggingface.co/pyannote/speaker-diarization-3.1) model use for diarization

3. **Annotated Transcription**: The transcriptions are combined with the speaker diarization to generate a full annotated transcription. This includes timestamps, speaker identities, and the transcribed text.

4. **Sub-Summary Generation**: The dialogue is processed using a Language Model of the user's choice (`openai-GPT3.5`, `GEMMA`, `MISTRAL`). This generates a sub-summary for each speaker.

5. **Conclusion Generation**: The sub-summaries are further processed in the same Language Model to generate a comprehensive conclusion for the meeting.

With AI Report Maker, you can transform lengthy meetings into concise, easy-to-read reports.

## Table of Contents

- [AI Report Maker](#ai-report-maker)
- [Flow-Chart](#flow-chart)
- [Features](#features)
- [Future Features](#future-features)
- [Installation](#installation)
  - [Language Model Requirements and Specifications](#language-model-requirements-and-specifications)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Flow-Chart
```mermaid
graph TD;
    A[Audio file .wav/.mp3]--> 
    B(Preprocess audio)-->C(Transcription) 
    B(Pre-process audio)-->D(Diarization)
    E(Pre-process_dialogue)
    C-->|text + timestamp|E
    D-->|speaker_id + timestamp|E
    E --> F{token length <
     context window}
    F -->|YES| G[Summarize dialogue]
    F -->|NO| H[Split Dialogue] --> F
    G --> I{token length < 
    context window}
    I -->|YES| J[Generate conclusion]
    I -->|NO| K[Split Summarize] --> I
    L[Generate Report]
    J --> |conclusion| L
    G --> |content summarize| L
    E --> |full transcription| L
```

## Features

### actual:
- Language supported: French only at the moment
- Automatic Speech Recognition with timestamps
- Speaker Diarization with timestamps
- Generate an annotated transcription with timestamps, speakers, and text
- Generate multiple sub-summary of the dialogues
- Generate a conclusion for the meeting
- Generate a report in markdown format with the annotated transcription, sub-summary, and conclusion

### next:
- Multilanguage support
- Add more Language Models to generate sub-summary and conclusion
- Add a GUI to interact with the program
- Automatic speaker naming and identification
- Extraction of important moments of the meeting from the audio file
- Extraction of Topics discussed in the meeting
- Sentiment analysis of the meeting
- Generate a report in different formats (pdf, docx, etc.)
- Generate report illustrations

## Installation

1. Clone the repository
2. Install the dependencies with `pip install -r requirements.txt`
3. Set up your environment variables for the Hugging Face and OpenAI API keys
4. If you want to use Mistral_AI you will need to Install `llama_cpp` according to the instructions on its [official documentation](https://link-to-llama-cpp-docs). Note: The installation of `llama_cpp` may vary depending on your distribution and whether you have a CUDA-enabled GPU.

### Language Model Requirements and Specifications

This project uses several language models, each with its own requirements and specifications:

- **openai-GPT3.5-Turbo**: This model is used to process the dialogue and generate sub-summaries. Context window of X tokens. It is not open-source, This model is gated with openai credentials.

- **GEMMA-7B/2B**: This model is used to process the sub-summary and generate a conclusion. Context window of X tokens. It is an open-source model and free tier but gated model via hugging face credentials.

- **MISTRAL-7B**: This model is also used to process the sub-summary and generate a conclusion. Context window of 32768 tokens. It is open-source, and has a free tier available. This model is not gated.

Please refer to the official documentation of each model for more detailed information and instructions on how to use them.

## Usage

Run the `main.py` script with your audio file as an argument.

```bash
python main.py your_audio_file.wav
```

This will generate a report in markdown format in the `report` directory.

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## License

[MIT](https://choosealicense.com/licenses/mit/)

