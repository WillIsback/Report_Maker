#!/bin/bash

# Get the audio file path from the command line arguments
audio_file=$1

# Check if audio file path is provided
if [ -z "$audio_file" ]
then
    echo "Please provide the audio file path as an argument."
    exit 1
fi

# Run the Python program with each LLM
for llm in gpt gemma bart; do
    python main.py $audio_file --mode dev --llm $llm
done