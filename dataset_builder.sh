#!/bin/bash

# Directory containing the audio files
dir=$1

# Find all audio files in the directory and its subdirectories, excluding files that end with "_preprocessed.wav"
audio_files=$(find $dir -type f \( \( -name "*.wav" -o -name "*.mp3" \) ! -name "*_preprocessed.wav" \) -exec readlink -f {} \;)

# Count the total number of audio files
total_files=$(echo "$audio_files" | wc -l)

# Print the total number of audio files
echo -e "\e[32mTotal number of audio files: $total_files\e[0m"

# Calculate 20% and 80% of the total number of files
val_files=$(echo "$total_files * 0.20" | bc)
train_files=$(echo "$total_files * 0.80" | bc)

# Round the numbers to the nearest integer
val_files=$(printf "%.0f" $val_files)
train_files=$(printf "%.0f" $train_files)

# Check if the training_data.json file exists before deleting it
if [ -f report/log/training_data.json ]; then
    rm -f report/log/training_data.json
fi

# Generate the validation dataset
echo "$audio_files" | head -n $val_files | while read file; do
    python main.py $file --mode build_dataset
done

# Check if the training_data.json file exists before Rename it
if [ -f report/log/training_data.json ]; then
    # Rename the training_data.json file to validation_data.json
    mv report/log/training_data.json report/log/validation_data.json
    echo -e "\e[32mFinished generating the validation set and renamed the file\e[0m"
fi

# Generate the training dataset
echo "$audio_files" | tail -n $train_files | while read file; do
    python main.py $file --mode build_dataset
done

echo -e  "\e[32mFinished generating the training set\e[0m"
# The training_data.json file is now ready