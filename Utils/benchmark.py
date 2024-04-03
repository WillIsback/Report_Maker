# benchmark.py is a utility file to benchmark the performance of the different models used in the AI Report Maker program.

import pandas as pd
import matplotlib.pyplot as plt
import os

def plot(logs_file_path='logs/benchmark.csv'):
    # Load the data from the CSV file
    df = pd.read_csv(logs_file_path, names=['index', 'log_entry_label', 'log_audio_file', 'device', 'device_info', 'whisper_time', 'pyannote_time', 'process_time', 'report_time', 'total_time'])
    
    # Now you can safely split the 'log_entry_label' column
    df[['whisper_model_id', 'pyannote_model_id', 'llm_model_id']] = df['log_entry_label'].str.split(',', expand=True)

    # Convert the time columns to numeric
    df['whisper_time'] = pd.to_numeric(df['whisper_time'])
    df['pyannote_time'] = pd.to_numeric(df['pyannote_time'])
    df['process_time'] = pd.to_numeric(df['process_time'])

    # Now you can calculate the mean processing time for each model
    mean_times = df[['whisper_time', 'pyannote_time', 'process_time']].mean()

    # Plot the mean processing times
    mean_times.plot(kind='bar')

    # Set the title of the figure to include the log_entry_label and log_audio_file
    plt.title(f'Mean Processing Time by Model\n{df["log_audio_file"].iloc[-1]}')

    plt.ylabel('Time (s)')
    plt.xticks(rotation=23)  # Rotate x-axis labels
    plt.tick_params(axis='x', labelsize=8)  # Reduce font size of x-axis labels
    plt.subplots_adjust(bottom=0.25)  # Adjust bottom margin

    # Change the labels of the bars to the model IDs
    ax = plt.gca()
    labels = [df['whisper_model_id'].iloc[-1], df['pyannote_model_id'].iloc[-1], df['llm_model_id'].iloc[-1]]
    ax.set_xticklabels(labels)

    # Ensure the figures directory exists
    if not os.path.exists('logs/figures'):
        os.makedirs('logs/figures')

    # Save the figure
    plt.savefig(f'logs/figures/figure_{df["index"].iloc[-1]}.png')