"""
    This script takes 1% of the corpus, chosen at random, and 
    1: Splits the files into 90/10 train/test datasets
    2: Extracts a set of all unique characters stored in a file named 'vocab.txt'
    3: Spawns PROCESSES concurrent processes each of which opens files and appends
       the text either to a training or test data file (FIXME: I am suspicious of
       how the concurrency works on this one)
"""


import os
import lzma
from tqdm import tqdm
import concurrent.futures
import random

PROCESSES = 4
RATE = 0.05
SPLIT = 0.9

folder_path = "data/subsets/openwebtext"
output_file_train = "data/output_train.txt"
output_file_val = "data/output_val.txt"
vocab_file = "data/vocab.txt"


def process_file(args):
    """ Reads text from input file, appends it to an output file """
    directory, filename, output_file, vocab = args
    file_path = os.path.join(directory, filename)
    with lzma.open(file_path, "rt", encoding="utf-8") as infile:
        text = infile.read()
    with open(output_file, "a", encoding="utf-8") as outfile:
        outfile.write(text)
    characters = set(text)
    return characters

def xz_files_in_dir(directory):
    """ Returns a list of all the .xz files in the specified directory """
    files = []
    for filename in os.listdir(directory):
        is_file = os.path.isfile(os.path.join(directory, filename))
        if filename.endswith('.xz') and is_file:
            files.append(filename)
    return files

def process_files_in_parallel(files, folder_path, output_file):
    """ Launches multiple Processes that ingest the data in parallel """
    vocab = set()
    with concurrent.futures.ProcessPoolExecutor(max_workers=PROCESSES) as executor:
        args = [(folder_path, filename, output_file, vocab) for filename in files]
        for characters in tqdm(executor.map(process_file, args), total=len(files)):
            vocab.update(characters)
    return vocab

def ingest_data():
    """ The main entry porint for this script """
    files = xz_files_in_dir(folder_path)
    total_files = len(files)
    split_index = int(total_files * SPLIT)  # 90% for training
    files_train = files[:split_index]
    files_val = files[split_index:]

    # Sampling a hundredth of the files for each split
    sample_rate = RATE
    files_train_sampled = random.sample(files_train, max(1, int(len(files_train) * sample_rate)))
    files_val_sampled = random.sample(files_val, max(1, int(len(files_val) * sample_rate)))

    # Ensure output files are empty before appending
    open(output_file_train, 'w').close()
    open(output_file_val, 'w').close()

    # Process the sampled training files
    vocab_train = process_files_in_parallel(files_train_sampled, folder_path, output_file_train)

    # Process the sampled validation files
    vocab_val = process_files_in_parallel(files_val_sampled, folder_path, output_file_val)

    # Combine vocabularies (if needed) and write to vocab.txt
    vocab = vocab_train.union(vocab_val)
    with open(vocab_file, "w", encoding="utf-8") as vfile:
        for char in sorted(vocab):
            vfile.write(char + '\n')

if __name__ == "__main__":
    ingest_data()
