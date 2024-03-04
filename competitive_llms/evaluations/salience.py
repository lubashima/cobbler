import sys
import os

import nltk
nltk.download('punkt')

from nltk.tokenize import word_tokenize
from .utils import check_result_dir

import json

def count_tokens(input_string):
    tokens = word_tokenize(input_string)
    num_tokens = len(tokens)
    return num_tokens

def organize_data(file_path):
    nested_arrays = []
    current_array = []

    with open(file_path, 'r') as file:
        lines = file.readlines()

        for line in lines:
            line = line.strip()

            if not line:
                # Treat line breaks as separate arrays
                if current_array:
                    nested_arrays.append(current_array)
                    current_array = []
                continue

            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                # Skip lines that are not valid JSON objects
                continue

            current_array.append(data)

    if current_array:
        # Append the last array if it's not empty
        nested_arrays.append(current_array)

    return nested_arrays


def find_length_bias(pairs, responses):
    shorter_bias = 0
    longer_bias = 0
    valid_responses = 0
    
    for idx, samples in enumerate(pairs):
        for sample in samples:
            winner = sample['model']
            combo = sample['combination']
            # No order bias -> can't have length bias if there is order bias
            if winner != "Invalid response" and "fo bias" not in combo and "lo bias" not in combo:
                if len(responses[combo[0]][idx]) > len(responses[combo[1]][idx]):
                    if winner == combo[0]:
                        longer_bias += 1
                    else:
                        shorter_bias += 1
                else:
                    if winner == combo[1]:
                        longer_bias +=1
                    else:
                        shorter_bias += 1
                valid_responses += 1
                
    return valid_responses, shorter_bias, longer_bias

def read_json_file(file):
    with open(file, "r") as r:
        response = r.read()
        response = response.replace('\n', '')
        response = response.replace('}{', '},{')
        response = "[" + response + "]"
        return json.loads(response)

# def extract_valid_responses(path):
#     with open(path) as file:
#         lines = file.readlines()

#     valid_responses_line = [line for line in lines if 'Valid responses' in line]
#     if valid_responses_line:
#         valid_responses = valid_responses_line[0].split(':')[1].strip()
#         return int(valid_responses)
#         # print(valid_responses)
#     else:
#         print("Valid responses line not found in the file.")
    
#     return None

def organize_data(file_path):
    nested_arrays = []
    current_array = []

    with open(file_path, 'r') as file:
        lines = file.readlines()

        for line in lines:
            line = line.strip()

            if not line:
                # Treat line breaks as separate arrays
                if current_array:
                    nested_arrays.append(current_array)
                    current_array = []
                continue

            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                # Skip lines that are not valid JSON objects
                continue

            current_array.append(data)

    if current_array:
        # Append the last array if it's not empty
        nested_arrays.append(current_array)

    return nested_arrays

def evaluate_salience(results_dict):
    salience_results_dict = {}
    valid_responses = results_dict['order']['stats']['valid_responses_count']
    this_file_path = os.path.realpath(os.path.dirname(os.path.realpath(__file__)))
    responses = read_json_file(os.path.join(this_file_path, "../n15_responses/full_n15_model_generations_test.json"))[0]
    
    pairs = results_dict['order']['true_order']
    lb = find_length_bias(pairs, responses)

    valid = lb[0]
    short = lb[1]
    long = lb[2]
    salience_results_dict['valid_responses_count'] = valid
    salience_results_dict['retention_percentage'] = valid / (valid_responses + 1e-6)
    salience_results_dict['short_bias'] = short / valid
    salience_results_dict['long_bias'] = long / valid

    return salience_results_dict
    
def main(arg1, arg2):
    evaluate_salience(arg1, arg2)
    
    
if __name__ == '__main__':
    arg1 = sys.argv[1]
    # Include "human" response / ground truth
    if len(sys.argv) > 2:
        arg2 = sys.argv[2]
        print(f"Evaluating _size experiments")
    else:
        arg2 = None
    main(arg1,arg2)