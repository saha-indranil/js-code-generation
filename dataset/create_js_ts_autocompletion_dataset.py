import os
import re
import sys
import tqdm
import random
import logging
import subprocess
import numpy as np
import pandas as pd
from argparse import ArgumentParser
from setup_logger import setup_logger
import tree_sitter_utils.utils as tsutils


logger = setup_logger(name="ds_creation_logger", log_file="/var/log/gym/ds_creation_log_file.log", level=logging.DEBUG)

JS_TS_AUTOCOMPLETION_DS_PATH = "/var/log/gym/js_ts_autocompletion_dataset.csv"

parser = ArgumentParser()
parser.add_argument("--repo_path", type=str, help="Path to the project.")
parser.add_argument("--map_path", type=str, help="Path to the JS/TS function/class->testcase mapping csv.")


def get_repo_url_and_commit_hash(repo_path):
    url = subprocess.check_output(["bash", "-c", f"git config --global --add safe.directory {repo_path} && cd {repo_path} && git config --get remote.origin.url"], encoding="utf-8")
    commit = subprocess.check_output(["bash", "-c", f"git config --global --add safe.directory {repo_path} && cd {repo_path} && git rev-parse HEAD"], encoding="utf-8")

    return url, commit


def _filter_df_based_on_fields_or_null(df, field):
    q1, q3 = df[field].quantile([0.05, 0.95])
    df[field] = np.where((df[field].isna()) | ( (df[field] >= q1) & (df[field] <= q3) ), df[field], None)
    return df


class PrepareFIMDataset:

    def __init__(self, repo_df):
        self.repo_df = repo_df

    def get_base_dataset_with_metadata(self):
        return self.repo_df
    
    def additional_clean_df(self, df):
        df['cleaned_code'] = df['identifier-body-content']
        df['function_block_data'] = df['metadata'].apply(lambda x : x['function_block_data'])
        df['if_block_data'] = df['metadata'].apply(lambda x : x['if_block_data'])
        df['call_expression'] = df['metadata'].apply(lambda x : x['call_exp_block_data'])
        df['assignment_block_data'] = df['metadata'].apply(lambda x : x['assignment_block_data'])
        df['num_lines_cleaned_code'] = df['cleaned_code'].apply(lambda x : len(x.split('\n')))
        return df

    def post_process_fim_datapoint(row, fim_datapoint):
        if fim_datapoint is None:
            return fim_datapoint
            
        global_prefix = row["file-prefix-to-identifier"]+row["identifier-header-content"]
        global_suffix = row["file-suffix-to-identifier"]

        fim_datapoint["prefix"] = global_prefix + fim_datapoint["prefix"]
        fim_datapoint["suffix"] = fim_datapoint["suffix"] + global_suffix

        return fim_datapoint

    def get_fim_datapoint(row):
        fim_datapoint = {}
        fim_datapoint['call_expression'] = PrepareFIMDataset.call_expression_based_fim_sampling(row) # 
        fim_datapoint['if_block_data'] = PrepareFIMDataset.if_condition_based_fim_sampling(row) # 
        fim_datapoint['function_block_data'] = PrepareFIMDataset.function_based_fim_sampling(row) # 
        fim_datapoint['random'] = PrepareFIMDataset.random_fim_sampling(row) # 
        fim_datapoint['assigment_based'] = PrepareFIMDataset.assigment_based_fim_sampling(row) # 

        # post-process
        fim_datapoint['call_expression'] = PrepareFIMDataset.post_process_fim_datapoint(row, fim_datapoint['call_expression'])
        fim_datapoint['if_block_data'] = PrepareFIMDataset.post_process_fim_datapoint(row, fim_datapoint['if_block_data'])
        fim_datapoint['function_block_data'] = PrepareFIMDataset.post_process_fim_datapoint(row, fim_datapoint['function_block_data'])
        fim_datapoint['random'] = PrepareFIMDataset.post_process_fim_datapoint(row, fim_datapoint['random'])
        fim_datapoint['assigment_based'] = PrepareFIMDataset.post_process_fim_datapoint(row, fim_datapoint['assigment_based'])

        return fim_datapoint

    def function_based_fim_sampling(row):
        if len(row['function_block_data'])==0:
            return None
        random_no = random.random()
        some_function_to_complete = random.choice(row['function_block_data'])
        code_bytes = row['cleaned_code'].encode('utf-8')
        prefix = code_bytes[:some_function_to_complete['byte_range'][0]].decode('utf-8')
        suffix = code_bytes[some_function_to_complete['byte_range'][1]:].decode('utf-8')
        middle = some_function_to_complete['text']

        if random_no < 0.30:
            some_point_in_middle = PrepareFIMDataset.find_alphanumeric_to_left(middle, random.randint(0, len(middle)-1))
            prefix+=middle[:some_point_in_middle]
            middle = middle[some_point_in_middle:]
            return {
                'prefix': prefix,
                'suffix': suffix,
                'middle': middle,
                'strategy': 'function_block-function_prefixed'
            }
        elif random_no < 0.70:
            middle_lines = middle.split('\n')
            middle_selected_line = random.randint(0, len(middle_lines)-1)
            middle_pointer = PrepareFIMDataset.find_alphanumeric_to_right(middle, PrepareFIMDataset.find_nth_occurrence(middle, '\n', middle_selected_line+1))
            prefix+=middle[:middle_pointer]
            middle = middle[middle_pointer:]
            return {
                'prefix': prefix,
                'suffix': suffix,
                'middle': middle,
                'strategy': 'function_block-function_prefix_line'
            }
        else:
            np_rng = np.random.RandomState()
            boundaries = list(np_rng.randint(low=0, high=len(middle)-1, size=2))
            boundaries.sort()
            prefix+=middle[:boundaries[0]]
            suffix=middle[boundaries[1]:]+suffix
            middle=middle[boundaries[0]:boundaries[1]]
            return {
                'prefix': prefix,
                'suffix': suffix,
                'middle': middle,
                'strategy': 'function_block-random_span'
            }

    def if_condition_based_fim_sampling(row):
        if len(row['if_block_data'])==0:
            return None

        condition_based_data = random.choice(row['if_block_data'])
        return PrepareFIMDataset.get_fim_as_per_range(row['cleaned_code'], condition_based_data['byte_range'][0], condition_based_data['byte_range'][1], 'if_condition')

    def call_expression_based_fim_sampling(row):
        if len(row['call_expression'])==0:
            return None
        condition_based_data = random.choice(row['call_expression'])
        return PrepareFIMDataset.get_fim_as_per_range(row['cleaned_code'], condition_based_data['byte_range'][0], condition_based_data['byte_range'][1], 'call_expression')

    def assigment_based_fim_sampling(row):
        if len(row['assignment_block_data'])==0:
            return None
        condition_based_data = random.choice(row['assignment_block_data'])
        return PrepareFIMDataset.get_fim_as_per_range(row['cleaned_code'], condition_based_data['byte_range'][0], condition_based_data['byte_range'][1], 'assignment_statement')

    def random_fim_sampling(row):
        sample = row['cleaned_code']
        no_lines = sample.count("\n")
        if no_lines<4:
            return {
            'prefix': "",
            'suffix': "",
            'middle': "",
            'strategy': 'random'
        }
        
        np_rng = np.random.RandomState()
        first_boundary = list(np_rng.randint(low=2, high=no_lines, size=1))[0]
        second_boundary = list(np_rng.randint(low=max(3, first_boundary-5), high=min(no_lines, first_boundary+5), size=1))[0]
        boundaries = [first_boundary, second_boundary]
        boundaries.sort()
        
        prefix = "\n".join(sample.split("\n")[: boundaries[0]])
        middle = "\n".join(sample.split("\n")[boundaries[0] : boundaries[1]])
        suffix = "\n".join(sample.split("\n")[boundaries[1] :])
        return {
            'prefix': prefix,
            'suffix': suffix,
            'middle': middle,
            'strategy': 'random'
        }

    def find_alphanumeric_to_left(s, index):
        index = max(0, min(len(s)-1, index))
        while index >= 0:
            if s[index].isalnum():
                return index
            index -= 1
        return 0

    def find_alphanumeric_to_right(s, index):
        index = max(0, min(len(s)-1, index))
        while index <= len(s)-1:
            if s[index].isalnum():
                return index
            index += 1
        return len(s)-1

    def find_nth_occurrence(string, character, n):
        start = -1
        for _ in range(n):
            start = string.find(character, start + 1)
            if start == -1:
                break
        return 0 if start==-1 else start

    def get_fim_as_per_range(cleaned_code, start_byte, end_byte, strategy):
        code_bytes = cleaned_code.encode('utf-8')
        prefix = code_bytes[:start_byte].decode('utf-8')
        suffix = code_bytes[end_byte:].decode('utf-8')
        middle = code_bytes[start_byte:end_byte].decode('utf-8')
        return {
            'prefix': prefix,
            'suffix': suffix,
            'middle': middle,
            'strategy': strategy
        }

    def get_final_fim_datapoint(row, weights):
        lst = [
            'fim_point_call',
            'fim_point_assignment',
            'fim_point_if',
            'fim_point_function',
            'fim_point_random'
        ]
        assert len(weights) == len(lst)
        normalized_weights = np.array([(wei/sum(weights)) for wei in weights])
        normalized_weights = np.cumsum(normalized_weights)
        assert abs(normalized_weights[-1]-1.0) < 1e-5, f"{normalized_weights[-1]}"
        rand_no = random.random()
        for field, weight in zip(lst, normalized_weights):
            if rand_no <= weight and row[field] is not None:
                return row[field]

    def fim_sampling_post_processing(df):
        df['fim_point_call'] = df['fim_datapoint'].apply(lambda x : x['call_expression'])
        df['fim_point_if'] = df['fim_datapoint'].apply(lambda x : x['if_block_data'])
        df['fim_point_function'] = df['fim_datapoint'].apply(lambda x : x['function_block_data'])
        df['fim_point_assignment'] = df['fim_datapoint'].apply(lambda x : x['assigment_based'])
        df['fim_point_random'] = df['fim_datapoint'].apply(lambda x : x['random'])

        df['fim_point_call_length'] = df['fim_point_call'].apply(lambda x : None if x is None else len(x['middle']))
        df['fim_point_if_length'] = df['fim_point_if'].apply(lambda x : None if x is None else len(x['middle']))
        df['fim_point_function_length'] = df['fim_point_function'].apply(lambda x : None if x is None else len(x['middle']))
        df['fim_point_assignment_length'] = df['fim_point_assignment'].apply(lambda x : None if x is None else len(x['middle']))
        df['fim_point_random_length'] = df['fim_point_random'].apply(lambda x : None if x is None else len(x['middle']))

        df = _filter_df_based_on_fields_or_null(df, 'fim_point_call_length')
        df = _filter_df_based_on_fields_or_null(df, 'fim_point_if_length')
        df = _filter_df_based_on_fields_or_null(df, 'fim_point_function_length')
        df = _filter_df_based_on_fields_or_null(df, 'fim_point_assignment_length')
        df = _filter_df_based_on_fields_or_null(df, 'fim_point_random_length')

        df = df.reset_index(drop=True)

        df_point_call = pd.concat([df, pd.json_normalize(df['fim_point_call'])], axis=1)
        df_point_if = pd.concat([df, pd.json_normalize(df['fim_point_if'])], axis=1)
        df_point_function = pd.concat([df, pd.json_normalize(df['fim_point_function'])], axis=1)
        df_point_assignment = pd.concat([df, pd.json_normalize(df['fim_point_assignment'])], axis=1)
        df_point_random = pd.concat([df, pd.json_normalize(df['fim_point_random'])], axis=1)

        df_point_call = df_point_call.dropna(subset=["import filepath", "import name", "prefix", "middle", "suffix", "strategy"])
        df_point_if = df_point_if.dropna(subset=["import filepath", "import name", "prefix", "middle", "suffix", "strategy"])
        df_point_function = df_point_function.dropna(subset=["import filepath", "import name", "prefix", "middle", "suffix", "strategy"])
        df_point_assignment = df_point_assignment.dropna(subset=["import filepath", "import name", "prefix", "middle", "suffix", "strategy"])
        df_point_random = df_point_random.dropna(subset=["import filepath", "import name", "prefix", "middle", "suffix", "strategy"])
        
        final_df = pd.concat([df_point_call, df_point_if, df_point_function, df_point_assignment, df_point_random], axis=0)

        return final_df


def extract_node_data(node):
    return {
        "type": node.type,
        "text": node.text.decode('utf-8'),
        "start_point": node.start_point,
        "end_point": node.end_point,
        "byte_range": node.byte_range
    }


def extract_metadata(row):
    try:
        code = row["identifier-body-content"]
        extension = row["import filepath"].split(".")[-1]
    
        lang = tsutils.get_language_from_extension(extension)
        parser = tsutils.get_parser(lang)
        patterns_to_match = tsutils.get_capture_patterns(lang)
        patterns = tsutils.extract_capture_patterns(
                    bytes(code, "utf-8"),
                    parser,
                    tsutils.get_tree_sitter_lang(lang),
                    patterns_to_match,
                    True
                )
    
        # categorize the patterns
        function_block_data = [extract_node_data(node) for node in patterns.get("function_declaration", []) if len(node.text.decode('utf-8').split('\n')) < 300]
        if_block_data = [extract_node_data(node) for node in patterns.get("if_statement", []) if len(node.text.decode('utf-8').split('\n'))]
        assignment_blocks = [extract_node_data(node) for node in patterns.get("assignment_statement", [])]
        assignment_blocks = sorted(assignment_blocks, key=lambda x: len(x["text"]), reverse=True)
        call_exp_blocks = [extract_node_data(node) for node in patterns.get("call_expression", [])]
        call_exp_blocks = sorted(call_exp_blocks, key=lambda x: len(x["text"]), reverse=True)

        patterns = {"function_block_data": function_block_data, "if_block_data": if_block_data, "assignment_block_data": assignment_blocks, "call_exp_block_data": call_exp_blocks}

    except Exception as e:
        print(f"Error for {row}")
        patterns = {"function_block_data": [], "if_block_data": [], "assignment_block_data": [], "call_exp_block_data": []}
    
    return patterns
    

def extract_metadata_for_df(df):
    df["metadata"] = df.parallel_apply(lambda x: extract_metadata(x), axis=1)
    
    return df
    

def process_test_df(test_df):
    test_df["tests"] = test_df.apply(lambda x: (x["test filepath"], x["test name"]), axis=1)
    test_df = test_df[["import filepath", "import name", "tests"]]
    test_df = test_df.groupby(["import filepath", "import name"]).agg({"tests": list})
    test_df = test_df.reset_index(drop=False)

    return test_df


def create_prefix_middle_suffix_dataset(df):
    fim_ds = PrepareFIMDataset(df)
    df_with_metadata = extract_metadata_for_df(df)

    df_with_metadata_cleaned = fim_ds.additional_clean_df(df_with_metadata)
    df_with_metadata_cleaned['fim_datapoint'] = df.apply(lambda x : PrepareFIMDataset.get_fim_datapoint(x), axis=1)

    final_df = PrepareFIMDataset.fim_sampling_post_processing(df_with_metadata_cleaned)
    final_df = final_df.dropna(subset=["import filepath", "import name"])
    final_df = final_df.dropna(subset=["strategy"])

    return final_df
    

def create_dataset(df):
    df = df.dropna(subset=["identifier-content", "identifier-header-content", "identifier-body-content"])
    
    identifier_df = df.drop(["test filepath", "test name"], axis=1)
    identifier_df = identifier_df.drop_duplicates()
    test_df = df[["import filepath", "import name", "test filepath", "test name"]]
    test_df = test_df.drop_duplicates()
    
    processed_test_df = process_test_df(test_df)
    processed_identifier_df = create_prefix_middle_suffix_dataset(identifier_df)

    final_df = pd.merge(processed_identifier_df, processed_test_df, on=["import filepath", "import name"], how="left")

    return final_df


if __name__ == "__main__":
    args = parser.parse_args()
    repo_path = args.repo_path
    repo_name = [directory for directory in os.listdir(repo_path) if not directory.startswith(".")]
    if len(repo_name) > 1:
        logger.error(f"More than 1 repo was found! found repos - {repo_name}")
        raise ValueError(f"More than 1 repo was found! found repos - {repo_name}")
    repo_path = os.path.join(repo_path, repo_name[0])
    repo_url, repo_commit_hash = get_repo_url_and_commit_hash(repo_path)

    map_path = args.map_path
    map_df = pd.read_csv(map_path)
    map_df = map_df[map_df["valid"]==True]
    ds = create_dataset(map_df, repo_url, repo_commit_hash)
    ds["repo_url"] = repo_url
    ds["commit"] = repo_commit_hash
    ds.to_csv(JS_TS_AUTOCOMPLETION_DS_PATH, index=False)

