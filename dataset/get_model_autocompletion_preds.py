import os
import tqdm
import json
import datasets
import traceback
import pandas as pd
from abc import ABC, abstractmethod
from argparse import ArgumentParser

from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed


def get_deepseek_context_prompt(lang, context_list, top=15):
    context_prompt_list = []
    for context_file_path, context_file_code in context_list[:top]:
        context_prompt_list.append(f"#{context_file_path}\n{context_file_code}")
    return "\n".join(context_prompt_list)


def get_string_as_per_token(s: str, token: int, take_prefix = True):
    max_chars = token * 4
    return s[:max_chars] if take_prefix else s[-max_chars:]


def get_common_prefix_suffix_budget(max_tokens, context_items):
    if context_items is None or len(context_items)==0:
        prefix_budget = int(0.6 * max_tokens)
        suffix_budget = int(0.4 * max_tokens)
    else:
        prefix_budget = int(0.5 * max_tokens)
        suffix_budget = int(0.2 * max_tokens)

    return prefix_budget, suffix_budget


def get_prefix_and_suffix(max_tokens, fim_prefix_code, fim_suffix_code, context_items):
    prefix_budget, suffix_budget = get_common_prefix_suffix_budget(max_tokens, context_items)
    prefix_code = get_string_as_per_token(fim_prefix_code, prefix_budget, take_prefix=False)
    suffix_code = get_string_as_per_token(fim_suffix_code, suffix_budget)
    return prefix_code, suffix_code

def get_context_items_tuple(context_items):
    return  [
        (item['filename'], item['retrieved_chunk'])
        for item in context_items
    ]


def get_prompt_for_infilling_code(
    max_context_tokens, fim_prefix_code, fim_suffix_code, repo_name, file_name, context_items, lang,
    context_prompt_callback,
    next_prompt_callback
    ):
    prefix_code, suffix_code = get_prefix_and_suffix(max_context_tokens, fim_prefix_code, fim_suffix_code, context_items)
    context_items = get_context_items_tuple(context_items)
    prompt = ""
    for i in range(len(context_items)+1):
        context_prompt = context_prompt_callback(lang, context_items[:i])
        next_prompt = next_prompt_callback(context_prompt, prefix_code, suffix_code)
        if len(next_prompt) >= max_context_tokens*4:
            break
        prompt = next_prompt
    return prompt


def get_deepseek_prompt(max_context_tokens, fim_prefix_code, fim_suffix_code, repo_name, file_name, context_items, lang, get_left_to_right):
    def next_prompt_cb(context_prompt, prefix_code, suffix_code):
        if get_left_to_right:
            return (f"<repo_name>{repo_name}\n{context_prompt}\n#{file_name}\n", f"<｜fim▁begin｜>{prefix_code}<｜fim▁hole｜>{suffix_code}<｜fim▁end｜>")
        else:
            return f"<repo_name>{repo_name}\n{context_prompt}\n#{file_name}\n<｜fim▁begin｜>{prefix_code}<｜fim▁hole｜>{suffix_code}<｜fim▁end｜>"

    return get_prompt_for_infilling_code(
        max_context_tokens, fim_prefix_code, fim_suffix_code, repo_name, file_name, context_items, lang,
        get_deepseek_context_prompt,
        next_prompt_cb
    )


class BaseInfillPromptProvider(ABC):
    @abstractmethod
    def get_provider_name(self) -> str:
        pass

    @abstractmethod
    def get_prompt(self,  prefix: str, suffix: str, repo_name: str, file_name: str, context_items: list, kwargs: dict):
        pass

class DeepSeekModelInfillPromptProvider(BaseInfillPromptProvider):
    def get_provider_name(self) -> str:
        return "deepseek_model"

    def get_prompt(self, prefix, suffix, repo_name, file_name, context_items, kwargs = None) -> str:
        return get_deepseek_prompt(50_000, prefix, suffix, repo_name, file_name, context_items, kwargs['lang'], get_left_to_right=False)

    def get_left_to_right_completion(self, prefix, suffix, repo_name, file_name, context_items, kwargs = None) -> str:
        return get_deepseek_prompt(50_000, prefix, suffix, repo_name, file_name, context_items, kwargs['lang'], get_left_to_right=True)


def get_fireworks_code_completions(prompt, **kwargs):
    kwargs = kwargs.copy()
    api_key = kwargs['api_key']
    del kwargs['api_key']

    Tries = 10
    for _ in range(Tries):
        try:
            with OpenAI(
                base_url = "https://api.fireworks.ai/inference/v1",
                api_key = api_key
            ) as client:
                params_dict = {
                    'prompt': prompt,
                    **kwargs,
                }
                response = client.completions.create(**params_dict)
                return response.choices[0].text
        except Exception as e:
            traceback.print_exc()
            print(f"Unknown error: {e}\n For model: {kwargs['model']}, tokens: {len(prompt)/4}")
            raise e


def generate_infill_eval_completions(inputs, dataset: datasets.Dataset):
    is_infill = True
    lang, completion = inputs['lang'], inputs['completion']

    infill_prompt_obj, api, kwargs = completion
    template_name = infill_prompt_obj.get_provider_name()

    llm_generation_base_path = os.path.join(os.environ['HOME'], 'infill-eval')

    all_datapoint_to_get = []
    prediction_path = os.path.join(
        llm_generation_base_path, lang, os.path.basename(kwargs['model']),
        template_name,
        f"{kwargs['temperature']}-{kwargs['max_tokens']}-{kwargs.get('top_p', 'none')}"
    )
    for index, datapoint in enumerate(dataset):
        problem_name = f"problem_{index}.json"
        file_save_path = os.path.join(
            prediction_path,
            problem_name
        )

        if not is_infill or datapoint['suffix'] is None:
            datapoint['suffix'] = ""

        context = ""
        prompt = infill_prompt_obj.get_prompt(
            datapoint['prefix'],
            datapoint['suffix'],
            datapoint['repo_url'],
            datapoint['import filepath'],
            context,
            {'lang': lang}
        )

        # Check for empty prompt
        if isinstance(prompt, str) and len(prompt) <= 0:
            print(f"Skipping LLM call for empty prompt on lang: {lang}, template: {template_name}, model: {kwargs['model']}, problem: {problem_name}")
            # Directly return empty output structure without calling the LLM
            empty_output = (
                file_save_path,
                api,
                "",
                kwargs,
                datapoint,
                lang,
                template_name,
                problem_name
            )
            all_datapoint_to_get.append(empty_output)
            continue

        all_datapoint_to_get.append((
            file_save_path,
            api,
            prompt,
            kwargs,
            datapoint,
            lang,
            template_name,
            problem_name
        ))

    all_completions = []
    with ThreadPoolExecutor(max_workers=min(5, os.cpu_count()-1)) as executor:
        futures = [executor.submit(save_infill_evaluation_predictions_if_not_empty, data) for data in all_datapoint_to_get]
        for output in tqdm.tqdm(as_completed(futures), total=len(futures), desc=f"Calling LLM for prompt completion for model {kwargs['model']} on lang: {lang}:"):
            all_completions.append(output.result())

    return all_completions


def post_process_infill_code_completions_based_on_model(model, code):
    if model=='gpt-4o':
        if code.startswith('<new_code>'):
            new_code = code[len('<new_code>'):]
        if code.endswith('<new_code>'):
            new_code = code[:-len('<new_code>')]

        code = '\n'.join(code.split('\n')[1:]).strip('`') \
                if code.startswith('```') \
                else code

        if code[0] == '\n':
            code = code[1:]

        return code

    return code


def save_infill_evaluation_predictions(query):
    file_save_path, api, prompt, kwargs, datapoint, lang, template_name, problem_name = query
    os.makedirs(os.path.dirname(file_save_path), exist_ok=True)

    if os.path.isfile(file_save_path):
        try:
            with open(file_save_path, 'r') as f:
                data = json.load(f)

            if len(data['llm-predicted_code']) > 0:
                predicted_code = data['llm-predicted_code']
            else:
                predicted_code = api(prompt, **kwargs)
        except:
            predicted_code = api(prompt, **kwargs)
    else:
        predicted_code = api(prompt, **kwargs)

    ground_truth_code = datapoint['ground_truth']
    predicted_code = post_process_infill_code_completions_based_on_model(
        kwargs['model'],
        predicted_code
    )

    # is_first_line_prefix = predicted_code.startswith(ground_truth_code)
    is_first_line_prefix = predicted_code.strip().startswith(ground_truth_code.strip())
    # is_first_line_prefix = predicted_code.strip()==ground_truth_code.strip()

    data = {
        **datapoint,
        'kwargs': kwargs,
        'model': kwargs['model'],
        'llm-prompt': prompt,
        'llm-predicted_code': predicted_code,
        'is_first_line_prefix': is_first_line_prefix,
        'problem_name': problem_name,
    }
    with open(file_save_path, 'w') as f:
        json.dump(data, f, indent=4)
    return data


def save_infill_evaluation_predictions_if_not_empty(query):
    file_save_path, api, prompt, kwargs, datapoint, lang, template_name, problem_name = query
    if not prompt:
        return {
            **datapoint,
            'kwargs': kwargs,
            'model': kwargs['model'],
            'llm-prompt': "",
            'llm-predicted_code': "",
            'is_first_line_prefix': False
        }

    return save_infill_evaluation_predictions(query)


def load_csv_and_pre_process_convert_to_hf_ds(csv_path):
    df = pd.read_csv(csv_path)

    df = df[["repo_url", "commit", "import filepath", "import name", "prefix", "middle", "suffix", "strategy"]]
    df = df[df["middle"]!=""]
    df = df.drop_duplicates().dropna(subset=["prefix", "middle", "suffix"]).dropna(subset=["strategy"])
    print(f"AFTER POST-PROCESS of Load: {df.shape}")

    df = df.rename(columns={'middle': 'ground_truth'})
    dataset = datasets.Dataset.from_pandas(df)

    return dataset

def main():
    parser = ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default=None)
    parser.add_argument("--evals_save_path", type=str, default=None)
    parser.add_argument("--FIREWORKS_API_KEY", type=str, default=None)
    args = parser.parse_args()
    dataset_path = args.dataset_path
    evals_save_path = args.evals_save_path
    fw_api_key = args.FIREWORKS_API_KEY

    all_completion_models = [
        (DeepSeekModelInfillPromptProvider(), get_fireworks_code_completions,
         {'api_key': fw_api_key, 'model': 'accounts/sourcegraph/models/deepseek-coder-v2-lite-base', 'temperature': 0.2,
          'max_tokens': 512}),
        (DeepSeekModelInfillPromptProvider(), get_fireworks_code_completions,
         {'api_key': fw_api_key, 'model': 'accounts/sourcegraph/models/deek-seek-v2-code-gym-neg-v1-ep-2', 'temperature': 0.2,
          'max_tokens': 512}),
        (DeepSeekModelInfillPromptProvider(), get_fireworks_code_completions,
         {'api_key': fw_api_key, 'model': 'accounts/sourcegraph/models/deek-seek-v2-code-gym-neg-v1-ep-3',
          'temperature': 0.2, 'max_tokens': 512}),
        (DeepSeekModelInfillPromptProvider(), get_fireworks_code_completions,
         {'api_key': fw_api_key, 'model': 'accounts/sourcegraph/models/deek-seek-v2-code-gym-neg-v1-ep-5', 'temperature': 0.2,
          'max_tokens': 512}),
    ]

    dataset: datasets.Dataset = load_csv_and_pre_process_convert_to_hf_ds(dataset_path)

    lang = 'ts'
    all_completions_data = []
    new_all_completion_models = all_completion_models.copy()
    for completion_data in new_all_completion_models:
        all_completions_data.append({
            'lang': lang,
            'completion': (completion_data[0], completion_data[1], completion_data[2]),
        })
        print(f"\n #### Completion for the model {completion_data[2]['model']} start: ##### \n")

        # High Qps results in fireworks complaining about traffic
        all_generations_artifacts = []
        with ThreadPoolExecutor(max_workers=min(1, os.cpu_count()-1)) as executor:
            futures = [executor.submit(generate_infill_eval_completions, completion_data, dataset) for completion_data in all_completions_data]
            for output in tqdm.tqdm(as_completed(futures), total=len(futures), desc="Generating predictions report:"):
                all_generations_artifacts.extend(output.result())

        df = pd.DataFrame(all_generations_artifacts)
        print(f"#### Completion for the model {completion_data[2]['model']} End ##### \n")
        model_name = completion_data[2]['model'].split("accounts/sourcegraph/models/")[-1]

        evals_save_path = os.path.join(evals_save_path, 'infill_eval_results')
        os.makedirs(evals_save_path, exist_ok=True)
        df.to_csv(os.path.join(evals_save_path, f'infill_eval_results_V1_{model_name}.csv'), index=False)

        df = pd.read_csv(os.path.join(evals_save_path, f'infill_eval_results_V1_{model_name}.csv'))
        df['model'] = df['model'].apply(lambda x : x.replace('accounts/fireworks/models/', '').replace('accounts/sourcegraph/models/', ''))
        df['ext'] = df['import filepath'].str.split('.').str[-1]
        df = pd.pivot_table(df, index=['model'], values='is_first_line_prefix', columns='ext')
        df = df.round(2)
        print(df)


if __name__ == "__main__":
    main()

