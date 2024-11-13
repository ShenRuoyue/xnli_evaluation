import os
import csv
import json
import numpy as np
from tqdm import tqdm
import datetime
import torch
import transformers
import multiprocessing
from functools import partial
from accelerate import PartialState
from accelerate.utils import gather_object
from accelerate import Accelerator
from sklearn.metrics import accuracy_score
from transformers.data.processors.utils import DataProcessor, InputExample
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM
from torch.utils.data import DataLoader, Dataset

class XnliDataset(Dataset):
    def __init__(self, data_dir, language):
        self.language = language
        self.data_dir = data_dir
        if self.language == 'ja':
            self.examples = self.load_jnli(data_dir)
        else:
            self.examples = self.load_xnli(data_dir)

    def load_plain_text(self):
        if self.language == 'ja':
            with open(os.path.join(self.data_dir, "jnli_valid-v1.1.json")) as f:
                lines = [json.loads(l) for l in f.readlines()]
            sentences, targets = [], []
            for i, line in enumerate(lines):
                sentences.append((line["sentence1"], line["sentence2"]))
                targets.append(line['label'])
        else:
            lines = self._read_tsv(os.path.join(self.data_dir, "xnli.test.tsv"))
            sentences, targets = [], []
            for i, line in enumerate(lines):
                if i == 0:
                    continue
                language = line[0]
                if language != self.language:
                    continue
                sentences.append((line[6], line[7]))
                targets.append(line[1])
        return sentences, targets

    def load_jnli(self, data_dir):
        with open(os.path.join(data_dir, "jnli_valid-v1.1.json")) as f:
            lines = [json.loads(l) for l in f.readlines()]
        examples = []
        # label_trans = { "entailment": "含意", "neutral": "中立", "contradiction": "矛盾"}
        for i, line in enumerate(lines):
            # prompt = f'Sentence 1: {line["sentence1"]} \nSentence 2: {line["sentence2"]} \nIs the two sentence entailment, neutral, or contradiction?\nAnswer from "entailment", "neutral", and "contradiction".\nAnswer:'
            # prompt = f'文 1: {line["sentence1"]} \n文 2: {line["sentence2"]} \n2 つの文は含意、中立、または矛盾ですか?\n答えは「含意」、「中立」、「矛盾」から選べます。\n答え:'
            prompt = f'### 指示:\n前提と仮説の関係をentailment、contradiction、neutralの中から回答してください。それ以外には何も含めないことを厳守してください。\n\n制約：\n- 前提から仮説が、論理的知識や常識的知識を用いて導出可能である場合はentailmentと出力\n- 前提と仮説が両立しえない場合はcontradictionと出力\n- そのいずれでもない場合はneutralと出力\n\n\n### 入力:\n前提：{line["sentence1"]}\n仮説：{line["sentence2"]}\n\n### 応答:'
            # examples.append({'guid': line['sentence_pair_id'], 'prompt': prompt, 'label': label_trans[line['label']]})
            examples.append({'guid': line['sentence_pair_id'], 'prompt': prompt, 'label': line['label']})
        return examples


    def load_xnli(self, data_dir):
        lines = self._read_tsv(os.path.join(data_dir, "xnli.test.tsv"))
        examples = []
        for i, line in enumerate(lines):
            if i == 0:
                continue
            language = line[0]
            if language != self.language:
                continue
            guid = f"test-{i}"
            text_a = line[6]
            text_b = line[7]
            label = line[1]
            if not isinstance(text_a, str):
                raise TypeError(f"Training input {text_a} is not a string")
            if not isinstance(text_b, str):
                raise TypeError(f"Training input {text_b} is not a string")
            if not isinstance(label, str):
                raise TypeError(f"Training label {label} is not a string")
            prompt = f'Sentence 1: {text_a} \nSentence 2: {text_b} \nIs the two sentence entailment, neutral, or contradiction?\nAnswer from "entailment", "neutral", and "contradiction".\nAnswer: '
            examples.append({'guid': guid, 'prompt': prompt, 'label': label})
        return examples
        
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                lines.append(line)
            return lines

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


def load_model(model_name, device='cuda'):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map=device,
        torch_dtype=torch.bfloat16,
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    return model, tokenizer


def complete_prompt(lang, sentences):
    text_a, text_b = sentences
    if lang != 'ja':
        # prompt = f'Sentence 1: {text_a} \nSentence 2: {text_b} \nIs the two sentence entailment, neutral, or contradiction?\nAnswer from "entailment", "neutral", and "contradiction".\nAnswer: '
        # prompt = f'### Instructions:\nPlease choose the relationship between the premise and the hypothesis from among entailment, contradiction, and neutral. Do not include anything else.\n\nConstraints:\n- If the hypothesis can be derived from the premise using logical and common sense knowledge, output "entailment".\n- If the premise and the hypothesis are incompatible, output "contradiction".\n- If neither of these is true, output "neutral".\n\n\n### Input:\nPremise: {text_a}\nHypothesis: {text_b}\n\n### Answer: '
        # prompt = f'Sentence 1: {text_a} \nSentence 2: {text_b} \nDetermine the relationship between the two sentences. Is the two sentence entailment, neutral, or contradiction?\nRespond with one word from the following options: "entailment", "neutral", or "contradiction".\nOutput: '
        prompt = f'Sentence 1: {text_a} \nSentence 2: {text_b} \nIs the two sentence entailment, neutral, or contradiction?\nAnswer from "entailment", "neutral", and "contradiction".\nAnswer: '
    else:
        prompt = f'### 指示:\n前提と仮説の関係をentailment、contradiction、neutralの中から回答してください。それ以外には何も含めないことを厳守してください。\n\n制約：\n- 前提から仮説が、論理的知識や常識的知識を用いて導出可能である場合はentailmentと出力\n- 前提と仮説が両立しえない場合はcontradictionと出力\n- そのいずれでもない場合はneutralと出力\n\n\n### 入力:\n前提：{text_a}\n仮説：{text_b}\n\n### 応答:'
    return prompt

def post_process(outputs, prompts):
    answers = []
    for output, prompt in zip(outputs, prompts):
        answer = output.split(prompt)[-1].strip().split('\n')[0].split('<')[0].lower().replace('answer', '')
        answer = ''.join(x for x in answer if x.islower())
        if answer not in ['entailment', 'neutral', 'contradiction']:
            print(output.split(prompt)[-1].strip(), '|', answer)
        answers.append(answer)
    # import pdb; pdb.set_trace()
    return answers

def distributed_eval():
    distributed_state = PartialState()

    # Tokenizer and model load
    model_name = 'google/gemma-2-2b-it'
    # model_name = 'google/gemma-2-2b-jpn-it'
    # model_name = 'meta-llama/Meta-Llama-3-8B'
    # model_name = 'tokyotech-llm/Llama-3-Swallow-8B-v0.1'
    # model_name = '/app/mergekit/output_models_mergekit/linear_gemma_0.1'
    model, tokenizer = load_model(model_name, device=distributed_state.device)


    # Data load
    language = 'de'
    sentences, targets = XnliDataset(language=language, data_dir='/app/xnli_evaluation/').load_plain_text()

    # Preprocess
    # You can change the batch size depending on your GPU RAM
    batch_size = 64
    # We set it to 8 since it is better for some hardware. More information here https://github.com/huggingface/tokenizers/issues/991
    pad_to_multiple_of = 8

    with multiprocessing.Pool(processes=4) as p:
        prompts = p.map(partial(complete_prompt, language), sentences)
    # print(len(prompts))
    # import pdb; pdb.set_trace()
    # Split into batches
    formatted_prompts = [(prompts[i: i + batch_size], targets[i: i + batch_size]) for i in range(0, len(prompts), batch_size)]
    # batched_targets = [targets[i: i + batch_size] for i in range(0, len(prompts), batch_size)]

    # Distributed inference
    completions_per_process = []
    accuracies = []
    with distributed_state.split_between_processes(formatted_prompts, apply_padding=True) as batched_prompts:
        with tqdm(total=len(batched_prompts), desc="inference progress") as pbar:
            for batch in batched_prompts:
                # Move the batch to the device
                batched_prompts, batched_targets = batch
                # import pdb; pdb.set_trace()
                tokenized_batch = tokenizer(batched_prompts, padding=True, pad_to_multiple_of=pad_to_multiple_of, return_tensors="pt")
                tokenized_batch = tokenized_batch.to(distributed_state.device)
                # We generate the text, decode it and add it to the list completions_per_process
                outputs = model.generate(**tokenized_batch,
                    max_new_tokens=32,
                    pad_token_id=tokenizer.eos_token_id,
                )
                generated_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)
                answers = post_process(generated_text, batched_prompts)
                accuracy = accuracy_score(answers, batched_targets)
                accuracies.append(accuracy)
                # completions_per_process.extend(answers)
                pbar.update(1)

    # We are gathering string, so we need to use gather_object.
    # If you need to gather tensors, you can use gather from accelerate.utils
    # completions_gather = gather_object(completions_per_process)
    # results = completions_gather[: len(prompts)] 

    if distributed_state.is_main_process:
        print(f'{model_name} Accuracy on language {language}: ', np.mean(accuracies))
        # print(f'No order accuracy: ', accuracy_score(results, targets))


if __name__ == '__main__':
    distributed_eval()
