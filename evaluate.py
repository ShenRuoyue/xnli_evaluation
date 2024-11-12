import os
import csv
import json
import datetime
import torch
import transformers
from accelerate import PartialState
from accelerate import Accelerator
# import tensorflow as tf
from sklearn.metrics import accuracy_score
from transformers.data.processors.utils import DataProcessor, InputExample
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM
from torch.utils.data import DataLoader, Dataset

class XnliDataset(Dataset):
    def __init__(self, data_dir, language):
        self.language = language

        if self.language == 'ja':
            self.examples = self.load_jnli(data_dir)
        else:
            self.examples = self.load_xnli(data_dir)

    def load_jnli(self, data_dir):
        with open(os.path.join(data_dir, "XNLI-1.0/jnli_valid-v1.1.json")) as f:
            # lines = json.load(f)
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
        lines = self._read_tsv(os.path.join(data_dir, "XNLI-1.0/xnli.test.tsv"))
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
            # self.examples.append({'guid': guid, 'text_a': text_a, 'text_b': text_b, 'label': label})
                # InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
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
        torch_dtype=
        torch.bfloat16,
    )
    # if tokenizer.pad_token is None:
    #     tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    #     model.resize_token_embeddings(len(tokenizer))
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    return model, tokenizer


def execution():
    tic = datetime.datetime.now().replace(microsecond=0)
    # model_name = 'google/gemma-2-2b-it'
    # model_name = 'google/gemma-2-2b-jpn-it'
    # model_name = 'meta-llama/Meta-Llama-3-8B'
    model_name = 'tokyotech-llm/Llama-3-Swallow-8B-v0.1'
    # model_name = '/app/mergekit/output_models_mergekit/linear_gemma'
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    # Spanish(es), German(de), Greek(el), Bulgarian(bg), Russian(ru), Turkish(tr), Arabic(ar), Vietnamese(vi), Thai(th), Chinese(zh), Hindi(hi), Swahili(sw) and Urdu(ur)
    langs = ['ar','bg','de','el','es','fr','hi','ru','sw','th','tr','ur','vi','zh', 'en']
    language = 'ja'
    dataset = XnliDataset(language=language, data_dir='/app/evaluation/xnli/')
    dataloader = DataLoader(dataset, batch_size=4, shuffle=False)
    model, tokenizer = load_model(model_name, device)
    count = 0
    answers = []
    gts = []

    for idx, x in enumerate(dataloader):
        input_tokens = tokenizer(x['prompt'], return_tensors="pt", padding=True).to("cuda")
        outputs = model.generate(**input_tokens, max_new_tokens=32, pad_token_id=tokenizer.eos_token_id)
        for i, output in enumerate(outputs):
            answer = tokenizer.decode(output)
            answer = answer.split(x['prompt'][i])[-1].strip().split('\n')[0].split('<')[0].lower()
            # answer = answer.split(x['prompt'][i])[-1].split('\n')[0].replace('*', '').replace('"', '').replace("'", '').replace('.', '').replace('<|end_of_text|>', '').strip().lower()
            # answer = answer.split(':')[-1].split('/')[-1].split('(')[-1].split(')')[0]
            answer = ''.join(x for x in answer if x.islower())
            if answer not in ['entailment', 'neutral', 'contradiction']:
                # if answer:
                # import pdb; pdb.set_trace()
                print(answer, x['label'][i])
            answers.append(answer)
            
            if answer == x['label'][i]:
                count += 1
        gts.extend(x['label'])
        # import pdb; pdb.set_trace()
        if idx % 100 == 0:
            toc = datetime.datetime.now().replace(microsecond=0)-tic
            print(f'Evaluated [{idx}/{len(dataloader)}]: Spent time: {toc}, Rest time: ', toc/(idx+1)*(len(dataloader)-idx))
    accuracy = accuracy_score(answers, gts)
    print('dataset len, correct number: ', len(dataset), count)
    print(f'{model_name} Accuracy on language {language}: ', accuracy, count/len(dataset))
    


def distributed_execution():
    # distributed_state = PartialState()
    accelerator = Accelerator()
    tic = datetime.datetime.now().replace(microsecond=0)
    model_name = 'google/gemma-2-2b-it'
    langs = ['ar','bg','de','el','es','fr','hi','ru','sw','th','tr','ur','vi','zh', 'en']
    language = 'ja'
    device = accelerator.device
    dataset = XnliDataset(language=language, data_dir='/app/evaluation/xnli/')
    dataloader = DataLoader(dataset, batch_size=4, shuffle=False)
    # ！ add here for accelerate
    dataloader = accelerator.prepare(dataloader)
    model, tokenizer = load_model(model_name, device)
    count = 0
    answers = []
    gts = []
    

    completions_per_process = []
    # with distributed_state.split_between_processes(tokenized_prompts, apply_padding=True) as batched_prompts:
    for idx, x in enumerate(dataloader):
        inputs, targets = x['prompt'], x['label']
        input_tokens = tokenizer(inputs, return_tensors="pt", padding=True).to(device)
        outputs = model.generate(**input_tokens, max_new_tokens=32, pad_token_id=tokenizer.eos_token_id)
        # answers = []
        # for i, output in enumerate(outputs):
        #     answer = tokenizer.decode(output)
        #     answer = answer.split(x['prompt'][i])[-1].strip().split('\n')[0].split('<')[0].lower()
        #     answer = ''.join(x for x in answer if x.islower())
        #     # if answer not in ['entailment', 'neutral', 'contradiction']:
        #     #     # import pdb; pdb.set_trace()
        #     #     print(answer, x['label'][i])
        #     answers.append(answer)
        # Gather all predictions and targets
        all_predictions, all_targets = accelerator.gather_for_metrics((outputs, targets))
        if idx % 100 == 0:
            toc = datetime.datetime.now().replace(microsecond=0)-tic
            print(f'Evaluated [{idx}/{len(dataloader)}]: Spent time: {toc}, Rest time: ', toc/(idx+1)*(len(dataloader)-idx))
        # # Example of use with a *Datasets.Metric*
        # metric.add_batch(all_predictions, all_targets)
    if distributed_state.is_main_process:
        answers = []
        for i, output in enumerate(all_predictions):
            answer = tokenizer.decode(output)
            answer = answer.split(x['prompt'][i])[-1].strip().split('\n')[0].split('<')[0].lower()
            answer = ''.join(x for x in answer if x.islower())
            if answer not in ['entailment', 'neutral', 'contradiction']:
                # import pdb; pdb.set_trace()
                print(answer, x['label'][i])
            answers.append(answer)
        accuracy = accuracy_score(answers, all_targets)
    # for idx, x in enumerate(dataloader):
    #     input_tokens = tokenizer(x['prompt'], return_tensors="pt", padding=True).to("cuda")
    #     outputs = model.generate(**input_tokens, max_new_tokens=32, pad_token_id=tokenizer.eos_token_id)
    #     for i, output in enumerate(outputs):
    #         answer = tokenizer.decode(output)
    #         answer = answer.split(x['prompt'][i])[-1].strip().split('\n')[0].split('<')[0]
    #         # answer = answer.split(x['prompt'][i])[-1].split('\n')[0].replace('*', '').replace('"', '').replace("'", '').replace('.', '').replace('<|end_of_text|>', '').strip().lower()
    #         # answer = answer.split(':')[-1].split('/')[-1].split('(')[-1].split(')')[0]
    #         answer = ''.join(x for x in answer if x.islower())
    #         if answer not in ['entailment', 'neutral', 'contradiction']:
    #             # import pdb; pdb.set_trace()
    #             print(answer, x['label'][i])
    #         answers.append(answer)
            
    #         if answer == x['label'][i]:
    #             count += 1
    #     gts.extend(x['label'])
    #     # import pdb; pdb.set_trace()
    #     if idx % 100 == 0:
    #         toc = datetime.datetime.now().replace(microsecond=0)-tic
    #         print(f'Evaluated [{idx}/{len(dataloader)}]: Spent time: {toc}, Rest time: ', toc/(idx+1)*(len(dataloader)-idx))
    # accuracy = accuracy_score(answers, gts)
    print('dataset len, correct number: ', len(dataset), count)
    print(f'{model_name} Accuracy on language {language}: ', accuracy, count/len(dataset))
    


if __name__ == '__main__':
    # execution()
    distributed_execution()
