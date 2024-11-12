# import os
# import csv
# import json
# import datetime
# import torch
# import transformers
# # import tensorflow as tf
# from transformers.data.processors.utils import DataProcessor, InputExample
# from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM
# from torch.utils.data import DataLoader, Dataset

# class XnliDataset(Dataset):
#     def __init__(self, data_dir, language):
#         self.language = language

#         if self.language == 'ja':
#             self.examples = self.load_jnli(data_dir)
#         else:
#             self.examples = self.load_xnli(data_dir)

#     def load_jnli(self, data_dir):
#         with open(os.path.join(data_dir, "XNLI-1.0/jnli_valid-v1.1.json")) as f:
#             # lines = json.load(f)
#             lines = [json.loads(l) for l in f.readlines()]
#         examples = []
#         label_trans = { "entailment": "含意", "neutral": "中立", "contradiction": "矛盾"}
#         for i, line in enumerate(lines):
#             # prompt = f'Sentence 1: {line["sentence1"]} \nSentence 2: {line["sentence2"]} \nIs the two sentence entailment, neutral, or contradiction?\nAnswer from "entailment", "neutral", and "contradiction".\nAnswer:'
#             prompt = f'文 1: {line["sentence1"]} \n文 2: {line["sentence2"]} \n2 つの文は含意、中立、または矛盾ですか?\n答えは「含意」、「中立」、「矛盾」から選べます。\n答え:'
#             examples.append({'guid': line['sentence_pair_id'], 'prompt': prompt, 'label': label_trans[line['label']]})
#         return examples


#     def load_xnli(self, data_dir):
#         lines = self._read_tsv(os.path.join(data_dir, "XNLI-1.0/xnli.test.tsv"))
#         examples = []
#         for i, line in enumerate(lines):
#             if i == 0:
#                 continue
#             language = line[0]
#             if language != self.language:
#                 continue
#             guid = f"test-{i}"
#             text_a = line[6]
#             text_b = line[7]
#             label = line[1]
#             if not isinstance(text_a, str):
#                 raise TypeError(f"Training input {text_a} is not a string")
#             if not isinstance(text_b, str):
#                 raise TypeError(f"Training input {text_b} is not a string")
#             if not isinstance(label, str):
#                 raise TypeError(f"Training label {label} is not a string")
#             prompt = f'Sentence 1: {text_a} \nSentence 2: {text_b} \nIs the two sentence entailment, neutral, or contradiction?\nAnswer from "entailment", "neutral", and "contradiction".\nAnswer:'
#             examples.append({'guid': guid, 'prompt': prompt, 'label': label})
#             # self.examples.append({'guid': guid, 'text_a': text_a, 'text_b': text_b, 'label': label})
#                 # InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
#         return examples
        
#     def _read_tsv(cls, input_file, quotechar=None):
#         """Reads a tab separated value file."""
#         with open(input_file, "r") as f:
#             reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
#             lines = []
#             for line in reader:
#                 lines.append(line)
#             return lines

#     def __len__(self):
#         return len(self.examples)

#     def __getitem__(self, idx):
#         return self.examples[idx]

# class XnliProcessor(DataProcessor):
#     """
#     Processor for the XNLI dataset. Adapted from
#     https://github.com/google-research/bert/blob/f39e881b169b9d53bea03d2d341b31707a6c052b/run_classifier.py#L207
#     """

#     def __init__(self, language, train_language=None):
#         self.language = language
#         self.train_language = train_language


#     def get_train_examples(self, data_dir):
#         """See base class."""
#         lg = self.language if self.train_language is None else self.train_language
#         lines = self._read_tsv(os.path.join(data_dir, f"XNLI-MT-1.0/multinli/multinli.train.{lg}.tsv"))
#         examples = []
#         for i, line in enumerate(lines):
#             if i == 0:
#                 continue
#             guid = f"train-{i}"
#             text_a = line[0]
#             text_b = line[1]
#             label = "contradiction" if line[2] == "contradictory" else line[2]
#             if not isinstance(text_a, str):
#                 raise TypeError(f"Training input {text_a} is not a string")
#             if not isinstance(text_b, str):
#                 raise TypeError(f"Training input {text_b} is not a string")
#             if not isinstance(label, str):
#                 raise TypeError(f"Training label {label} is not a string")
#             examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
#         return examples

#     def get_test_examples(self, data_dir):
#         """See base class."""
#         lines = self._read_tsv(os.path.join(data_dir, "XNLI-1.0/xnli.test.tsv"))
#         examples = []
#         for i, line in enumerate(lines):
#             if i == 0:
#                 continue
#             language = line[0]
#             if language != self.language:
#                 continue
#             guid = f"test-{i}"
#             text_a = line[6]
#             text_b = line[7]
#             label = line[1]
#             if not isinstance(text_a, str):
#                 raise TypeError(f"Training input {text_a} is not a string")
#             if not isinstance(text_b, str):
#                 raise TypeError(f"Training input {text_b} is not a string")
#             if not isinstance(label, str):
#                 raise TypeError(f"Training label {label} is not a string")
#             examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
#         return examples
    
#     def get_dev_examples(self, data_dir):
#         """See base class."""
#         lines = self._read_tsv(os.path.join(data_dir, "XNLI-1.0/xnli.test.tsv"))
#         examples = []
#         for i, line in enumerate(lines):
#             if i == 0:
#                 continue
#             language = line[0]
#             if language != self.language:
#                 continue
#             guid = f"test-{i}"
#             text_a = line[6]
#             text_b = line[7]
#             label = line[1]
#             if not isinstance(text_a, str):
#                 raise TypeError(f"Training input {text_a} is not a string")
#             if not isinstance(text_b, str):
#                 raise TypeError(f"Training input {text_b} is not a string")
#             if not isinstance(label, str):
#                 raise TypeError(f"Training label {label} is not a string")
#             examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
#         return examples

#     def get_labels(self):
#         """See base class."""
#         return ["contradiction", "entailment", "neutral"]

# def load_model(model_name, device='cuda'):
#     tokenizer = AutoTokenizer.from_pretrained(model_name)
#     model = AutoModelForCausalLM.from_pretrained(
#         model_name,
#         device_map="auto",
#         torch_dtype=
#         torch.bfloat16,
#     )
#     model.generation_config.pad_token_id = tokenizer.pad_token_id
#     return model, tokenizer


# def execution():
#     tic = datetime.datetime.now().replace(microsecond=0)
#     model_name = 'google/gemma-2-2b-it'
#     # model_name = 'google/gemma-2-2b-jpn-it'
#     # model_name = '/app/mergekit/output_models_mergekit/linear_gemma'
#     device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
#     # Spanish(es), German(de), Greek(el), Bulgarian(bg), Russian(ru), Turkish(tr), Arabic(ar), Vietnamese(vi), Thai(th), Chinese(zh), Hindi(hi), Swahili(sw) and Urdu(ur)
#     langs = ['ar','bg','de','el','es','fr','hi','ru','sw','th','tr','ur','vi','zh', 'en']
#     language = 'ja'
#     dataset = XnliDataset(language=language, data_dir='/app/evaluation/xnli/')
#     dataloader = DataLoader(dataset, batch_size=4, shuffle=False)
#     # dataloader = accelerator.prepare(dataloader)
#     # processor = XnliProcessor(language='en')
#     # dev_set = processor.get_dev_examples(data_dir='/app/evaluation/xnli/')
#     model, tokenizer = load_model(model_name, device)
#     count = 0
#     # total = 0

#     # for idx, x in enumerate(dev_set):
#     for idx, x in enumerate(dataloader):
#         # import pdb; pdb.set_trace()
#         # input_text = prompt.replace('{premise}', x.text_a).replace('{hypothesis}', x.text_b)
#         input_tokens = tokenizer(x['prompt'], return_tensors="pt", padding=True).to("cuda")
#         outputs = model.generate(**input_tokens, max_new_tokens=32, pad_token_id=tokenizer.eos_token_id)
#         for i, output in enumerate(outputs):
#             answer = tokenizer.decode(output)
#             answer = answer.split(x['prompt'][i])[-1].split('\n')[0].replace('*', '').replace('"', '').replace("'", '').replace('.', '').replace('<|end_of_text|>', '').strip().lower()
#             # if answer not in ['entailment', 'neutral', 'contradiction']:
#             #     print(answer, x['prompt'][i])
#             print(answer, x['label'][i])
#             import pdb; pdb.set_trace()
#             if answer == x['label'][i]:
#                 count += 1
#             # total += 1
#         # import pdb; pdb.set_trace()
#         if idx % 100 == 0:
#             toc = datetime.datetime.now().replace(microsecond=0)-tic
#             print(f'Evaluated [{idx}/{len(dataloader)}]: Spent time: {toc}, Rest time: ', toc/(idx+1)*(len(dataloader)-idx))
#     print(f'{model_name} Accuracy on language {language}: ', count/len(dataset))
#     print('dataset len, correct number: ', len(dataset), count)

# execution()

# # print(device)
# # import pdb; pdb.set_trace()

# # import pdb; pdb.set_trace()
# # prompt = f'Is the two sentence entailment, neutral, or contradiction? "{premise}," and "{hypothesis}"'
# # # sentences = [premise, hypothesis]
# # # sentence_embeddings = model.encode(sentences)
# # # from sklearn.metrics.pairwise import paired_cosine_distances
# # # cosine_score = 1 - paired_cosine_distances([sentence_embeddings[0]],[sentence_embeddings[1]])
# # input = tokenizer(premise, hypothesis, truncation=True, return_tensors="pt")
# # output = model(input["input_ids"].to(device))

# # input_ids = tokenizer(input_text, return_tensors="pt").to("cuda")

# # outputs = model(input_ids["input_ids"].to(device))
# # print(tokenizer.decode(outputs[0]))
# # pipeline = transformers.pipeline(
# #     "text-generation", model=model_name, model_kwargs={"torch_dtype": torch.bfloat16}, device_map="auto"
# # )
# # pipeline("Hey how are you doing today?", max_new_tokens=20)

# # input = tokenizer(premise, hypothesis, truncation=True, return_tensors="pt")
# # output = model(input["input_ids"].to(device))  # device = "cuda:0" or "cpu"
# # prediction = torch.softmax(output["logits"][0], -1).tolist()
# # label_names = ["entailment", "neutral", "contradiction"]
# # prediction = {name: round(float(pred) * 100, 1) for pred, name in zip(prediction, label_names)}
# # print(prediction)


import os
import csv
import json
from tqdm import tqdm
import datetime
import torch
import transformers
import multiprocessing
from functools import partial
from accelerate import PartialState
from accelerate.utils import gather_object
from accelerate import Accelerator
# import tensorflow as tf
from sklearn.metrics import accuracy_score
from transformers.data.processors.utils import DataProcessor, InputExample
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM
from torch.utils.data import DataLoader, Dataset

class XnliDataset(Dataset):
    def __init__(self, data_dir, language):
        self.language = language
        self.data_dir = data_dir
        # if get_plain_text:
        #     sentences, targets = self.load_plain_text(data_dir)
        #     return sentences, targets
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
        torch_dtype=torch.bfloat16,
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
    model_name = 'google/gemma-2-2b-it'
    # model_name = 'google/gemma-2-2b-jpn-it'
    # model_name = 'meta-llama/Meta-Llama-3-8B'
    # model_name = 'tokyotech-llm/Llama-3-Swallow-8B-v0.1'
    # model_name = '/app/mergekit/output_models_mergekit/linear_gemma'
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    # Spanish(es), German(de), Greek(el), Bulgarian(bg), Russian(ru), Turkish(tr), Arabic(ar), Vietnamese(vi), Thai(th), Chinese(zh), Hindi(hi), Swahili(sw) and Urdu(ur)
    langs = ['ar','bg','de','el','es','fr','hi','ru','sw','th','tr','ur','vi','zh', 'en']
    language = 'ja'
    dataset = XnliDataset(language=language, data_dir='/app/xnli_evaluation/')
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
    dataset = XnliDataset(language=language, data_dir='/app/xnli_evaluation/')
    dataloader = DataLoader(dataset, batch_size=8, shuffle=False)
    # ！ add here for accelerate
    dataloader = accelerator.prepare(dataloader)
    model, tokenizer = load_model(model_name, device)
    count = 0
    answers = []
    gts = []
    
    # with distributed_state.split_between_processes(tokenized_prompts, apply_padding=True) as batched_prompts:
    all_predictions, all_targets
    for idx, x in enumerate(dataloader):
        inputs, targets = x['prompt'], x['label']
        input_tokens = tokenizer(inputs, padding=True, pad_to_multiple_of=pad_to_multiple_of, return_tensors="pt").to(device)
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
        predictions = accelerator.gather_for_metrics(outputs)
        # print(x)
        # import pdb; pdb.set_trace()
        # all_predictions, all_targets = accelerator.gather_for_metrics((outputs, targets))
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
    

def complete_prompt(sentences, lang='en'):
    # prompts = []
    # for sentence in sentences:
    text_a, text_b = sentences
    if lang !='ja':
        prompt = f'Sentence 1: {text_a} \nSentence 2: {text_b} \nIs the two sentence entailment, neutral, or contradiction?\nAnswer from "entailment", "neutral", and "contradiction".\nAnswer: '
    else:
        prompt = f'### 指示:\n前提と仮説の関係をentailment、contradiction、neutralの中から回答してください。それ以外には何も含めないことを厳守してください。\n\n制約：\n- 前提から仮説が、論理的知識や常識的知識を用いて導出可能である場合はentailmentと出力\n- 前提と仮説が両立しえない場合はcontradictionと出力\n- そのいずれでもない場合はneutralと出力\n\n\n### 入力:\n前提：{line["sentence1"]}\n仮説：{line["sentence2"]}\n\n### 応答:'
    return prompt

def post_process(outputs, prompts):
    answers = []
    for output, prompt in zip(outputs, prompts):
        answer = output.split(prompt)[-1].strip().split('\n')[0].split('<')[0].lower().replace('answer', '')
        answer = ''.join(x for x in answer if x.islower())
        if answer not in ['entailment', 'neutral', 'contradiction']:
            print(answer)
        answers.append(answer)
    return answers

def distributed_eval():
    distributed_state = PartialState()

    # Tokenizer and model load
    # model_name = 'google/gemma-2-2b-it'
    # model_name = 'google/gemma-2-2b-jpn-it'
    model_name = 'meta-llama/Meta-Llama-3-8B'
    # model_name = 'tokyotech-llm/Llama-3-Swallow-8B-v0.1'
    model, tokenizer = load_model(model_name, device=distributed_state.device)


    # Data load
    language = 'ja'
    sentences, targets = XnliDataset(language=language, data_dir='/app/xnli_evaluation/').load_plain_text()
    # dataloader = DataLoader(dataset, batch_size=8, shuffle=False)

    # Preprocess
    # You can change the batch size depending on your GPU RAM
    batch_size = 8
    # We set it to 8 since it is better for some hardware. More information here https://github.com/huggingface/tokenizers/issues/991
    pad_to_multiple_of = 8

    with multiprocessing.Pool(processes=4) as p:
        prompts = p.map(partial(complete_prompt, language), sentences)

    # Split into batches
    # We will get the following results:
    # [ ["I would like to", "hello how are you"], [ "what is going on", "roses are red and"], [ "welcome to the hotel"] ]
    formatted_prompts = [prompts[i: i + batch_size] for i in range(0, len(prompts), batch_size)]

    # Tokenize each batch
    # tokenized_prompts = [
    #     tokenizer(formatted_prompt, padding=True, pad_to_multiple_of=pad_to_multiple_of, return_tensors="pt")
    #     for formatted_prompt in formatted_prompts
    # ]

    # Distributed inference
    completions_per_process = []
    # for idx, x in enumerate(dataloader):
    #     inputs, targets = x['prompt'], x['label']
    #     input_tokens = tokenizer(inputs, padding=True, pad_to_multiple_of=pad_to_multiple_of, return_tensors="pt").to(device)
        # outputs = model.generate(**input_tokens, max_new_tokens=32, pad_token_id=tokenizer.eos_token_id)
    with distributed_state.split_between_processes(formatted_prompts, apply_padding=True) as batched_prompts:
        with tqdm(total=len(batched_prompts), desc="inference progress") as pbar:
            for batch in batched_prompts:
                # Move the batch to the device
                tokenized_batch = tokenizer(batch, padding=True, pad_to_multiple_of=pad_to_multiple_of, return_tensors="pt")
                tokenized_batch = tokenized_batch.to(distributed_state.device)
                # We generate the text, decode it and add it to the list completions_per_process
                outputs = model.generate(**tokenized_batch,
                    max_new_tokens=32,
                    pad_token_id=tokenizer.eos_token_id,
                )
                generated_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)
                answers = post_process(generated_text, batch)
                completions_per_process.extend(answers)
                # targets_per_process.extend(targets)
                pbar.update(1)

    # We are gathering string, so we need to use gather_object.
    # If you need to gather tensors, you can use gather from accelerate.utils
    completions_gather = gather_object(completions_per_process)
    results = completions_gather[: len(prompts)] 

    if distributed_state.is_main_process:
        print(len(results))
        # import pdb; pdb.set_trace()
        accuracy = accuracy_score(results, targets)
        print(f'{model_name} Accuracy on language {language}: ', accuracy)


if __name__ == '__main__':
    # execution()
    # distributed_execution()
    distributed_eval()
