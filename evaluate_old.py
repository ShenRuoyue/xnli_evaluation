import os
import csv
import json
import datetime
import torch
import transformers
# import tensorflow as tf
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
        label_trans = { "entailment": "含意", "neutral": "中立", "contradiction": "矛盾"}
        for i, line in enumerate(lines):
            # prompt = f'Sentence 1: {line["sentence1"]} \nSentence 2: {line["sentence2"]} \nIs the two sentence entailment, neutral, or contradiction?\nAnswer from "entailment", "neutral", and "contradiction".\nAnswer:'
            prompt = f'文 1: {line["sentence1"]} \n文 2: {line["sentence2"]} \n2 つの文は含意、中立、または矛盾ですか?\n答えは「含意」、「中立」、「矛盾」から選べます。\n答え:'
            examples.append({'guid': line['sentence_pair_id'], 'prompt': prompt, 'label': label_trans[line['label']]})
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
            prompt = f'Sentence 1: {text_a} \nSentence 2: {text_b} \nIs the two sentence entailment, neutral, or contradiction?\nAnswer from "entailment", "neutral", and "contradiction".\nAnswer:'
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

class XnliProcessor(DataProcessor):
    """
    Processor for the XNLI dataset. Adapted from
    https://github.com/google-research/bert/blob/f39e881b169b9d53bea03d2d341b31707a6c052b/run_classifier.py#L207
    """

    def __init__(self, language, train_language=None):
        self.language = language
        self.train_language = train_language


    def get_train_examples(self, data_dir):
        """See base class."""
        lg = self.language if self.train_language is None else self.train_language
        lines = self._read_tsv(os.path.join(data_dir, f"XNLI-MT-1.0/multinli/multinli.train.{lg}.tsv"))
        examples = []
        for i, line in enumerate(lines):
            if i == 0:
                continue
            guid = f"train-{i}"
            text_a = line[0]
            text_b = line[1]
            label = "contradiction" if line[2] == "contradictory" else line[2]
            if not isinstance(text_a, str):
                raise TypeError(f"Training input {text_a} is not a string")
            if not isinstance(text_b, str):
                raise TypeError(f"Training input {text_b} is not a string")
            if not isinstance(label, str):
                raise TypeError(f"Training label {label} is not a string")
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

    def get_test_examples(self, data_dir):
        """See base class."""
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
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples
    
    def get_dev_examples(self, data_dir):
        """See base class."""
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
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

    def get_labels(self):
        """See base class."""
        return ["contradiction", "entailment", "neutral"]

def load_model(model_name, device='cuda'):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=
        torch.bfloat16,
    )
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    return model, tokenizer


def execution():
    tic = datetime.datetime.now().replace(microsecond=0)
    model_name = 'google/gemma-2-2b-it'
    # model_name = 'google/gemma-2-2b-jpn-it'
    # model_name = '/app/mergekit/output_models_mergekit/linear_gemma'
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    # Spanish(es), German(de), Greek(el), Bulgarian(bg), Russian(ru), Turkish(tr), Arabic(ar), Vietnamese(vi), Thai(th), Chinese(zh), Hindi(hi), Swahili(sw) and Urdu(ur)
    langs = ['ar','bg','de','el','es','fr','hi','ru','sw','th','tr','ur','vi','zh', 'en']
    language = 'ja'
    dataset = XnliDataset(language=language, data_dir='/app/evaluation/xnli/')
    dataloader = DataLoader(dataset, batch_size=4, shuffle=False)
    # dataloader = accelerator.prepare(dataloader)
    # processor = XnliProcessor(language='en')
    # dev_set = processor.get_dev_examples(data_dir='/app/evaluation/xnli/')
    model, tokenizer = load_model(model_name, device)
    count = 0
    # total = 0

    # for idx, x in enumerate(dev_set):
    for idx, x in enumerate(dataloader):
        # import pdb; pdb.set_trace()
        # input_text = prompt.replace('{premise}', x.text_a).replace('{hypothesis}', x.text_b)
        input_tokens = tokenizer(x['prompt'], return_tensors="pt", padding=True).to("cuda")
        outputs = model.generate(**input_tokens, max_new_tokens=32, pad_token_id=tokenizer.eos_token_id)
        for i, output in enumerate(outputs):
            answer = tokenizer.decode(output)
            answer = answer.split(x['prompt'][i])[-1].split('\n')[0].replace('*', '').replace('"', '').replace("'", '').replace('.', '').replace('<|end_of_text|>', '').strip().lower()
            # if answer not in ['entailment', 'neutral', 'contradiction']:
            #     print(answer, x['prompt'][i])
            print(answer, x['label'][i])
            import pdb; pdb.set_trace()
            if answer == x['label'][i]:
                count += 1
            # total += 1
        # import pdb; pdb.set_trace()
        if idx % 100 == 0:
            toc = datetime.datetime.now().replace(microsecond=0)-tic
            print(f'Evaluated [{idx}/{len(dataloader)}]: Spent time: {toc}, Rest time: ', toc/(idx+1)*(len(dataloader)-idx))
    print(f'{model_name} Accuracy on language {language}: ', count/len(dataset))
    print('dataset len, correct number: ', len(dataset), count)

execution()

# print(device)
# import pdb; pdb.set_trace()

# import pdb; pdb.set_trace()
# prompt = f'Is the two sentence entailment, neutral, or contradiction? "{premise}," and "{hypothesis}"'
# # sentences = [premise, hypothesis]
# # sentence_embeddings = model.encode(sentences)
# # from sklearn.metrics.pairwise import paired_cosine_distances
# # cosine_score = 1 - paired_cosine_distances([sentence_embeddings[0]],[sentence_embeddings[1]])
# input = tokenizer(premise, hypothesis, truncation=True, return_tensors="pt")
# output = model(input["input_ids"].to(device))

# input_ids = tokenizer(input_text, return_tensors="pt").to("cuda")

# outputs = model(input_ids["input_ids"].to(device))
# print(tokenizer.decode(outputs[0]))
# pipeline = transformers.pipeline(
#     "text-generation", model=model_name, model_kwargs={"torch_dtype": torch.bfloat16}, device_map="auto"
# )
# pipeline("Hey how are you doing today?", max_new_tokens=20)

# input = tokenizer(premise, hypothesis, truncation=True, return_tensors="pt")
# output = model(input["input_ids"].to(device))  # device = "cuda:0" or "cpu"
# prediction = torch.softmax(output["logits"][0], -1).tolist()
# label_names = ["entailment", "neutral", "contradiction"]
# prediction = {name: round(float(pred) * 100, 1) for pred, name in zip(prediction, label_names)}
# print(prediction)


