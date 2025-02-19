"""
Downloads and evaluates HellaSwag in Python.
https://github.com/rowanz/hellaswag

Example HellaSwag json item:

{"ind": 24, "activity_label": "Roof shingle removal", "ctx_a": "A man is sitting on a roof.", "ctx_b": "he", "ctx": "A man is sitting on a roof. he", "split": "val", "split_type": "indomain", "label": 3, "endings": ["is using wrap to wrap a pair of skis.", "is ripping level tiles off.", "is holding a rubik's cube.", "starts pulling up roofing on a roof."], "source_id": "activitynet~v_-JhWjGDPHMY"}

ind: dataset ID
activity_label: The ActivityNet or WikiHow label for this example
context: There are two formats. The full context is in ctx. When the context ends in an (incomplete) noun phrase, like for ActivityNet, this incomplete noun phrase is in ctx_b, and the context up until then is in ctx_a. This can be useful for models such as BERT that need the last sentence to be complete. However, it's never required. If ctx_b is nonempty, then ctx is the same thing as ctx_a, followed by a space, then ctx_b.
endings: a list of 4 endings. The correct index is given by label (0,1,2, or 3)
split: train, val, or test.
split_type: indomain if the activity label is seen during training, else zeroshot
source_id: Which video or WikiHow article this example came from

gpt2 (124M)
- eleuther harness reports acc 28.92%, acc_norm 31.14% (multiple choice style)
- this script: 10042 acc: 0.2859 acc_norm: 0.2955 (completion style)

gpt2-xl (1558M)
- eleuther harness reports acc 40.04%, acc_norm 50.89% (multiple choice style)
- this script: 10042 acc: 0.3842 acc_norm: 0.4893 (completion style)

The validation set of HellaSwag has a total of 10,042 examples.
"""

import os
import json
import requests
import tiktoken
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers import GPT2LMHeadModel

# -----------------------------------------------------------------------------
class HellaSwagEvaluator:

    def render_example(self, example, tokenizer):
        """
        将数据处理为模型所需要的输入形式

        Given the example as a dictionary, render it as three torch tensors:
        - tokens (the tokens of context + completion, of size 4xN, as there are always 4 candidates)
        - mask (is 1 in the region of the candidate completion, where we evaluate likelihoods)
        - label (the index of the correct completion, which we hope has the highest likelihood)
        """
        ctx = example["ctx"]
        label = example["label"]
        endings = example["endings"]

        # data needed to reproduce this eval on the C size
        data = {
            "label": label,
            "ctx_tokens": None,
            "ending_tokens": [],
        }

        # gather up all the tokens
        ctx_tokens = tokenizer.encode(ctx)
        data["ctx_tokens"] = ctx_tokens
        tok_rows = []
        mask_rows = []
        for end in endings:
            end_tokens = tokenizer.encode(" " + end) # note: prepending " " because GPT-2 tokenizer
            tok_rows.append(ctx_tokens + end_tokens)
            mask_rows.append([0]*len(ctx_tokens) + [1]*len(end_tokens))
            data["ending_tokens"].append(end_tokens)

        # have to be careful during the collation because the number of tokens in each row can differ
        max_len = max(len(row) for row in tok_rows)
        tokens = torch.zeros((4, max_len), dtype=torch.long)
        mask = torch.zeros((4, max_len), dtype=torch.long)
        for i, (tok_row, mask_row) in enumerate(zip(tok_rows, mask_rows)):
            tokens[i, :len(tok_row)] = torch.tensor(tok_row)
            mask[i, :len(mask_row)] = torch.tensor(mask_row)

        return data, tokens, mask, label

    def iterate_examples(self, path):
        # there are 10,042 examples in total in val
        with open(path, "r") as f:
            for line in f:
                example = json.loads(line)
                yield example

    def evaluate_in_training(self, tokens, mask, logits):
        '''在训练的过程中评估'''
        # contiguous 方法在 PyTorch 中用于确保张量在内存中的存储是连续的
        shift_logits = (logits[..., :-1, :]).contiguous() # 去掉最后一个时间步的 logits
        shift_tokens = (tokens[..., 1:]).contiguous() # 去掉第一个时间步的 tokens
        # 将 shift_logits 和 shift_tokens 展平，使得每个位置的 logits 和 tokens 对应
        flat_shift_logits = shift_logits.view(-1, shift_logits.size(-1)) # shape=(4*19, 50257)
        flat_shift_tokens = shift_tokens.view(-1) # shape=(4*19)
        shift_losses = F.cross_entropy(flat_shift_logits, flat_shift_tokens, reduction='none') # 计算所有位置的交叉熵损失，不进行任何归约（reduction='none'），返回每个位置的损失值
        shift_losses = shift_losses.view(tokens.size(0), -1) # 损失值恢复为原来的形状
        # now get the average loss just for the completion region (where mask == 1), in each row
        shift_mask = (mask[..., 1:]).contiguous() # we must shift mask, so we start at the last prompt token
        masked_shift_losses = shift_losses * shift_mask # 将损失值与 mask 相乘，只保留掩码位置的损失
        # sum and divide by the number of 1s in the mask
        sum_loss = masked_shift_losses.sum(dim=1)
        avg_loss = sum_loss / shift_mask.sum(dim=1)
        # 得到了每个选项的loss，取得最小损失的选项就是最佳答案
        pred = sum_loss.argmin().item() # 找到损失最小的索引
        pred_norm = avg_loss.argmin().item() # 找到损失最小的索引
        return pred, pred_norm

    @torch.no_grad()
    def evaluate(self, model, device, tokenizer):
        '''训练结束得到权重文件后，对结果评估'''
        num_correct_norm = 0
        num_correct = 0
        num_total = 0
        for example in self.iterate_examples(HELLASWAG_PATH):
            data, tokens, mask, label = self.render_example(example, tokenizer)
            tokens = tokens.to(device)
            mask = mask.to(device)
            logits = model(tokens).logits
            pred, pred_norm = self.evaluate_in_training(tokens, mask, logits)
            # accumulate stats
            num_total += 1
            num_correct += int(pred == label)
            num_correct_norm += int(pred_norm == label)
            print(f"{num_total} acc_norm: {num_correct_norm}/{num_total}={num_correct_norm/num_total:.4f}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model_path", type=str, default="./model/gpt2", help="the model path to use")
    parser.add_argument("-d", "--device", type=str, default="cuda", help="the device to use")
    args = parser.parse_args()

    HELLASWAG_CACHE_DIR = os.path.join(os.path.dirname(__file__), "hellaswag")
    HELLASWAG_PATH = os.path.join(HELLASWAG_CACHE_DIR, f"hellaswag_val.jsonl")
    
    def download(split):
        """Downloads HellaSwag DATA_CACHE_DIR"""
        os.makedirs(HELLASWAG_CACHE_DIR, exist_ok=True)
        data_url = hellaswags[split]
        data_filename = os.path.join(HELLASWAG_CACHE_DIR, f"hellaswag_{split}.jsonl")
        if not os.path.exists(data_filename):
            print(f"Downloading {data_url} to {data_filename}...")
            download_file(data_url, data_filename)

    def download_file(url: str, fname: str, chunk_size=1024):
        """Helper function to download a file from a given url"""
        resp = requests.get(url, stream=True)
        total = int(resp.headers.get("content-length", 0))
        with open(fname, "wb") as file, tqdm(
            desc=fname,
            total=total,
            unit="iB",
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for data in resp.iter_content(chunk_size=chunk_size):
                size = file.write(data)
                bar.update(size)
    # download('val)
    hellaswags = {
        "train": "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_train.jsonl",
        "val": "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_val.jsonl",
        "test": "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_test.jsonl",
    }

    enc = tiktoken.get_encoding("gpt2")

    torch.set_float32_matmul_precision('high') # use tf32
    model = GPT2LMHeadModel.from_pretrained(args.model_path)
    model.to(args.device)
    model.eval()
    # model = torch.compile(model) # optionally torch compile the model

    hellaswag_evaluator = HellaSwagEvaluator()
    hellaswag_evaluator.evaluate(model, args.device, enc)
