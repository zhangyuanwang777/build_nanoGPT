'''
https://huggingface.co/datasets/LSDSem/story_cloze

This test requires a system to choose the correct ending to a four-sentence story.

Languages: English

需要手动下载数据集
'''

import os
import tiktoken
import torch
import pandas as pd
from tqdm import tqdm
from torch.nn import functional as F
from transformers import GPT2LMHeadModel

class StoryClozeEvaluator:

    def iterate_examples(self, path):
        '''数据据迭代器'''
        data = pd.read_csv(path)
        for i in range(len(data)):
            example = {}
            example["InputSentence1"] = data.iloc[i, :]["InputSentence1"]
            example["InputSentence2"] = data.iloc[i, :]["InputSentence2"]
            example["InputSentence3"] = data.iloc[i, :]["InputSentence3"]
            example["InputSentence4"] = data.iloc[i, :]["InputSentence4"]
            example["RandomFifthSentenceQuiz1"] = data.iloc[i, :]["RandomFifthSentenceQuiz1"]
            example["RandomFifthSentenceQuiz2"] = data.iloc[i, :]["RandomFifthSentenceQuiz2"]
            example["AnswerRightEnding"] = data.iloc[i, :]["AnswerRightEnding"]
            yield example

    def render_example(self, example, tokenizer):
        '''将数据处理为模型所需要的输入形式'''
        label = example["AnswerRightEnding"]
        input_sentence = [example["InputSentence1"], example["InputSentence2"], example["InputSentence3"], example["InputSentence4"]]
        input_sentence = " ".join(input_sentence) # 用空格将四个句子拼接起来
        input_tokens = tokenizer.encode(input_sentence)
        tok_rows = []
        mask_rows = []
        for end in [example["RandomFifthSentenceQuiz1"], example["RandomFifthSentenceQuiz2"]]:
            end_tokens = tokenizer.encode(" " + end)
            tok_rows.append(input_tokens + end_tokens)
            mask_rows.append([0] * len(input_tokens) + [1] * len(end_tokens))
        
        max_len = max([len(x) for x in tok_rows])
        tokens = torch.zeros((len(tok_rows), max_len), dtype=torch.long)
        mask = torch.zeros((len(tok_rows), max_len), dtype=torch.long)
        for i, (tok_row, mask_row) in enumerate(zip(tok_rows, mask_rows)):
            tokens[i, :len(tok_row)] = torch.tensor(tok_row)
            mask[i, :len(mask_row)] = torch.tensor(mask_row)

        return tokens, mask, label
    
    def evaluate_in_training(self, tokens, mask, logits):
        '''在训练的过程中评估'''
        # contiguous 方法在 PyTorch 中用于确保张量在内存中的存储是连续的
        shift_logits = (logits[..., :-1, :]).contiguous() # 去掉最后一个时间步的 logits
        shift_tokens = (tokens[..., 1:]).contiguous() # 去掉第一个时间步的 tokens
        shift_mask = (mask[..., 1:]).contiguous()
        # 将 shift_logits 和 shift_tokens 展平，使得每个位置的 logits 和 tokens 对应
        flat_shift_logits = shift_logits.view(-1, shift_logits.size(-1))
        flat_shift_tokens = shift_tokens.view(-1)
        shift_losses = F.cross_entropy(flat_shift_logits, flat_shift_tokens, reduction='none') # 计算所有位置的交叉熵损失，不进行任何归约（reduction='none'），返回每个位置的损失值
        shift_losses = shift_losses.view(tokens.size(0), -1) # 损失值恢复为原来的形状
        # now get the average loss just for the completion region (where mask == 1), in each row
        
        masked_shift_losses = shift_losses * shift_mask # 将损失值与 mask 相乘，只保留掩码位置的损失
        # sum and divide by the number of 1s in the mask
        sum_loss = masked_shift_losses.sum(dim=1)
        avg_loss = sum_loss / shift_mask.sum(dim=1)
        # 得到了每个选项的loss，取得最小损失的选项就是最佳答案
        pred = sum_loss.argmin().item() # 找到损失最小的索引
        pred_norm = avg_loss.argmin().item() # 找到损失最小的索引
        return pred, pred_norm
    
    @torch.no_grad()
    def evaluate(self, model, device, tokenizer, data_path):
        '''训练结束得到权重文件后，对结果评估'''
        num_correct_norm = 0
        num_correct = 0
        num_total = 0
        for example in tqdm(self.iterate_examples(data_path)):
            tokens, mask, label = self.render_example(example, tokenizer)
            tokens, mask = tokens.to(device), mask.to(device)
            label = label - 1 # label只有1和2，使其从0开始
            logits = model(tokens).logits
            pred, pred_norm = self.evaluate_in_training(tokens, mask, logits)
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

    DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__), "StoryCloze")
    PATH = os.path.join(DATA_CACHE_DIR, "cloze_test_val__winter2018-cloze_test_ALL_val - 1 - 1.csv")

    enc = tiktoken.get_encoding("gpt2")

    torch.set_float32_matmul_precision('high') # use tf32
    model = GPT2LMHeadModel.from_pretrained(args.model_path)
    model.to(args.device)
    model.eval()
    # model = torch.compile(model) # optionally torch compile the model

    storycloze_evaluator = StoryClozeEvaluator()
    storycloze_evaluator.evaluate(model, args.device, enc, PATH)
    

