'''
https://huggingface.co/datasets/EleutherAI/lambada_openai

This dataset is comprised of the LAMBADA test split as pre-processed by OpenAI (see relevant discussions here and here). It also contains machine translated versions of the split in German, Spanish, French, and Italian.
LAMBADA is used to evaluate the capabilities of computational models for text understanding by means of a word prediction task. LAMBADA is a collection of narrative texts sharing the characteristic that human subjects are able to guess their last word if they are exposed to the whole text, but not if they only see the last sentence preceding the target word. To succeed on LAMBADA, computational models cannot simply rely on local context, but must be able to keep track of information in the broader discourse.

Languages: English

需要手动下载数据集
'''

import os
import json
import tiktoken
from tqdm import tqdm
import torch
from torch.nn import functional as F
from transformers import GPT2LMHeadModel

class LambadaEvaluator:

    def iterate_examples(self, path):
        '''数据迭代器'''
        with open(path, "r") as f:
            for line in f:
                example = json.loads(line)
                yield example

    def render_example(self, example, tokenizer):
        '''将数据处理为模型所需要的输入形式'''
        text = example["text"]
        tokens = tokenizer.encode(text)
        tokens = torch.tensor(tokens, dtype=torch.long)
        tokens = tokens.unsqueeze(0) # 添加batch维度
        return tokens

    def evaluate_in_training(self, logits, strategy="Greedy"):
        '''在训练的过程中评估'''
        
        logits = logits[..., -2, :] # 预测最后一个 token 的 logit 
        flat_logits = logits.view(-1)
        last_token_probs = F.softmax(flat_logits, dim=-1)

        assert strategy in ["Sampling", "Greedy"]
        if strategy=="Greedy":
            # 贪心策略
            _, predicted_tokens = torch.max(last_token_probs, dim=-1)
        else:
            # 采样策略
            predicted_tokens = torch.multinomial(last_token_probs, num_samples=1) # 形状为 [batch_size, 1]
            predicted_tokens = predicted_tokens.squeeze(-1) # 形状为 [batch_size]
        return predicted_tokens

    @torch.no_grad()
    def evaluate(self, model, device, tokenizer, data_path):
        '''训练结束得到权重文件后，对结果评估'''
        num_total = 0
        num_correct = 0
        for example in tqdm(self.iterate_examples(data_path)):
            tokens = self.render_example(example, tokenizer)
            tokens = tokens.to(device)
            logits = model(tokens).logits
            predicted_tokens = self.evaluate_in_training(logits)
            num_total += 1
            num_correct += int(predicted_tokens == tokens.view(-1)[-1]) # tokens展平后取最后一个token
        print(f"{num_total} acc_norm: {num_correct}/{num_total}={num_correct/num_total:.4f}")
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model_path", type=str, default="./model/gpt2", help="the model path to use")
    parser.add_argument("-d", "--device", type=str, default="cuda", help="the device to use")
    args = parser.parse_args()

    DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__), "lambada_openai")
    PATH = os.path.join(DATA_CACHE_DIR, "lambada_test_en.jsonl")
    enc = tiktoken.get_encoding("gpt2")

    torch.set_float32_matmul_precision('high') # use tf32
    model = GPT2LMHeadModel.from_pretrained(args.model_path)
    model.to(args.device)
    model.eval()
    # model = torch.compile(model) # optionally torch compile the model

    lambada_evaluator = LambadaEvaluator()
    lambada_evaluator.evaluate(model, args.device, enc, PATH)