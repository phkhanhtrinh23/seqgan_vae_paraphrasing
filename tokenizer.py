from nltk.tokenize import TweetTokenizer
from transformers import BertTokenizer, BertConfig
import torch
from torch import nn
import numpy as np

# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

tokenizer = TweetTokenizer(preserve_case=False)

sentence = "the report said that israeli warplanes <mask> an air raid on wednesday night at guerrilla targets in the eastern sector of south lebanon"

tokens = tokenizer.tokenize(sentence)

print(tokenizer.tokenize(sentence))
# decode = tokenizer.encode(sentence, max_length=100, padding="max_length")
# decode[len(tokenizer.tokenize(sentence))+2-1] = 0
# print(decode)
# tokens = tokenizer.encode(sentence)
# print(tokenizer.decode(tokens))
# print(tokenizer.convert_tokens_to_ids("[PAD]"))
# print("[MASK] ", tokenizer.convert_tokens_to_ids("[MASK]"))
# print(BertConfig().vocab_size)

# m = nn.LogSoftmax(dim=1)
# loss = nn.CrossEntropyLoss()
# # input is of size N x C = 3 x 5
# input = torch.randn(3, 5, requires_grad=True)
# # each element in target has to have 0 <= value < C
# target = torch.tensor([1, 0, 4])
# output = loss(m(input), target)
# print(input)
# print()
# print(m(input), target, output)

# a= [1,2,3,4,5,6,7,8,9,10]
# print(np.mean(a))