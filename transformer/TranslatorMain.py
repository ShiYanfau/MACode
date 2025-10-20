

from TransformerTranslator import Transformer
import torch
import torch.nn as nn



d_model = 512
heads = 8
N = 6
src_vocab = len(EN_TEXT.vocab)
trg_vocab = len(FR_TEXT.vocab)

model = Transformer(src_vocab, trg_vocab, d_model, N, heads)

# 参数初始化
for p in model.parameters():
    if p.dim() > 1:
        nn.init.xavier_uniform_(p)

optim = torch.optim.Adam(
    model.parameters(),
    lr=0.0001,
    betas=(0.9, 0.98),
    eps=1e-9
)



import time
import torch.nn.functional as F

def train_model(epochs, print_every=100):
    model.train()
    start = time.time()
    temp = start
    total_loss = 0

    for epoch in range(epochs):
        for i, batch in enumerate(train_iter):
            src = batch.English.transpose(0, 1)  # [batch, seq]
            trg = batch.French.transpose(0, 1)   # [batch, seq]

            # decoder 输入（去掉最后一个 token）
            trg_input = trg[:, :-1]

            # 预测目标（去掉第一个 token <sos>）
            targets = trg[:, 1:].contiguous().view(-1)

            # 构建 mask
            src_mask, trg_mask = create_masks(src, trg_input)

            # 前向传播
            preds = model(src, trg_input, src_mask, trg_mask)

            optim.zero_grad()
            loss = F.cross_entropy(
                preds.view(-1, preds.size(-1)),
                targets,
                ignore_index=target_pad
            )
            loss.backward()
            optim.step()

            total_loss += loss.item()

            if (i + 1) % print_every == 0:
                loss_avg = total_loss / print_every
                print(
                    "time %dm, epoch %d, iter %d, loss = %.3f"
                    % ((time.time() - start) // 60, epoch + 1, i + 1, loss_avg)
                )
                total_loss = 0
                temp = time.time()


import numpy as np

def translate(model, src, max_len=80, custom_string=False):
    model.eval()
    if custom_string:
        src = tokenize_en(src)
        sentence = torch.LongTensor(
            [[EN_TEXT.vocab.stoi[tok] for tok in src]]
        ).to(next(model.parameters()).device)

    src_mask = (src != input_pad).unsqueeze(-2)
    e_outputs = model.encoder(src, src_mask)

    outputs = torch.zeros(max_len).long().to(src.device)
    outputs[0] = FR_TEXT.vocab.stoi["<sos>"]

    for i in range(1, max_len):
        trg_mask = np.triu(np.ones((1, i, i)), k=1).astype("uint8")
        trg_mask = torch.from_numpy(trg_mask) == 0
        trg_mask = trg_mask.to(src.device)

        out = model.decoder(outputs[:i].unsqueeze(0), e_outputs, src_mask, trg_mask)
        out = model.out(out)

        prob = F.softmax(out, dim=-1)
        _, ix = prob[:, -1].data.topk(1)

        outputs[i] = ix[0][0]

        if ix[0][0].item() == FR_TEXT.vocab.stoi["<eos>"]:
            break

    return " ".join([FR_TEXT.vocab.itos[ix] for ix in outputs[1:i]])







