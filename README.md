# CS224N Final Project

1. Without multitask implemented (total random guess for paraphrase and similaty tasks):

Accuracy: 0.518, 0.429, -0.016, (0.310)

2. With multitask implemeted version 1: same linear layers were used for both paraphrase and similarity task. 
Specifically, concat pooler output 1 and 2 together and feed into one dense layer.

Accuracy for option = pretrain: 0.372, 0.652, 0.288, (0.437)

Accuracy for option = finetune: 0.479, 0.781, 0.274, (0.511)

3. Using cosine similarity instead of single dense layer for training similarity increases similarity accuracy by a little.
However, this induces overfitting during training. Prior, sentence similiarity accuracy was low (~30%) on both training
and dev set during training epochs. After changing layer, it's good on training set (~80%) but still around 40% on dev set.

Accuracy for using cosine simiarity during training: 0.503, 0.722, 0.412, (0.546)

4. Tried learning rate decay and warmup steps. This time learning rate linearly rises up to 1e-5 on the first 20% steps, and linearly
drop to zero during the later 80% steps.

Accuracy: 0.520, 0.767, 0.419, (0.569)
