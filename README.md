# CS224N Final Project

Without multitask implemented (total random guess for paraphrase and similaty tasks):

Accuracy: 0.518, 0.429, -0.016, (0.310)

With multitask implemeted version 1: same linear layers were used for both paraphrase and similarity task. 
Specifically, concat pooler output 1 and 2 together and feed into one dense layer.

Accuracy for option = pretrain: 0.372, 0.652, 0.288, (0.437)

Accuracy for option = finetune: 0.479, 0.781, 0.274, (0.511)
