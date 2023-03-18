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

5. Added abs(diff(pool1, pool2)) along with the two original pooling outcomes to train similarity task.

Accuracy: 0.520, 0.795, 0.529, (0.615)

Update 03/17: tried concat -> dropout -> dnese to 64 -> dropout -> dense to 1 architecture, achieved 0.75
dev accuracy when training **only** the similarity task using dropout rate = 0.3 and epochs = 12. Can try 
15 epochs (Suspecting training for few more epochs may make it better?) 0.5 dropout rate has only 0.71 acc.
0.4 dropout = 0.74 acc. We'll keep dropout = 0.3 for later. Update 03/18: this method fails when training 
three tasks simutaneously. Accuracy are 0.497, 0.787, 0.531. No improvement on similarity, but worse on
sentiment classification. Also, the additional dataset approach also failed. Decreased performance in both
sentiment classification and similarity tasks compared to the best one. 
