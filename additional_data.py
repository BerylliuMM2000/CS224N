import time, random, numpy as np, argparse, sys, re, os
from types import SimpleNamespace

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from bert import BertModel
from optimizer import AdamW
from tqdm import tqdm

from datasets import SentenceClassificationDataset, SentencePairDataset, \
    load_multitask_data, load_multitask_test_data, load_multitask_data_merged

from evaluation import model_eval_sst, test_model_multitask, model_eval_multitask


TQDM_DISABLE=True

# fix the random seed
def seed_everything(seed=11711):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


BERT_HIDDEN_SIZE = 768
N_SENTIMENT_CLASSES = 5
##### Added Warmup period #####
WARMUP_PERCENTAGE = 0.2

class MultitaskBERT(nn.Module):
    '''
    This module should use BERT for 3 tasks:
    - Sentiment classification (predict_sentiment)
    - Paraphrase detection (predict_paraphrase)
    - Semantic Textual Similarity (predict_similarity)
    '''
    def __init__(self, config):
        super(MultitaskBERT, self).__init__()
        # You will want to add layers here to perform the downstream tasks.
        # Pretrain mode does not require updating bert paramters.
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        for param in self.bert.parameters():
            if config.option == 'pretrain':
                param.requires_grad = False
            elif config.option == 'finetune':
                param.requires_grad = True
        ### TODO
        # sentiment classification
        self.sentiment_layer = nn.Linear(config.hidden_size, 5)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # paraphrase detection
        self.paraphrase_layer = nn.Linear(config.hidden_size * 2, 1)

        # semantic similarity
        self.similarity_layer = nn.Linear(config.hidden_size * 3, 64)
        self.similarity_layer2 = nn.Linear(64, 1)
        # self.similarity_layer2 = nn.Linear(2, 1)
        # self.similarity_layer = nn.Linear(config.hidden_size * 2, 1)

    def forward(self, input_ids, attention_mask):
        'Takes a batch of sentences and produces embeddings for them.'
        # The final BERT embedding is the hidden state of [CLS] token (the first token)
        # Here, you can start by just returning the embeddings straight from BERT.
        # When thinking of improvements, you can later try modifying this
        # (e.g., by adding other layers).
        ### TODO
        # Get BERT hidden states
        output = self.bert(input_ids, attention_mask)
        return output['pooler_output']

    def predict_sentiment(self, input_ids, attention_mask):
        '''Given a batch of sentences, outputs logits for classifying sentiment.
        There are 5 sentiment classes:
        (0 - negative, 1- somewhat negative, 2- neutral, 3- somewhat positive, 4- positive)
        Thus, your output should contain 5 logits for each sentence.
        '''
        ### TODO
        pooler_output = self.forward(input_ids, attention_mask)
        cls = self.sentiment_layer(self.dropout(pooler_output))
        logits = F.log_softmax(cls, dim=-1)
        return logits

    def predict_paraphrase(self,
                           input_ids_1, attention_mask_1,
                           input_ids_2, attention_mask_2):
        '''Given a batch of pairs of sentences, outputs a single logit for predicting whether they are paraphrases.
        Note that your output should be unnormalized (a logit); it will be passed to the sigmoid function
        during evaluation, and handled as a logit by the appropriate loss function.
        '''
        ### TODO
        # Get BERT hidden states for both sentences
        pooler_output_1 = self.forward(input_ids_1, attention_mask_1)
        pooler_output_2 = self.forward(input_ids_2, attention_mask_2)

        # Concatenate the hidden states of the two sentences
        concat_output = torch.cat((pooler_output_1, pooler_output_2), dim=1)

        # Get paraphrase logits
        paraphrase_logits = self.paraphrase_layer(self.dropout(concat_output))

        # Return paraphrase logits
        return paraphrase_logits


    def predict_similarity(self,
                           input_ids_1, attention_mask_1,
                           input_ids_2, attention_mask_2):
        '''Given a batch of pairs of sentences, outputs a single logit corresponding to how similar they are.
        Note that your output should be unnormalized (a logit); it will be passed to the sigmoid function
        during evaluation, and handled as a logit by the appropriate loss function.
        '''
        ### TODO
        # Get BERT hidden states for both sentences
        pooler_output_1 = self.forward(input_ids_1, attention_mask_1)
        pooler_output_2 = self.forward(input_ids_2, attention_mask_2)

        # TODO: Add the difference of two outputs and concat them together along with two pooler outputs
        pooler_diff = torch.abs(torch.sub(pooler_output_1, pooler_output_2))
        # pooler_diff_norm = torch.linalg.vector_norm(pooler_diff)
        concat_out = torch.cat((pooler_output_1, pooler_output_2, pooler_diff), dim=1)
        # concat_out = torch.cat((pooler_output_1, pooler_output_2), dim=1)

        # TODO: check dimension to get only one number output -- passed

        ############# New approach: perform cosine similarity instead of dense layer ############
        #### Result is worse when also put this parallel to result induced from concat_out ###
        # cos = nn.CosineSimilarity()
        # sim_score = cos(pooler_output_1, pooler_output_2)
        # NEW: put this number also in the concat output

        # Get paraphrase logits
        similarity_logits = self.dropout(concat_out)
        similarity_logits = self.similarity_layer(similarity_logits)
        similarity_logits = self.dropout(similarity_logits)
        # similarity_logits2 = self.similarity_layer2(torch.cat((similarity_logits, sim_score.unsqueeze(1)), dim=1))
        similarity_logits2 = self.similarity_layer2(similarity_logits)
        # similarity_logits2 = self.similarity_layer2(torch.cat((similarity_logits, sim_score.unsqueeze(1)), dim=1))

        # Return similarity logits
        return similarity_logits2

def save_model(model, optimizer, args, config, filepath):
    save_info = {
        'model': model.state_dict(),
        'optim': optimizer.state_dict(),
        'args': args,
        'model_config': config,
        'system_rng': random.getstate(),
        'numpy_rng': np.random.get_state(),
        'torch_rng': torch.random.get_rng_state(),
    }

    torch.save(save_info, filepath)
    print(f"save the model to {filepath}")


## Currently only trains on sst dataset
def train_multitask(args):
    device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
    # Load data
    # Create the data and its corresponding datasets and dataloader
    sst_train_data, num_labels,para_train_data, sts_train_data = load_multitask_data_merged(
        args.sst_train, args.sent_additional, args.para_train, args.para_additional, args.sts_train, args.sim_additional, split ='train')
    sst_dev_data, num_labels,para_dev_data, sts_dev_data = load_multitask_data(args.sst_dev,args.para_dev,args.sts_dev, split ='train')

    sst_train_data = SentenceClassificationDataset(sst_train_data, args)
    sst_dev_data = SentenceClassificationDataset(sst_dev_data, args)
    para_train_data = SentencePairDataset(para_train_data, args)
    para_dev_data = SentencePairDataset(para_dev_data, args) 
    sts_train_data = SentencePairDataset(sts_train_data, args)
    sts_dev_data = SentencePairDataset(sts_dev_data, args, isRegression=True)


    sst_train_dataloader = DataLoader(sst_train_data, shuffle=True, batch_size=args.batch_size,
                                      collate_fn=sst_train_data.collate_fn)
    sst_dev_dataloader = DataLoader(sst_dev_data, shuffle=False, batch_size=args.batch_size,
                                    collate_fn=sst_dev_data.collate_fn)
    ##### Added #####
    para_train_dataloader = DataLoader(para_train_data, shuffle=True, batch_size=args.batch_size,
                                      collate_fn=para_train_data.collate_fn) 
    para_dev_dataloader = DataLoader(para_dev_data, shuffle=False, batch_size=args.batch_size,
                                    collate_fn=para_dev_data.collate_fn)
    sts_train_dataloader = DataLoader(sts_train_data, shuffle=True, batch_size=args.batch_size,
                                      collate_fn=sts_train_data.collate_fn) 
    sts_dev_dataloader = DataLoader(sts_dev_data, shuffle=False, batch_size=args.batch_size,
                                    collate_fn=sts_dev_data.collate_fn)
    ##################

    # ################## Load additional datasets ####################
    # sent_additional_data, num_labels_additional, para_additional_data, sim_additional_data = load_multitask_data_additional(
    #     args.sent_additional,args.para_additional,args.sim_additional, split ='train')
    # sent_additional_data = SentenceClassificationDataset(sent_additional_data, args)
    # para_additional_data = SentencePairDataset(para_additional_data, args)
    # sim_additional_data = SentencePairDataset(sim_additional_data, args)
    # sent_additional_dataloader = DataLoader(sent_additional_data, shuffle=True, batch_size=args.batch_size,
    #                                   collate_fn=sent_additional_data.collate_fn)
    # para_additional_dataloader = DataLoader(para_additional_data, shuffle=True, batch_size=args.batch_size,
    #                                   collate_fn=para_additional_data.collate_fn)
    # sim_additional_dataloader = DataLoader(sim_additional_data, shuffle=True, batch_size=args.batch_size,
    #                                   collate_fn=sim_additional_data.collate_fn)
    # #################################################################


    # Init model
    config = {'hidden_dropout_prob': args.hidden_dropout_prob,
              'num_labels': num_labels,
              'hidden_size': 768,
              'data_dir': '.',
              'option': args.option}

    config = SimpleNamespace(**config)

    model = MultitaskBERT(config)
    model = model.to(device)

    lr = args.lr
    optimizer = AdamW(model.parameters(), lr=lr)

    ######## Add warmup to learning rate scheduler, and apply linear decay after warmup ######
    warmup_steps = int(args.epochs * WARMUP_PERCENTAGE)
    def warmup(current_step):
        if current_step < warmup_steps:
            return float(2**current_step / 2**warmup_steps)
        else:
            return max(0.0, float(args.epochs - current_step) / float(max(1, args.epochs - warmup_steps)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup)
    ##########################################################################################

    best_dev_acc = 0

    # Run for the specified number of epochs
    for epoch in range(args.epochs):
        print("-------------------- Training Epoch {} --------------------".format(epoch))
        model.train()
        train_loss = 0
        num_batches = 0

        # Sentiment classification
        for batch in tqdm(sst_train_dataloader, desc=f'train-{epoch}', disable=TQDM_DISABLE):
            b_ids, b_mask, b_labels = (batch['token_ids'],
                                       batch['attention_mask'], batch['labels'])

            b_ids = b_ids.to(device)
            b_mask = b_mask.to(device)
            b_labels = b_labels.to(device)

            optimizer.zero_grad()
            logits = model.predict_sentiment(b_ids, b_mask)
            loss = F.cross_entropy(logits, b_labels.view(-1), reduction='sum') / args.batch_size

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            num_batches += 1

        # ################### Train on additional dataset #################
        # for batch in tqdm(sent_additional_dataloader, desc=f'train-{epoch}', disable=TQDM_DISABLE):
        #     b_ids, b_mask, b_labels = (batch['token_ids'],
        #                                batch['attention_mask'], batch['labels'])

        #     b_ids = b_ids.to(device)
        #     b_mask = b_mask.to(device)
        #     b_labels = b_labels.to(device)

        #     optimizer.zero_grad()
        #     logits = model.predict_sentiment(b_ids, b_mask)
        #     loss = F.cross_entropy(logits, b_labels.view(-1), reduction='sum') / args.batch_size

        #     loss.backward()
        #     optimizer.step()

        #     train_loss += loss.item()
        #     num_batches += 1
        # ##################################################################

        # Paraphrase
        for batch in tqdm(para_train_dataloader, desc=f'train-{epoch}', disable=TQDM_DISABLE):
            (b_ids1, b_mask1,
            b_ids2, b_mask2,
            b_labels2, b_sent_ids) = (batch['token_ids_1'], batch['attention_mask_1'],
                        batch['token_ids_2'], batch['attention_mask_2'],
                        batch['labels'], batch['sent_ids'])

            b_ids1 = b_ids1.to(device)
            b_mask1 = b_mask1.to(device)
            b_ids2 = b_ids2.to(device)
            b_mask2 = b_mask2.to(device)
            b_labels2 = b_labels2.to(device)

            optimizer.zero_grad()
            logits2 = model.predict_paraphrase(b_ids1, b_mask1, b_ids2, b_mask2)
            loss2 = F.binary_cross_entropy_with_logits(logits2, b_labels2.float().unsqueeze(1), reduction='sum') / args.batch_size

            loss2.backward()
            optimizer.step()
            num_batches += 1
            train_loss += loss2.item()

        # ################# Train on additional dataset ##################
        # for batch in tqdm(para_additional_dataloader, desc=f'train-{epoch}', disable=TQDM_DISABLE):
        #     (b_ids1, b_mask1,
        #     b_ids2, b_mask2,
        #     b_labels2, b_sent_ids) = (batch['token_ids_1'], batch['attention_mask_1'],
        #                 batch['token_ids_2'], batch['attention_mask_2'],
        #                 batch['labels'], batch['sent_ids'])

        #     b_ids1 = b_ids1.to(device)
        #     b_mask1 = b_mask1.to(device)
        #     b_ids2 = b_ids2.to(device)
        #     b_mask2 = b_mask2.to(device)
        #     b_labels2 = b_labels2.to(device)

        #     optimizer.zero_grad()
        #     logits2 = model.predict_paraphrase(b_ids1, b_mask1, b_ids2, b_mask2)
        #     loss2 = F.binary_cross_entropy_with_logits(logits2, b_labels2.float().unsqueeze(1), reduction='sum') / args.batch_size

        #     loss2.backward()
        #     optimizer.step()
        #     num_batches += 1
        #     train_loss += loss2.item()
        # ##################################################################
        
        # Similarity
        for batch in tqdm(sts_train_dataloader, desc=f'train-{epoch}', disable=TQDM_DISABLE):
            (b_ids1, b_mask1,
            b_ids2, b_mask2,
            b_labels3, b_sent_ids) = (batch['token_ids_1'], batch['attention_mask_1'],
                        batch['token_ids_2'], batch['attention_mask_2'],
                        batch['labels'], batch['sent_ids'])

            b_ids1 = b_ids1.to(device)
            b_mask1 = b_mask1.to(device)
            b_ids2 = b_ids2.to(device)
            b_mask2 = b_mask2.to(device)
            b_labels3 = b_labels3.to(device)

            optimizer.zero_grad()
            logits3 = model.predict_similarity(b_ids1, b_mask1, b_ids2, b_mask2)

            mse = F.mse_loss
            # Normalize [0, 5] to [-1, 1] to align with cosine similarity
            loss3 = mse(logits3.flatten(), (b_labels3.float()/2.5-1)) / args.batch_size
            loss3.backward()
            optimizer.step()
            num_batches += 1
            train_loss += loss3.item()

        # ################## Train on additional dataset #######################
        # for batch in tqdm(sim_additional_dataloader, desc=f'train-{epoch}', disable=TQDM_DISABLE):
        #     (b_ids1, b_mask1,
        #     b_ids2, b_mask2,
        #     b_labels3, b_sent_ids) = (batch['token_ids_1'], batch['attention_mask_1'],
        #                 batch['token_ids_2'], batch['attention_mask_2'],
        #                 batch['labels'], batch['sent_ids'])

        #     b_ids1 = b_ids1.to(device)
        #     b_mask1 = b_mask1.to(device)
        #     b_ids2 = b_ids2.to(device)
        #     b_mask2 = b_mask2.to(device)
        #     b_labels3 = b_labels3.to(device)

        #     optimizer.zero_grad()
        #     logits3 = model.predict_similarity(b_ids1, b_mask1, b_ids2, b_mask2)

        #     mse = F.mse_loss
        #     # Normalize [0, 5] to [-1, 1] to align with cosine similarity
        #     loss3 = mse(logits3.flatten(), (b_labels3.float()/2.5-1)) / args.batch_size
        #     loss3.backward()
        #     optimizer.step()
        #     num_batches += 1
        #     train_loss += loss3.item()
        # ################################################################################
            
        # Change learning rate
        scheduler.step()

        # # Modify criterion #
        train_loss = train_loss / (num_batches)
        print(f"Epoch {epoch}: train loss :: {train_loss :.3f}")

        print("EPOCH {}, training accuracy:".format(epoch))
        _ = model_eval_multitask(sst_train_dataloader, para_train_dataloader, sts_train_dataloader, model, device)
        print("EPOCH {}, validation accuracy:".format(epoch))
        paraphrase_accuracy, _, _, sentiment_accuracy, _, _, sts_corr, _, _ = model_eval_multitask(
            sst_dev_dataloader, para_dev_dataloader, sts_dev_dataloader, model, device)
        dev_acc = (paraphrase_accuracy + sentiment_accuracy + sts_corr) / 3
        if dev_acc > best_dev_acc:
            best_dev_acc = dev_acc
            save_model(model, optimizer, args, config, args.filepath)


def test_model(args):
    with torch.no_grad():
        device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
        saved = torch.load(args.filepath)
        config = saved['model_config']

        model = MultitaskBERT(config)
        model.load_state_dict(saved['model'])
        model = model.to(device)
        print(f"Loaded model to test from {args.filepath}")

        test_model_multitask(args, model, device)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sst_train", type=str, default="data/ids-sst-train.csv")
    parser.add_argument("--sst_dev", type=str, default="data/ids-sst-dev.csv")
    parser.add_argument("--sst_test", type=str, default="data/ids-sst-test-student.csv")

    parser.add_argument("--para_train", type=str, default="data/quora-train.csv")
    parser.add_argument("--para_dev", type=str, default="data/quora-dev.csv")
    parser.add_argument("--para_test", type=str, default="data/quora-test-student.csv")

    parser.add_argument("--sts_train", type=str, default="data/sts-train.csv")
    parser.add_argument("--sts_dev", type=str, default="data/sts-dev.csv")
    parser.add_argument("--sts_test", type=str, default="data/sts-test-student.csv")

    ############# Add in additional dataset #############
    parser.add_argument("--sent_additional", type=str, default="data/sent_yelp.csv")
    parser.add_argument("--para_additional", type=str, default="data/para_adv.csv")
    parser.add_argument("--sim_additional", type=str, default="data/sim_sick.csv")
    #####################################################

    parser.add_argument("--seed", type=int, default=11711)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--option", type=str,
                        help='pretrain: the BERT parameters are frozen; finetune: BERT parameters are updated',
                        choices=('pretrain', 'finetune'), default="pretrain")
    parser.add_argument("--use_gpu", action='store_true')

    parser.add_argument("--sst_dev_out", type=str, default="predictions/sst-dev-output.csv")
    parser.add_argument("--sst_test_out", type=str, default="predictions/sst-test-output.csv")

    parser.add_argument("--para_dev_out", type=str, default="predictions/para-dev-output.csv")
    parser.add_argument("--para_test_out", type=str, default="predictions/para-test-output.csv")

    parser.add_argument("--sts_dev_out", type=str, default="predictions/sts-dev-output.csv")
    parser.add_argument("--sts_test_out", type=str, default="predictions/sts-test-output.csv")

    # hyper parameters
    parser.add_argument("--batch_size", help='sst: 64, cfimdb: 8 can fit a 12GB GPU', type=int, default=8)
    parser.add_argument("--hidden_dropout_prob", type=float, default=0.3)
    parser.add_argument("--lr", type=float, help="learning rate, default lr for 'pretrain': 1e-3, 'finetune': 1e-5",
                        default=1e-5)

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_args()
    args.filepath = f'additional-data-{args.option}-{args.epochs}-{args.lr}-multitask.pt' # save path
    seed_everything(args.seed)  # fix the seed for reproducibility
    train_multitask(args)
    test_model(args)