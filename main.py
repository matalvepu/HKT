#!/usr/bin/env python2
# -*- coding: utf-8 -*-


import argparse
import csv
import logging
import os
import random
import pickle
import sys
from global_config import *
import numpy as np
import wandb 

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score, f1_score

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

from torch.nn import CrossEntropyLoss, L1Loss, BCEWithLogitsLoss
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import matthews_corrcoef
from transformers import (
    AlbertConfig,
    AlbertTokenizer,
    AlbertForSequenceClassification,
    BertForNextSentencePrediction,
    BertTokenizer,
    get_linear_schedule_with_warmup,
)
from models import *
from transformers.optimization import AdamW


def return_unk():
    return 0

parser = argparse.ArgumentParser()
parser.add_argument(
    "--model", type=str, choices=["HKT","language_only", "acoustic_only", "visual_only","hcf_only"], default="HKT",
)

parser.add_argument("--dataset", type=str, choices=["humor", "sarcasm"], default="sarcasm")
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--max_seq_length", type=int, default=85)
parser.add_argument("--n_layers", type=int, default=1)
parser.add_argument("--n_heads", type=int, default=1)
parser.add_argument("--cross_n_layers", type=int, default=1)
parser.add_argument("--cross_n_heads", type=int, default=4)
parser.add_argument("--fusion_dim", type=int, default=172)
parser.add_argument("--dropout", type=float, default=0.2366)
parser.add_argument("--epochs", type=int, default=20)

parser.add_argument("--seed", type=int, default=100)

parser.add_argument("--learning_rate", type=float, default=0.000005)
parser.add_argument("--learning_rate_a", type=float, default=0.003)
parser.add_argument("--learning_rate_h", type=float, default=0.0003)
parser.add_argument("--learning_rate_v", type=float, default=0.003)
parser.add_argument("--warmup_ratio", type=float, default=0.07178)
parser.add_argument("--save_weight", type=str, choices=["True","False"], default="False")


args = parser.parse_args()



class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, visual, acoustic,hcf,label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.visual = visual
        self.acoustic = acoustic
        self.hcf = hcf
        self.label_id = label_id

def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""
    pop_count = 0
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) == 0:
            tokens_b.pop()
        else:
            pop_count += 1
            tokens_a.pop(0)
    return pop_count

#albert tokenizer split words in to subwords. "_" marker helps to find thos sub words
#our acoustic and visual features are aligned on word level. So we just create copy the same 
#visual/acoustic vectors that belong to same word.
def get_inversion(tokens, SPIECE_MARKER="▁"):
    inversion_index = -1
    inversions = []
    for token in tokens:
        if SPIECE_MARKER in token:
            inversion_index += 1
        inversions.append(inversion_index)
    return inversions


def convert_humor_to_features(examples, tokenizer, punchline_only=False):
    features = []

    for (ex_index, example) in enumerate(examples):
        
        #p denotes punchline, c deontes context
        #hid is the utterance unique id. these id's are provided by the authors of urfunny and mustard
        #label is either 1/0 . 1=humor, 0=not humor
        (
            (p_words, p_visual, p_acoustic, p_hcf),
            (c_words, c_visual, c_acoustic, c_hcf),
            hid,
            label
        ) = example
                
        text_a = ". ".join(c_words)
        text_b = p_words + "."
        tokens_a = tokenizer.tokenize(text_a)
        tokens_b = tokenizer.tokenize(text_b)
        
        inversions_a = get_inversion(tokens_a)
        inversions_b = get_inversion(tokens_b)

        pop_count = _truncate_seq_pair(tokens_a, tokens_b, args.max_seq_length - 3)

        inversions_a = inversions_a[pop_count:]
        inversions_b = inversions_b[: len(tokens_b)]

        visual_a = []
        acoustic_a = []
        hcf_a=[]        
        #our acoustic and visual features are aligned on word level. So we just 
        #create copy of the same visual/acoustic vectors that belong to same word.
        #because ber tokenizer split word into subwords
        for inv_id in inversions_a:
            visual_a.append(c_visual[inv_id, :])
            acoustic_a.append(c_acoustic[inv_id, :])
            hcf_a.append(c_hcf[inv_id, :])
            


        visual_a = np.array(visual_a)
        acoustic_a = np.array(acoustic_a)
        hcf_a = np.array(hcf_a)
        
        visual_b = []
        acoustic_b = []
        hcf_b = []
        for inv_id in inversions_b:
            visual_b.append(p_visual[inv_id, :])
            acoustic_b.append(p_acoustic[inv_id, :])
            hcf_b.append(p_hcf[inv_id, :])
        
        visual_b = np.array(visual_b)
        acoustic_b = np.array(acoustic_b)
        hcf_b = np.array(hcf_b)
        
        tokens = ["[CLS]"] + tokens_a + ["[SEP]"] + tokens_b + ["[SEP]"]

        acoustic_zero = np.zeros((1, ACOUSTIC_DIM_ALL))
        if len(tokens_a) == 0:
            acoustic = np.concatenate(
                (acoustic_zero, acoustic_zero, acoustic_b, acoustic_zero)
            )
        else:
            acoustic = np.concatenate(
                (acoustic_zero, acoustic_a, acoustic_zero, acoustic_b, acoustic_zero)
            )

        visual_zero = np.zeros((1, VISUAL_DIM_ALL))
        if len(tokens_a) == 0:
            visual = np.concatenate((visual_zero, visual_zero, visual_b, visual_zero))
        else:
            visual = np.concatenate(
                (visual_zero, visual_a, visual_zero, visual_b, visual_zero)
            )
        
        
        hcf_zero = np.zeros((1,4))
        if len(tokens_a) == 0:
            hcf = np.concatenate((hcf_zero, hcf_zero, hcf_b, hcf_zero))
        else:
            hcf = np.concatenate(
                (hcf_zero, hcf_a, hcf_zero, hcf_b, hcf_zero)
                
            )
        
        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        segment_ids = [0] * (len(tokens_a) + 2) + [1] * (len(tokens_b) + 1)
        input_mask = [1] * len(input_ids)
            
        acoustic_padding = np.zeros(
            (args.max_seq_length - len(input_ids), acoustic.shape[1])
        )
        acoustic = np.concatenate((acoustic, acoustic_padding))
        #original urfunny acoustic feature dimension is 81.
        #we found many features are highly correllated. so we removed
        #highly correlated feature to reduce dimension
        acoustic=np.take(acoustic, acoustic_features_list,axis=1)
        
        visual_padding = np.zeros(
            (args.max_seq_length - len(input_ids), visual.shape[1])
        )
        visual = np.concatenate((visual, visual_padding))
        #original urfunny visual feature dimension is more than 300.
        #we only considred the action unit and face shape parameter features
        visual = np.take(visual, visual_features_list,axis=1)
        
        
        hcf_padding= np.zeros(
            (args.max_seq_length - len(input_ids), hcf.shape[1])
        )
        
        hcf = np.concatenate((hcf, hcf_padding))
        
        padding = [0] * (args.max_seq_length - len(input_ids))

        input_ids += padding
        input_mask += padding
        segment_ids += padding

        assert len(input_ids) == args.max_seq_length
        assert len(input_mask) == args.max_seq_length
        assert len(segment_ids) == args.max_seq_length
        assert acoustic.shape[0] == args.max_seq_length
        assert visual.shape[0] == args.max_seq_length
        assert hcf.shape[0] == args.max_seq_length
        
        label_id = float(label)
        
        
        features.append(
            InputFeatures(
                input_ids=input_ids,
                input_mask=input_mask,
                segment_ids=segment_ids,
                visual=visual,
                acoustic=acoustic,
                hcf=hcf,
                label_id=label_id,
            )
        )
            
    return features



def get_appropriate_dataset(data, tokenizer, parition):
    

    features = convert_humor_to_features(data, tokenizer)
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    all_visual = torch.tensor([f.visual for f in features], dtype=torch.float)
    all_acoustic = torch.tensor([f.acoustic for f in features], dtype=torch.float)
    hcf = torch.tensor([f.hcf for f in features], dtype=torch.float)
    all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.float)
    

    dataset = TensorDataset(
        all_input_ids,
        all_visual,
        all_acoustic,
        all_input_mask,
        all_segment_ids,
        hcf,
        all_label_ids,
    )
    
    return dataset


def set_up_data_loader():
    if args.dataset=="humor":
        data_file = "ur_funny.pkl"
    elif args.dataset=="sarcasm":
        data_file = "mustard.pkl"
        
    with open(
        os.path.join(DATASET_LOCATION, data_file),
        "rb",
    ) as handle:
        all_data = pickle.load(handle)
        
    train_data = all_data["train"]
    dev_data = all_data["dev"]
    test_data = all_data["test"]

    tokenizer = AlbertTokenizer.from_pretrained("albert-base-v2")

    train_dataset = get_appropriate_dataset(train_data, tokenizer, "train")
    dev_dataset = get_appropriate_dataset(dev_data, tokenizer, "dev")
    test_dataset = get_appropriate_dataset(test_data, tokenizer, "test")

    train_dataloader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=1
    )

    dev_dataloader = DataLoader(
        dev_dataset, batch_size=args.batch_size, shuffle=True, num_workers=1
    )

    test_dataloader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=True, num_workers=1
    )
    
    
    return train_dataloader, dev_dataloader, test_dataloader

def train_epoch(model, train_dataloader, optimizer, scheduler, loss_fct):
    model.train()
    tr_loss = 0
    nb_tr_examples, nb_tr_steps = 0, 0

    for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):

        batch = tuple(t.to(DEVICE) for t in batch)
        (
            input_ids,
            visual,
            acoustic,
            input_mask,
            segment_ids,
            hcf,
            label_ids
        ) = batch
        
        visual = torch.squeeze(visual, 1)
        acoustic = torch.squeeze(acoustic, 1)

        if args.model == "language_only":
            outputs = model(
                input_ids,
                token_type_ids=segment_ids,
                attention_mask=input_mask,
                labels=None,
            )
        elif args.model == "acoustic_only":
            outputs = model(
                acoustic
            )
        elif args.model == "visual_only":
            outputs = model(
                visual
            )
        elif args.model=="hcf_only":
            outputs=model(hcf)
            
        elif args.model=="HKT":
            outputs = model(input_ids, visual, acoustic,hcf, token_type_ids=segment_ids, attention_mask=input_mask,)
        
        
            
        logits = outputs[0]
        
        loss = loss_fct(logits.view(-1), label_ids.view(-1))

        tr_loss += loss.item()
        nb_tr_examples += input_ids.size(0)
        nb_tr_steps += 1

        loss.backward()
        
        for o_i in range(len(optimizer)):
            optimizer[o_i].step()
            scheduler[o_i].step()
        
        model.zero_grad()

    return tr_loss/nb_tr_steps



def eval_epoch(model, dev_dataloader, loss_fct):
    
    model.eval()
    dev_loss = 0
    nb_dev_examples, nb_dev_steps = 0, 0
    
    with torch.no_grad():
        for step, batch in enumerate(tqdm(dev_dataloader, desc="Iteration")):
            batch = tuple(t.to(DEVICE) for t in batch)
            (
                input_ids,
                visual,
                acoustic,
                input_mask,
                segment_ids,
                hcf,
                label_ids
            ) = batch
                    
            visual = torch.squeeze(visual, 1)
            acoustic = torch.squeeze(acoustic, 1)
    
            if args.model == "language_only":
                outputs = model(
                    input_ids,
                    token_type_ids=segment_ids,
                    attention_mask=input_mask,
                    labels=None,
                )
            elif args.model == "acoustic_only":
                outputs = model(
                    acoustic
                )
            elif args.model == "visual_only":
                outputs = model(
                    visual
                )
            elif args.model=="hcf_only":
                outputs=model(hcf)
                
            elif args.model=="HKT":
                outputs = model(input_ids, visual, acoustic,hcf, token_type_ids=segment_ids, attention_mask=input_mask,)
            
            
            logits = outputs[0]
            loss = loss_fct(logits.view(-1), label_ids.view(-1))
    
            dev_loss += loss.item()
            nb_dev_examples += input_ids.size(0)
            nb_dev_steps += 1

    return dev_loss/nb_dev_steps

def test_epoch(model, test_data_loader, loss_fct):
    """ Epoch operation in evaluation phase """
    model.eval()

    eval_loss = 0.0
    nb_eval_steps = 0
    preds = []
    all_labels = []

    with torch.no_grad():
        for step, batch in enumerate(tqdm(test_data_loader, desc="Iteration")):
            
            batch = tuple(t.to(DEVICE) for t in batch)

            (
                input_ids,
                visual,
                acoustic,
                input_mask,
                segment_ids,
                hcf,
                label_ids
            ) = batch
                    
            visual = torch.squeeze(visual, 1)
            acoustic = torch.squeeze(acoustic, 1)
            
            if args.model == "language_only":
                outputs = model(
                    input_ids,
                    token_type_ids=segment_ids,
                    attention_mask=input_mask,
                    labels=None,
                )
            elif args.model == "acoustic_only":
                outputs = model(
                    acoustic
                )
            elif args.model == "visual_only":
                outputs = model(
                    visual
                )
            elif args.model=="hcf_only":
                outputs=model(hcf)
                
            elif args.model=="HKT":
                outputs = model(input_ids, visual, acoustic,hcf, token_type_ids=segment_ids, attention_mask=input_mask,)
            
            
            logits = outputs[0]
            
            
            tmp_eval_loss = loss_fct(logits.view(-1), label_ids.view(-1))

            eval_loss += tmp_eval_loss.mean().item()
            nb_eval_steps += 1
            
            logits = torch.sigmoid(logits)
            
            if len(preds) == 0:
                preds=logits.detach().cpu().numpy()
                all_labels=label_ids.detach().cpu().numpy()
            else:
                preds= np.append(preds, logits.detach().cpu().numpy(), axis=0)
                all_labels = np.append(
                    all_labels, label_ids.detach().cpu().numpy(), axis=0
                )
                
                
                
        eval_loss = eval_loss / nb_eval_steps
        preds = np.squeeze(preds)
        all_labels = np.squeeze(all_labels)

    return preds, all_labels, eval_loss



def test_score_model(model, test_data_loader, loss_fct, exclude_zero=False):

    predictions, y_test, test_loss = test_epoch(model, test_data_loader, loss_fct)
    
    predictions = predictions.round()

    f_score = f1_score(y_test, predictions, average="weighted")
    accuracy = accuracy_score(y_test, predictions)

    print("Accuracy:", accuracy,"F score:", f_score)
    return accuracy, f_score, test_loss




def train(
    model,
    train_dataloader,
    dev_dataloader,
    test_dataloader,
    optimizer,
    scheduler,
    loss_fct,
):
       
    best_valid_loss = 9e+9
    run_name = str(wandb.run.id)
    valid_losses = []
    
    n_epochs=args.epochs
        
    
    for epoch_i in range(n_epochs):
        
        train_loss = train_epoch(
            model, train_dataloader, optimizer, scheduler, loss_fct
        )
        valid_loss = eval_epoch(model, dev_dataloader, loss_fct)

        valid_losses.append(valid_loss)
        print(
            "\nepoch:{},train_loss:{}, valid_loss:{}".format(
                epoch_i, train_loss, valid_loss
            )
        )

        test_accuracy, test_f_score, test_loss = test_score_model(
            model, test_dataloader, loss_fct
        )
        
            
        if(valid_loss <= best_valid_loss):
            best_valid_loss = valid_loss
            best_valid_test_accuracy = test_accuracy
            best_valid_test_fscore= test_f_score
            
            if(args.save_weight == "True"):
                torch.save(model.state_dict(),'./best_weights/'+run_name+'.pt')
        
        #we report test_accuracy of the best valid loss (best_valid_test_accuracy)
        wandb.log(
            {
                "train_loss": train_loss,
                "valid_loss": valid_loss,
                "test_loss": test_loss,
                "best_valid_loss": best_valid_loss,
                "best_valid_test_accuracy": best_valid_test_accuracy,
                "best_valid_test_fscore":best_valid_test_fscore
            }
        )
        



def get_optimizer_scheduler(params,num_training_steps,learning_rate=1e-5):
    
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in params if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.01,
        },
        {
            "params": [
                p for n, p in params if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    
    optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(num_training_steps * args.warmup_ratio),
        num_training_steps=num_training_steps,
    )
    
    return optimizer,scheduler

def prep_for_training(num_training_steps):
    
    
    if args.model == "language_only":
        model = AlbertForSequenceClassification.from_pretrained(
            "albert-base-v2", num_labels=1
        )
    elif args.model == "acoustic_only":
        model = Transformer(ACOUSTIC_DIM, num_layers=args.n_layers, nhead=args.n_heads, dim_feedforward=args.fc_dim)
        
    elif args.model == "visual_only":
        model = Transformer(VISUAL_DIM, num_layers=args.n_layers, nhead=args.n_heads, dim_feedforward=args.fc_dim)
        
    elif args.model=="hcf_only":
        model=Transformer(HCF_DIM, num_layers=args.n_layers, nhead=args.n_heads, dim_feedforward=args.fc_dim)
        
    elif args.model == "HKT" :
        #HKT model has 4 unimodal encoders. But the language one is ALBERT pretrained model. But other enocders are
        #trained from scratch with low level features. We have found that many times most of the the gardients flows to albert encoders only as it
        #already has rich contextual representation. So in the beginning the gradient flows ignores other encoders which are trained from low level features. 
        # We found that if we intitalize the weights of the acoustic, visual and hcf encoders of HKT model from the best unimodal models that we already ran for ablation study then
        #the model converege faster. Other wise it takes very long time to converge. 
        if args.dataset=="humor":
            visual_model = Transformer(VISUAL_DIM, num_layers=7, nhead=3, dim_feedforward= 128)
            visual_model.load_state_dict(torch.load("./model_weights/init/humor/humorVisualTransformer.pt"))
            acoustic_model = Transformer(ACOUSTIC_DIM, num_layers=8, nhead=3, dim_feedforward = 256)
            acoustic_model.load_state_dict(torch.load("./model_weights/init/humor/humorAcousticTransformer.pt"))
            hcf_model = Transformer(HCF_DIM, num_layers=3, nhead=2, dim_feedforward = 128)
            hcf_model.load_state_dict(torch.load("./model_weights/init/humor/humorHCFTransformer.pt"))
            
        elif args.dataset=="sarcasm":
            visual_model = Transformer(VISUAL_DIM, num_layers=8, nhead=4, dim_feedforward=1024)
            visual_model.load_state_dict(torch.load("./model_weights/init/sarcasm/sarcasmVisualTransformer.pt"))
            acoustic_model = Transformer(ACOUSTIC_DIM, num_layers=1, nhead=3, dim_feedforward=512)
            acoustic_model.load_state_dict(torch.load("./model_weights/init/sarcasm/sarcasmAcousticTransformer.pt"))
            hcf_model = Transformer(HCF_DIM, num_layers=8, nhead=4, dim_feedforward=128)
            hcf_model.load_state_dict(torch.load("./model_weights/init/sarcasm/sarcasmHCFTransformer.pt"))
        
        text_model = AlbertModel.from_pretrained('albert-base-v2')
        model = HKT(text_model, visual_model, acoustic_model,hcf_model, args)

    else:
        raise ValueError("Requested model is not available")

    model.to(DEVICE)
    
    loss_fct = BCEWithLogitsLoss()
    

    # Prepare optimizer
    # used different learning rates for different componenets.
    
    if args.model == "HKT" :
        
        acoustic_params,visual_params,hcf_params,other_params = model.get_params()
        optimizer_o,scheduler_o=get_optimizer_scheduler(other_params,num_training_steps,learning_rate=args.learning_rate)
        optimizer_h,scheduler_h=get_optimizer_scheduler(hcf_params,num_training_steps,learning_rate=args.learning_rate_h)
        optimizer_v,scheduler_v=get_optimizer_scheduler(visual_params,num_training_steps,learning_rate=args.learning_rate_v)
        optimizer_a,scheduler_a=get_optimizer_scheduler(acoustic_params,num_training_steps,learning_rate=args.learning_rate_a)
        
        optimizers=[optimizer_o,optimizer_h,optimizer_v,optimizer_a]
        schedulers=[scheduler_o,scheduler_h,scheduler_v,scheduler_a]
        
    else:
        params = list(model.named_parameters())

        optimizer_l, scheduler_l = get_optimizer_scheduler(
            params, num_training_steps, learning_rate=args.learning_rate
        )
        
        optimizers=[optimizer_l]
        schedulers=[scheduler_l]
        
        
    return model, optimizers, schedulers,loss_fct




def set_random_seed(seed):
    """
    This function controls the randomness by setting seed in all the libraries we will use.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True

    

def main():
    
    wandb.init(project="HKT")
    wandb.config.update(args)
    
    if(args.seed == -1):
        seed = random.randint(0, 9999)
        print("seed",seed)
    else:
        seed = args.seed
    
    wandb.config.update({"seed": seed}, allow_val_change=True)
    
    set_random_seed(seed)
    
    train_dataloader,dev_dataloader,test_dataloader=set_up_data_loader()
    print("Dataset Loaded: ",args.dataset)
    num_training_steps = len(train_dataloader) * args.epochs
    
    model, optimizers, schedulers, loss_fct = prep_for_training(
        num_training_steps
    )
    print("Model Loaded: ",args.model)
    train(
        model,
        train_dataloader,
        dev_dataloader,
        test_dataloader,
        optimizers,
        schedulers,
        loss_fct,
    )
    

if __name__ == "__main__":
    main()