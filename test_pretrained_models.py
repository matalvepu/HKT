import argparse
import csv
import logging
import os
import random
import pickle
import sys
from global_config import *
import numpy as np

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
    "--model", type=str, choices=["language_only", "acoustic_only", "visual_only","hcf_only","HKT"], default="HKT",
)

parser.add_argument("--dataset", type=str, choices=["humor", "sarcasm"], default="sarcasm")#humor=UR-FUNNY, sarcasm=MUsTARD
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--max_seq_length", type=int, default=77)
parser.add_argument("--max_concept_length", type=int, default=5)
parser.add_argument("--n_layers", type=int, default=1)
parser.add_argument("--n_heads", type=int, default=1)
parser.add_argument("--cross_n_layers", type=int, default=1)
parser.add_argument("--cross_n_heads", type=int, default=2)
parser.add_argument("--fusion_dim", type=int, default=172)
parser.add_argument("--dropout", type=float, default=0.09379)
parser.add_argument("--seed", type=int, default=5149)

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
        acoustic=np.take(acoustic, acoustic_features_list,axis=1)
        
        visual_padding = np.zeros(
            (args.max_seq_length - len(input_ids), visual.shape[1])
        )
        visual = np.concatenate((visual, visual_padding))
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
        os.path.join(DATASET_LOCATION, args.dataset, data_file),
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
    
    return (train_dataloader, dev_dataloader, test_dataloader)



def get_model():
    
    if args.model == "HKT" :
        
        if args.dataset=="humor":
            visual_model = Transformer(VISUAL_DIM, num_layers=7, nhead=3, dim_feedforward= 128)
            acoustic_model = Transformer(ACOUSTIC_DIM, num_layers=8, nhead=3, dim_feedforward = 256)
            hcf_model = Transformer(HCF_DIM, num_layers=3, nhead=2, dim_feedforward = 128)
            text_model = AlbertModel.from_pretrained('albert-base-v2')
            model = HKT(text_model, visual_model, acoustic_model,hcf_model, args)
            model.load_state_dict(torch.load("./model_weights/best/humor/humorHKT.pt"))
        elif args.dataset=="sarcasm":
            visual_model = Transformer(VISUAL_DIM, num_layers=8, nhead=4, dim_feedforward=1024)
            acoustic_model = Transformer(ACOUSTIC_DIM, num_layers=1, nhead=3, dim_feedforward=512)
            hcf_model = Transformer(HCF_DIM, num_layers=8, nhead=4, dim_feedforward=128)    
            text_model = AlbertModel.from_pretrained("albert-base-v2")
            model = HKT(text_model, visual_model, acoustic_model, hcf_model, args)
            model.load_state_dict(torch.load("./model_weights/best/sarcasm/sarcasmHKT.pt"))
            
    
    
    model.to(DEVICE)
    
    return model


def test_epoch(model, data_loader, loss_fct):
    """ Epoch operation in evaluation phase """
    model.eval()

    eval_loss = 0.0
    nb_eval_steps = 0
    preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(
            data_loader, mininterval=2, desc="  - (Validation)   ", leave=False
        ):
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

            
            if args.model == "HKT":
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
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
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

    print("Accuracy, F score", accuracy, f_score)
    return accuracy, f_score


def set_random_seed(seed):
    """
    This function controls the randomness by setting seed in all the libraries we will use.
    """
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def main():
   
   set_random_seed(args.seed)
   (
        train_data_loader,
        dev_data_loader,
        test_data_loader,
    ) = set_up_data_loader()
   
   model = get_model()
   print("loaded")
   loss_fct = BCEWithLogitsLoss()
   test_score_model(model, test_data_loader, loss_fct)

if __name__ == "__main__":
    main()


