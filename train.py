import os
import sys
import copy
import random
import logging
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn import metrics
from time import strftime, localtime
from torch.utils.data import DataLoader
from transformers import RobertaModel
from torch.optim import AdamW
from models.dce_tmt import DCE_TMT
from models.layers import SupConLoss
from data_utils import SentenceDataset, build_tokenizer, build_embedding_matrix, Tokenizer4BertGCN, ABSAGCNData 
from prepare_vocab import VocabHelp 
from transformers.optimization import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup 

logger = logging.getLogger(__name__) 
logger.setLevel(logging.INFO) 
logger.addHandler(logging.StreamHandler(sys.stdout)) 

def setup_seed(seed): 
    torch.manual_seed(seed) 
    torch.cuda.manual_seed_all(seed) 
    np.random.seed(seed) 
    random.seed(seed) 
    torch.backends.cudnn.deterministic = True 

class Instructor:

    def __init__(self, opt):
        self.opt = opt

        print("Using RoBERTa branch...")
        LOCAL_ROBERTA_PATH = "/root/DCEGTF/models/Roberta"
        tokenizer = Tokenizer4BertGCN(opt.max_length)
        bert = RobertaModel.from_pretrained(LOCAL_ROBERTA_PATH)
        opt.inputs_cols = ['text_bert_indices', 'bert_segments_ids', 'attention_mask', 'adj_dep', 'src_mask', 'aspect_mask']
        pos_vocab = VocabHelp.load_vocab(opt.vocab_dir + '/vocab_post.vocab')
        dep_vocab = VocabHelp.load_vocab(opt.vocab_dir + '/vocab_dep.vocab')
        opt.deprel_size = len(dep_vocab) 
        self.model = opt.model_class(bert, opt).to(opt.device)
        trainset = ABSAGCNData(opt.dataset_file['train'], tokenizer, pos_vocab, dep_vocab, opt)
        testset  = ABSAGCNData(opt.dataset_file['test'],  tokenizer, pos_vocab, dep_vocab, opt)
        print("Trainset size:", len(trainset))
        print("Testset size:", len(testset))
        
        self.train_dataloader = DataLoader(dataset=trainset, batch_size=opt.batch_size, shuffle=True)
        self.test_dataloader = DataLoader(dataset=testset, batch_size=opt.batch_size)

    def _print_args(self):
        n_trainable_params, n_nontrainable_params = 0, 0
        for p in self.model.parameters():
            n_params = torch.prod(torch.tensor(p.shape))
            if p.requires_grad:
                n_trainable_params += n_params
            else:
                n_nontrainable_params += n_params
        logger.info('n_trainable_params: {0}, n_nontrainable_params: {1}'.format(n_trainable_params, n_nontrainable_params))
        logger.info('training arguments:')
        for arg in vars(self.opt):
            logger.info('>>> {0}: {1}'.format(arg, getattr(self.opt, arg)))
    
    def _reset_params(self): 
        for p in self.model.parameters(): 
            if p.requires_grad:
                if len(p.shape) > 1:
                    self.opt.initializer(p)
                else:
                    stdv = 1. / (p.shape[0]**0.5)
                    torch.nn.init.uniform_(p, a=-stdv, b=stdv)

    def get_bert_optimizer(self, model): 
        no_decay = ['bias', 'LayerNorm.weight'] 
        diff_part = ["bert.embeddings", "bert.encoder"] 
        logger.info("layered learning rate on") 
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if
                        not any(nd in n for nd in no_decay) and any(nd in n for nd in diff_part)],
                "weight_decay": self.opt.finetune_weight_decay,
                "lr": self.opt.bert_lr
            },
            {
                "params": [p for n, p in model.named_parameters() if
                        any(nd in n for nd in no_decay) and any(nd in n for nd in diff_part)],
                "weight_decay": 0.0,
                "lr": self.opt.bert_lr
            },
            {
                "params": [p for n, p in model.named_parameters() if
                        not any(nd in n for nd in no_decay) and not any(nd in n for nd in diff_part)],
                "weight_decay": self.opt.weight_decay,
                "lr": self.opt.learning_rate
            },
            {
                "params": [p for n, p in model.named_parameters() if
                        any(nd in n for nd in no_decay) and not any(nd in n for nd in diff_part)],
                "weight_decay": 0.0,
                "lr": self.opt.learning_rate
            },
        ] 
        optimizer = AdamW(optimizer_grouped_parameters, eps=self.opt.adam_epsilon) 
        return optimizer 
    
    def _train(self, criterion, optimizer, max_test_acc_overall=0):
        max_test_acc = 0
        max_f1 = 0
        global_step = 0
        model_path = '' 
        if self.opt.scheduler == 'cosine': 
            scheduler = get_cosine_schedule_with_warmup( 
                optimizer, int(self.opt.warmup*len(self.train_dataloader)), self.opt.num_epoch*len(self.train_dataloader)) 
        elif self.opt.scheduler == 'linear': 
            scheduler = get_linear_schedule_with_warmup( 
                optimizer, int(self.opt.warmup*len(self.train_dataloader)), self.opt.num_epoch*len(self.train_dataloader)) 
        elif self.opt.scheduler == 'none': 
            scheduler = None 
        for epoch in range(self.opt.num_epoch): 
            logger.info('>' * 60)
            logger.info('epoch: {}'.format(epoch))
            n_correct, n_total = 0, 0
            for i_batch, sample_batched in enumerate(self.train_dataloader):
                global_step += 1
                self.model.train()
                optimizer.zero_grad()
                inputs = [sample_batched[col].to(self.opt.device) for col in self.opt.inputs_cols]
                outputs, ate_logits = self.model(inputs) 
                targets = sample_batched['polarity'].to(self.opt.device)
                aspect_mask = inputs[-1]
                
                loss = criterion(outputs, targets)
                
                if hasattr(self.opt, 'use_mtl') and self.opt.use_mtl:
                    ate_loss = F.binary_cross_entropy_with_logits(ate_logits, aspect_mask.float())
                    loss = loss + self.opt.mtl_lambda * ate_loss
                
                if self.cl_criterion is not None:
                    cl_loss = self.cl_criterion(self.model.features, targets)
                    loss = loss + self.opt.cl_lambda * cl_loss
                
                loss.backward()
                optimizer.step() 
                if scheduler: 
                    scheduler.step() 
                if global_step % self.opt.log_step == 0:
                    n_correct += (torch.argmax(outputs, -1) == targets).sum().item()
                    n_total += len(outputs)
                    train_acc = n_correct / n_total 
                    test_acc, f1 = self._evaluate()
                    if test_acc > max_test_acc:
                        max_test_acc = test_acc
                        if test_acc > max_test_acc_overall:
                            if not os.path.exists('./state_dict'):
                                os.mkdir('./state_dict')
                            model_path = './state_dict/{}_{}_acc_{:.4f}_f1_{:.4f}'.format(self.opt.model_name, self.opt.dataset, test_acc, f1)
                            self.best_model = copy.deepcopy(self.model) 
                            logger.info('>> saved: {}'.format(model_path))
                    if f1 > max_f1:
                        max_f1 = f1
                    logger.info('loss: {:.4f}, acc: {:.4f}, test_acc: {:.4f}, f1: {:.4f}'.format(loss.item(), train_acc, test_acc, f1))
        return max_test_acc, max_f1, model_path
    
    def _evaluate(self, show_results=False):
        self.model.eval()
        n_test_correct, n_test_total = 0, 0
        targets_all, outputs_all = None, None
        with torch.no_grad():
            for batch, sample_batched in enumerate(self.test_dataloader):
                inputs = [sample_batched[col].to(self.opt.device) for col in self.opt.inputs_cols]
                targets = sample_batched['polarity'].to(self.opt.device)
                outputs, _ = self.model(inputs)
                n_test_correct += (torch.argmax(outputs, -1) == targets).sum().item()
                n_test_total += len(outputs)
                targets_all = torch.cat((targets_all, targets), dim=0) if targets_all is not None else targets
                outputs_all = torch.cat((outputs_all, outputs), dim=0) if outputs_all is not None else outputs
        test_acc = n_test_correct / n_test_total
        f1 = metrics.f1_score(targets_all.cpu(), torch.argmax(outputs_all, -1).cpu(), labels=[0, 1, 2], average='macro')
        labels = targets_all.data.cpu()
        predic = torch.argmax(outputs_all, -1).cpu()
        if show_results:
            report = metrics.classification_report(labels, predic, digits=4)
            confusion = metrics.confusion_matrix(labels, predic)
            return report, confusion, test_acc, f1
        return test_acc, f1 

    @torch.no_grad() 
    def _show_cases(self):
        self.model.eval() 
        cases_result = open("target_predict.txt", 'w') 
        for sample_batched in self.test_dataloader: 
            inputs = [sample_batched[col].to(self.opt.device) for col in self.opt.inputs_cols]
            targets = sample_batched['polarity'].to(self.opt.device)
            outputs, _ = self.model(inputs)
            predict = torch.argmax(outputs, -1) 
            for i in range(targets.size()[0]): 
                cases_result.write(str(targets[i].item()) ) 
                cases_result.write(", ") 
                cases_result.write(str(predict[i].item()) ) 
                cases_result.write("\n") 
        cases_result.close() 

    def _test(self):
        self.model = self.best_model
        self.model.eval()
        test_report, test_confusion, acc, f1 = self._evaluate(show_results=True)
        logger.info("Precision, Recall and F1-Score...")
        logger.info(test_report)
        logger.info("Confusion Matrix...")
        logger.info(test_confusion) 
    
    def run(self):
        self.cl_criterion = SupConLoss(temperature=self.opt.cl_temp) if hasattr(self.opt, 'use_cl') and self.opt.use_cl else None
        
        label_weights = torch.tensor([1, 1, 1.], device=self.opt.device) 
        if self.opt.balance_loss: 
            if self.opt.dataset == 'restaurant': 
                label_weights = torch.tensor([1/2164, 1/807, 1/637], device=self.opt.device)
            elif self.opt.dataset == 'laptop': 
                label_weights = torch.tensor([1/976, 1/851, 1/455], device=self.opt.device) 
            elif self.opt.dataset == 'twitter': 
                label_weights = torch.tensor([1/1507, 1/1528, 1/3016], device=self.opt.device) 
            elif self.opt.dataset == 'rest16': 
                label_weights = torch.tensor([1/1240, 1/439, 1/69], device=self.opt.device) 
        criterion = nn.CrossEntropyLoss(weight=label_weights) 
        if self.opt.model_name not in ['dce-tmt']:
            no_decay = ['bias', 'LayerNorm.weight'] 
            optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in self.model.named_parameters() if
                            not any(nd in n for nd in no_decay) and p.requires_grad ],
                    "weight_decay": self.opt.weight_decay,
                    "lr": self.opt.learning_rate 
                }, 
                {
                    "params": [p for n, p in self.model.named_parameters() if
                            any(nd in n for nd in no_decay) and p.requires_grad ], 
                    "weight_decay": 0.0, 
                    "lr": self.opt.learning_rate 
                } 
            ] 
            optimizer = self.opt.optimizer(optimizer_grouped_parameters) 
        else:
            optimizer = self.get_bert_optimizer(self.model) 
        max_test_acc_overall = 0
        max_f1_overall = 0
        max_test_acc, max_f1, model_path = self._train(criterion, optimizer, max_test_acc_overall)
        logger.info('max_test_acc: {0}, max_f1: {1}'.format(max_test_acc, max_f1))
        max_test_acc_overall = max(max_test_acc, max_test_acc_overall)
        max_f1_overall = max(max_f1, max_f1_overall)
        torch.save(self.best_model.state_dict(), model_path)
        logger.info('>> saved: {}'.format(model_path)) 
        logger.info('#' * 60) 
        logger.info('max_test_acc_overall:{}'.format(max_test_acc_overall)) 
        logger.info('max_f1_overall:{}'.format(max_f1_overall)) 
        self._test() 

def main():
    model_classes = {
        'dce-tmt': DCE_TMT, 
    } 

    vocab_dirs = {
        'restaurant': './dataset/Restaurants_corenlp',
        'laptop': './dataset/Laptops_corenlp',
        'twitter': './dataset/Tweets_corenlp',
        'rest16': './dataset/Restaurants16', 
    } 
    
    dataset_files = {
        'restaurant': {
            'train': './dataset/Restaurants_corenlp/train.json',
            'test': './dataset/Restaurants_corenlp/test.json',
        },
        'laptop': {
            'train': './dataset/Laptops_corenlp/train.json',
            'test': './dataset/Laptops_corenlp/test.json'
        },
        'twitter': {
            'train': './dataset/Tweets_corenlp/train.json',
            'test': './dataset/Tweets_corenlp/test.json',
        }, 
        'rest16': { 
            'train': './dataset/Restaurants16/train.json', 
            'test': './dataset/Restaurants16/test.json', 
        } 
    }
    
    input_colses = { 
        'non-bert': ['text', 'aspect', 'pos', 'head', 'deprel', 'post', 'mask', 'length', 'adj'],
        'bert': ['text_bert_indices', 'bert_segments_ids', 'attention_mask', 'asp_start', 'asp_end', 'adj_matrix', 'src_mask', 'aspect_mask'] 
    } 
    
    initializers = { 
        'xavier_uniform_': torch.nn.init.xavier_uniform_, 
        'xavier_normal_': torch.nn.init.xavier_normal_,
        'orthogonal_': torch.nn.init.orthogonal_,
    }
    
    optimizers = {
        'adadelta': torch.optim.Adadelta,
        'adagrad': torch.optim.Adagrad, 
        'adam': torch.optim.Adam,
        'adamW': torch.optim.AdamW,
        'adamax': torch.optim.Adamax, 
        'asgd': torch.optim.ASGD,
        'rmsprop': torch.optim.RMSprop,
        'sgd': torch.optim.SGD,
    }
    
    parser = argparse.ArgumentParser() 
    parser.add_argument('--model_name', default='dce-tmt', type=str, help='dce-tmt')
    parser.add_argument('--dataset', default='laptop', type=str, help='laptop, restaurant, twitter, rest16')
    parser.add_argument('--optimizer', default='adamW', type=str, help='adamW')
    parser.add_argument('--initializer', default='xavier_uniform_', type=str, help='xavier_uniform_') 
    parser.add_argument('--learning_rate', default=0.001, type=float) 
    parser.add_argument('--num_epoch', default=20, type=int) 
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--log_step', default=5, type=int)
    parser.add_argument('--embed_dim', default=300, type=int)
    parser.add_argument('--post_dim', type=int, default=60, help='Position embedding dimension.')
    parser.add_argument('--pos_dim', type=int, default=30, help='Pos embedding dimension.') 
    parser.add_argument('--deprel_dim', type=int, default=30, help='Dependent relation embedding dimension.')
    parser.add_argument('--hidden_dim', type=int, default=768, help='GCN mem dim.') 
    parser.add_argument('--polarities_dim', default=3, type=int, help='3') 
    parser.add_argument('--num_layers', default=2, type=int, help='Number of graph layers.')
    parser.add_argument('--input_dropout', type=float, default=0.7, help='Input dropout rate.')
    parser.add_argument('--lower', default=True, help='Lowercase all words.')
    parser.add_argument('--directed', default=False, help='directed graph or undirected graph')
    parser.add_argument('--add_self_loop', default=True) 
    parser.add_argument('--use_rnn', action='store_true') 
    parser.add_argument('--bidirect', default=True, help='Do use bi-RNN layer.')
    parser.add_argument('--rnn_hidden', type=int, default=60, help='RNN hidden state size.')
    parser.add_argument('--rnn_layers', type=int, default=1, help='Number of RNN layers.')
    parser.add_argument('--rnn_dropout', type=float, default=0.1, help='RNN dropout rate.') 
    parser.add_argument('--max_length', default=85, type=int) 
    parser.add_argument('--device', default=None, type=str, help='cpu, cuda')
    parser.add_argument('--seed', default=1000, type=int)
    parser.add_argument("--weight_decay", default=1e-4, type=float, help="Weight deay if we apply some.") 
    parser.add_argument('--vocab_dir', type=str, default='./dataset/Restaurants_corenlp') 
    parser.add_argument('--pad_id', default=0, type=int) 
    parser.add_argument('--graph_conv_type', type=str, default='hgnn', choices=['eela', 'gcn', 'gin', 'gat', 'dec-gcn', 'hgnn'])
    parser.add_argument('--graph_conv_attention_heads', default=4, type=int) 
    parser.add_argument('--graph_conv_attn_dropout', type=float, default=0.0) 
    parser.add_argument('--attention_heads', default=4, type=int) 
    parser.add_argument('--attn_dropout', type=float, default=0.1) 
    parser.add_argument('--ffn_dropout', type=float, default=0.3) 
    parser.add_argument('--norm', type=str, default='ln', choices=['ln', 'bn']) 
    parser.add_argument('--max_position', type=int, default=9) 
    parser.add_argument('--scheduler', type=str, default='none', choices=['linear', 'cosine', 'none']) 
    parser.add_argument('--warmup', type=float, default=2) 
    parser.add_argument('--use_cl', action='store_true', help='Use Supervised Contrastive Learning')
    parser.add_argument('--cl_temp', type=float, default=0.07, help='Temperature for SupCon Loss')
    parser.add_argument('--cl_lambda', type=float, default=0.1, help='Weight for SupCon Loss')
    parser.add_argument('--use_mtl', action='store_true', help='Use Multi-task Learning (ATE)')
    parser.add_argument('--mtl_lambda', type=float, default=0.1, help='Weight for ATE Loss')
    parser.add_argument('--use_knowledge', action='store_true', help='Use External Knowledge (SenticNet)')
    parser.add_argument('--balance_loss', action='store_true') 
    parser.add_argument('--cuda', default='0', type=str) 
    parser.add_argument('--lambda_sent', default=0.1, type=float)
    parser.add_argument('--pretrained_bert_name', default='Roberta', type=str)
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument('--bert_dim', type=int, default=768) 
    parser.add_argument('--bert_dropout', type=float, default=0.5, help='BERT dropout rate.')
    parser.add_argument('--bert_lr', default=2e-5, type=float) 
    parser.add_argument("--finetune_weight_decay", default=0.01, type=float) 
    parser.add_argument('--early_stop', default=5, type=int, help='Early stopping patience.')
    parser.add_argument('--max_grad_norm', default=1.0, type=float, help='Max gradient norm.')
    parser.add_argument('--srd', type=int, default=3, help='Semantic Relative Distance for LCF')
    
    opt = parser.parse_args()
        
    opt.model_class = model_classes[opt.model_name] 
    opt.dataset_file = dataset_files[opt.dataset]
    opt.initializer = initializers[opt.initializer]
    opt.optimizer = optimizers[opt.optimizer] 
    opt.vocab_dir = vocab_dirs[opt.dataset] 

    if 'bert' in opt.model_name: 
        opt.inputs_cols = input_colses['bert']
        opt.max_length = 100 
        opt.num_epoch = 6
    else: 
        opt.inputs_cols = input_colses['non-bert'] 
        opt.max_length = 85 
        opt.num_epoch = 50 

    if opt.cuda == '-1':
        opt.device = torch.device('cpu')
        print("Using CPU for training")
    else:
        opt.device = torch.device('cuda:'+opt.cuda)
        print(f"Using GPU {opt.cuda} for training")
    
    setup_seed(opt.seed) 

    if not os.path.exists('./logging'): 
        os.makedirs('./logging', mode=0o777) 
    log_file = '{}-{}-{}.log'.format(opt.model_name, opt.dataset, strftime("%Y-%m-%d_%H%M%S", localtime()))
    logger.addHandler(logging.FileHandler("%s/%s" % ('./logging', log_file)))

    ins = Instructor(opt)
    ins.run()

if __name__ == '__main__': 
    main()