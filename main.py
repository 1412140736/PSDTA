import json
import pandas as pd
import torch
import numpy as np
import os
import random
# Utils
from utils.utils import DataLoader, compute_pna_degrees, virtual_screening, CustomWeightedRandomSampler
from utils.dataset import *  # data
from utils.trainer import Trainer
from utils.metrics import *
from utils.utils import extract_data_from_files
# Preprocessing
from utils import protein_init, ligand_init
# Model

import argparse
import ast

from BAN.ban_net import net
from utils.draw import Trainer_draw
from Protein_family.family_trainer import Trainer_family


parser.add_argument('--betas', type=tuple_type, default="(0.9,0.99)")
parser.add_argument('--batch_size', type=int, default=14)

parser.add_argument('--config_path', type=str, default='config.json')
parser.add_argument('--classification_task', type=bool, help='')

parser.add_argument('--device', type=str, default='cuda:0', help='')
parser.add_argument('--datafolder', type=str, default='./dataset/davis/', help='')

parser.add_argument('--epochs', type=int, default=200 , help='')
parser.add_argument('--evaluate_epoch', type=int, default=1)
parser.add_argument('--evaluate_step', type=int, default=500)
parser.add_argument('--eps', type=float, default=1e-5, help='')

parser.add_argument('--finetune_modules', type=list_type, default=None)

parser.add_argument('--lrate', type=float, default=1e-5,help=' ')

parser.add_argument('--result_path', type=str, default='', help='')

parser.add_argument('--seed', type=int, default=2)
parser.add_argument('--save_interpret', type=bool, default=True, help='')
parser.add_argument('--sampling_col', type=str, default='')

parser.add_argument('--total_iters', type=int, default=None)
parser.add_argument('--trained_model_path', type=str, default=' ', )

def tuple_type(s):
    try:
        value = ast.literal_eval(s)
        if not isinstance(value, tuple):
            raise ValueError
    except (ValueError, SyntaxError):
        raise argparse.ArgumentTypeError(f"Invalid tuple value: {s}")
    return value


def list_type(s):
    try:
        value = ast.literal_eval(s)
        if not isinstance(value, list):
            raise ValueError
    except (ValueError, SyntaxError):
        raise argparse.ArgumentTypeError(f"Invalid list value: {s}")
    return value


parser = argparse.ArgumentParser()
args = parser.parse_args()


if args.trained_model_path:
    with open(args.config_path, 'r') as f:
        config = json.load(f)
else:
    with open(os.path.join(args.trained_model_path, 'config.json'), 'r') as f:
        config = json.load(f)
# overwrite
config['optimizer']['lrate'] = args.lrate  #parser
config['optimizer']['eps'] = args.eps #1e-8  #parser
config['optimizer']['betas'] = args.betas  ##parser (0.9,0.999)
config['tasks']['regression_task'] = args.regression_task  #parser 回归任务

# device
device = torch.device(args.device)
if not os.path.exists(args.result_path):
    os.makedirs(args.result_path)  # 保存结果

model_path = os.path.join(args.result_path, 'save_model_seed{}'.format(args.seed))
if not os.path.exists(model_path):
    os.makedirs(model_path)  # 保存模型

interpret_path = os.path.join(args.result_path, 'interpretation_result_seed{}'.format(args.seed))
if not os.path.exists(interpret_path):
    os.makedirs(interpret_path)  # 保存解释结果

if args.epochs is not None and args.total_iters is not None:
    args.epochs = None

print(args)
with open(os.path.join(args.result_path, 'model_params.txt'), 'w') as f:
    f.write(str(args))

# seed initialize
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
random.seed(args.seed)
os.environ['PYTHONHASHSEED'] = str(args.seed)


## 2016 import files
train_file=os.path.join(args.datafolder, 'train.csv')
valid_file=os.path.join(args.datafolder, 'valid.csv')
test_file=os.path.join(args.datafolder, 'test.csv')  #test.csv
train_df = pd.read_csv(train_file)
test_df = pd.read_csv(test_file)
valid_path = os.path.join(args.datafolder, 'valid.csv')
valid_df = None
files = [train_file, valid_file, test_file]


if os.path.exists(valid_path):
    valid_df = pd.read_csv(valid_path)
    protein_tuples = extract_data_from_files(files)
    ligand_smiles = list(
        set(train_df['Ligand'].tolist() + test_df['Ligand'].tolist() + valid_df['Ligand'].tolist()))  # 配体SMILES串
else:
    protein_tuples = extract_data_from_files(files)
    ligand_smiles = list(set(train_df['Ligand'].tolist() + test_df['Ligand'].tolist()))

protein_path = os.path.join(args.datafolder, 'protein.pt')
if os.path.exists(protein_path):
    print('Loading Protein Graph data...')
    protein_dict = torch.load(protein_path)
else:
    print('Initialising Protein Sequence to Protein Graph...')
    protein_dict = protein_init(protein_tuples)
    torch.save(protein_dict, protein_path)

ligand_path = os.path.join(args.datafolder, 'ligand.pt')
if os.path.exists(ligand_path):
    print('Loading Ligand Graph data...')
    ligand_dict = torch.load(ligand_path)
else:
    print('Initialising Ligand SMILES to Ligand Graph...')
    ligand_dict = ligand_init(ligand_smiles)
    torch.save(ligand_dict, ligand_path)

torch.cuda.empty_cache()

train_shuffle = True
train_sampler = None

if args.sampling_col:
    train_weights = torch.from_numpy(train_df[args.sampling_col].values)


    def sampler_from_weights(weights):
        sampler = CustomWeightedRandomSampler(weights, len(weights), replacement=True)

        return sampler


    train_shuffle = False
    train_sampler = sampler_from_weights(train_weights)

if train_sampler is not None:
    print('shuffle should be False: ', train_shuffle)

train_dataset = ProteinMoleculeDataset(train_df, ligand_dict, protein_dict,
                                       device=args.device)
test_dataset = ProteinMoleculeDataset(test_df, ligand_dict, protein_dict, device=args.device)


train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=train_shuffle,
                          sampler=train_sampler, follow_batch=['mol_x', 'clique_x', 'prot_node_aa','tsml'])

test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                         follow_batch=['mol_x', 'clique_x', 'prot_node_aa'])

valid_dataset, valid_loader = None, None
if valid_df is not None:
    valid_dataset = ProteinMoleculeDataset(valid_df, ligand_dict, protein_dict, device=args.device)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False,
                              follow_batch=['mol_x', 'clique_x', 'prot_node_aa','tsml']
                              )


if not args.trained_model_path:
    degree_path = os.path.join(args.datafolder, 'degree.pt')
    if not os.path.exists(degree_path):
        print('Computing training data degrees for PNA')
        mol_deg, clique_deg, prot_deg = compute_pna_degrees(train_loader)
        degree_dict = {'ligand_deg': mol_deg, 'clique_deg': clique_deg, 'protein_deg': prot_deg}
    else:
        degree_dict = torch.load(degree_path)
        mol_deg, clique_deg, prot_deg = degree_dict['ligand_deg'], degree_dict['clique_deg'], degree_dict['protein_deg']

    torch.save(degree_dict, os.path.join(args.result_path, 'save_model_seed{}'.format(args.seed), 'degree.pt'))
else:
    degree_dict = torch.load(os.path.join(args.trained_model_path, 'save_model_seed2/degree.pt'))
    param_dict = os.path.join(args.trained_model_path, 'model_test.pt')
    mol_deg, prot_deg = degree_dict['ligand_deg'], degree_dict['protein_deg']

model = net(mol_deg, prot_deg,
            # MOLECULE
            mol_in_channels=config['params']['mol_in_channels'], prot_in_channels=config['params']['prot_in_channels'],
            prot_evo_channels=config['params']['prot_evo_channels'],
            hidden_channels=config['params']['hidden_channels'], pre_layers=config['params']['pre_layers'],
            post_layers=config['params']['post_layers'], aggregators=config['params']['aggregators'],
            scalers=config['params']['scalers'], total_layer=config['params']['total_layer'],
            K=config['params']['K'], heads=config['params']['heads'],
            dropout=config['params']['dropout'],
            dropout_attn_score=config['params']['dropout_attn_score'],
            # output
            regression_head=config['tasks']['regression_task'],
            classification_head=config['tasks']['classification_task'],
            multiclassification_head=config['tasks']['mclassification_task'],
            device=device).to(device)


model.reset_parameters()
if args.trained_model_path:
    model.load_state_dict(torch.load(param_dict, map_location=args.device), strict=False)
    print('Pretrained model loaded!!!')

nParams = sum([p.nelement() for p in model.parameters()])
print('Model loaded with number of parameters being:', str(nParams))

with open(os.path.join(args.result_path, 'save_model_seed{}'.format(args.seed), 'config.json'), 'w') as f:
    json.dump(config, f, indent=4)


evaluation_metric = 'mse'

engine = Trainer(model=model, lrate=config['optimizer']['lrate'], min_lrate=config['optimizer']['min_lrate'],
                 wdecay=config['optimizer']['weight_decay'], betas=config['optimizer']['betas'],
                 eps=config['optimizer']['eps'], amsgrad=config['optimizer']['amsgrad'],
                 clip=config['optimizer']['clip'], steps_per_epoch=len(train_loader),
                 num_epochs=args.epochs, total_iters=args.total_iters,
                 warmup_iters=config['optimizer']['warmup_iters'],
                 lr_decay_iters=config['optimizer']['lr_decay_iters'],
                 schedule_lr=config['optimizer']['schedule_lr'], regression_weight=1, classification_weight=1,
                 evaluate_metric=evaluation_metric, result_path=args.result_path, runid=args.seed,
                 finetune_modules=args.finetune_modules,
                 device=device)

print('-' * 50)
print('start training model')
if args.epochs:
    engine.train_epoch(train_loader, val_loader=valid_loader, test_loader=test_loader,
                       evaluate_epoch=args.evaluate_epoch)
else:
    engine.train_step(train_loader, val_loader=valid_loader, test_loader=test_loader, evaluate_step=args.evaluate_step)

print('finished training model')
print('-' * 50)

print('loading best checkpoint and predicting test data')
print('-' * 50)
stat_dict_path=os.path.join(args.result_path, 'save_model_seed{}'.format(args.seed), 'model.pt')
model.load_state_dict(torch.load())
screen_df = virtual_screening(test_df, model, test_loader,
                              result_path=os.path.join(args.result_path,
                                                       "interpretation_result_seed{}".format(args.seed)),
                              save_interpret=args.save_interpret,
                              ligand_dict=ligand_dict, device=args.device)

screen_df.to_csv(os.path.join(args.result_path, 'test_prediction_seed{}.csv'.format(args.seed)), index=False)

