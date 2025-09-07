import numpy as np
import pandas as pd
import sys

# Check if the code is running in a Jupyter notebook
if 'ipykernel' in sys.modules:
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm

import torch
import esm
from torch_geometric.utils import degree, add_self_loops, subgraph, to_undirected, remove_self_loops, coalesce
import networkx as nx
import math

from utils.esm_loader import load_esm_saprot
from utils.foldseek_util import get_struc_seq
import os


def protein_init(protein_tuples):

    model_location = "esm2_t33_650M_UR50D"
    model_A, alphabet_A = esm.pretrained.load_model_and_alphabet(model_location)
    model_A.eval()
    if torch.cuda.is_available():
        model_A = model_A.cuda()
    batch_converter_A = alphabet_A.get_batch_converter()

    model_path = "./checkpoint/SaProt_650M_AF2.pt"
    model, alphabet = load_esm_saprot(model_path)
    result_dict = {}  # 初始化空字典
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()
    batch_converter = alphabet.get_batch_converter()

    pdb_path = "./dataset/davis/PDB"  #./dataset/davis/PDB          ./example
    processed_proteins = set()
    for protein in tqdm(protein_tuples):  # ['ID', 'Protein', 'Target_Chain']
        pdb_id = protein[0]
        protein_seq = protein[1]
        if protein_seq in processed_proteins:
            continue  # 跳过已经处理过的 protein[0]
        pdb_path = os.path.join(pdb_path, f"{pdb_id}.pdb")
        Target_Chain = protein[2]
        print("Target_Chain:", Target_Chain)
        parsed_seqs = get_struc_seq("./bin/foldseek", pdb_path, [Target_Chain], plddt_mask=False)[Target_Chain]
        pdb_path = "./dataset/davis/PDB"  #./dataset/davis/PDB       ./example
        seq, foldseek_seq, combined_seq = parsed_seqs
        if (len(seq) == len(protein[1])):
            seq = seq
            combined_seq = combined_seq
        else:
            multiple = math.ceil(len(protein[1]) / len(seq))
            seq = seq * multiple
            combined_seq = combined_seq * multiple

        seq_feat = seq_feature(seq)  # 序列特征。形状为（氨基酸数量*33）

        # token_repr：形状为（氨基酸数量*1280）
        # contact_map_proba：形状为（氨基酸数量*氨基酸数量）
        # logits：形状为（氨基酸数量*layer(33)） 表示分类得分
        token_repr, contact_map_proba = esm_extract(model, batch_converter, combined_seq, layer=33,
                                                            approach='last',
                                                            dim=1280)

        _, contact_map_proba_A = esm_extract_A(model_A, batch_converter_A, seq, layer=33,
                                                    approach='last',
                                                    dim=1280)

        assert len(contact_map_proba) == len(seq)

        # edge_index：连接的氨基酸索引，形状为(连接氨基酸成对数*2)
        # edge_weight：连接的氨基酸权重，形状为(1*连接氨基酸成对数)
        edge_index, edge_weight = contact_map(contact_map_proba_A)

        num_nodes = len(seq)

        node_degrees = degree(edge_index[0], dtype=torch.long)#后加

        result_dict[protein[1]] = {
            'seq': seq,
            'seq_feat': torch.from_numpy(seq_feat),
            'token_representation': token_repr.half(),  # half:（16-bit 浮点数）
            'num_nodes': num_nodes,
            'num_pos': torch.arange(len(seq)).reshape(-1, 1),  # 序列中每个位置的索引.形状为 (序列长度, 1)
            'edge_index': edge_index,
            'edge_weight': edge_weight,
            'degree': node_degrees, #每个结点的度 1*N

        }
        processed_proteins.add(protein_seq)

    return result_dict


# normalize
def dic_normalize(dic):
    # print(dic)
    max_value = dic[max(dic, key=dic.get)]
    min_value = dic[min(dic, key=dic.get)]
    # print(max_value)
    interval = float(max_value) - float(min_value)
    for key in dic.keys():
        dic[key] = (dic[key] - min_value) / interval
    dic['X'] = (max_value + min_value) / 2.0
    return dic


pro_res_table = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y',
                 'X']

pro_res_aliphatic_table = ['A', 'I', 'L', 'M', 'V']  # 脂肪族氨基酸
pro_res_aromatic_table = ['F', 'W', 'Y']  # 芳香族氨基酸
pro_res_polar_neutral_table = ['C', 'N', 'Q', 'S', 'T']  # 极性中性氨基酸
pro_res_acidic_charged_table = ['D', 'E']  # 酸性带电氨基酸
pro_res_basic_charged_table = ['H', 'K', 'R']  # 碱性带电氨基酸

res_weight_table = {'A': 71.08, 'C': 103.15, 'D': 115.09, 'E': 129.12, 'F': 147.18, 'G': 57.05, 'H': 137.14,
                    'I': 113.16, 'K': 128.18, 'L': 113.16, 'M': 131.20, 'N': 114.11, 'P': 97.12, 'Q': 128.13,
                    'R': 156.19, 'S': 87.08, 'T': 101.11, 'V': 99.13, 'W': 186.22, 'Y': 163.18}  # 氨基酸的相对分子质量

res_pka_table = {'A': 2.34, 'C': 1.96, 'D': 1.88, 'E': 2.19, 'F': 1.83, 'G': 2.34, 'H': 1.82, 'I': 2.36,
                 'K': 2.18, 'L': 2.36, 'M': 2.28, 'N': 2.02, 'P': 1.99, 'Q': 2.17, 'R': 2.17, 'S': 2.21,
                 'T': 2.09, 'V': 2.32, 'W': 2.83, 'Y': 2.32}  # 氨基酸的酸性 pKa 值

res_pkb_table = {'A': 9.69, 'C': 10.28, 'D': 9.60, 'E': 9.67, 'F': 9.13, 'G': 9.60, 'H': 9.17,
                 'I': 9.60, 'K': 8.95, 'L': 9.60, 'M': 9.21, 'N': 8.80, 'P': 10.60, 'Q': 9.13,
                 'R': 9.04, 'S': 9.15, 'T': 9.10, 'V': 9.62, 'W': 9.39, 'Y': 9.62}  # 氨基酸的碱性 pKb 值

res_pkx_table = {'A': 0.00, 'C': 8.18, 'D': 3.65, 'E': 4.25, 'F': 0.00, 'G': 0, 'H': 6.00,
                 'I': 0.00, 'K': 10.53, 'L': 0.00, 'M': 0.00, 'N': 0.00, 'P': 0.00, 'Q': 0.00,
                 'R': 12.48, 'S': 0.00, 'T': 0.00, 'V': 0.00, 'W': 0.00, 'Y': 0.00}  # 其他形式的 pKa 或 pKb 值

res_pl_table = {'A': 6.00, 'C': 5.07, 'D': 2.77, 'E': 3.22, 'F': 5.48, 'G': 5.97, 'H': 7.59,
                'I': 6.02, 'K': 9.74, 'L': 5.98, 'M': 5.74, 'N': 5.41, 'P': 6.30, 'Q': 5.65,
                'R': 10.76, 'S': 5.68, 'T': 5.60, 'V': 5.96, 'W': 5.89, 'Y': 5.96}  # 氨基酸的等电点

res_hydrophobic_ph2_table = {'A': 47, 'C': 52, 'D': -18, 'E': 8, 'F': 92, 'G': 0, 'H': -42, 'I': 100,
                             'K': -37, 'L': 100, 'M': 74, 'N': -41, 'P': -46, 'Q': -18, 'R': -26, 'S': -7,
                             'T': 13, 'V': 79, 'W': 84, 'Y': 49}  # 在 pH 2 下，氨基酸的疏水性程度

res_hydrophobic_ph7_table = {'A': 41, 'C': 49, 'D': -55, 'E': -31, 'F': 100, 'G': 0, 'H': 8, 'I': 99,
                             'K': -23, 'L': 97, 'M': 74, 'N': -28, 'P': -46, 'Q': -10, 'R': -14, 'S': -5,
                             'T': 13, 'V': 76, 'W': 97, 'Y': 63}  # 在 pH 7 下，氨基酸的疏水性程度

res_weight_table = dic_normalize(res_weight_table)
res_pka_table = dic_normalize(res_pka_table)
res_pkb_table = dic_normalize(res_pkb_table)
res_pkx_table = dic_normalize(res_pkx_table)
res_pl_table = dic_normalize(res_pl_table)
res_hydrophobic_ph2_table = dic_normalize(res_hydrophobic_ph2_table)
res_hydrophobic_ph7_table = dic_normalize(res_hydrophobic_ph7_table)


def residue_features(residue):
    """
    给每个残基添加理化性质，特征：长度为12的张量
    """
    res_property1 = [1 if residue in pro_res_aliphatic_table else 0,
                     1 if residue in pro_res_aromatic_table else 0,
                     1 if residue in pro_res_polar_neutral_table else 0,
                     1 if residue in pro_res_acidic_charged_table else 0,
                     1 if residue in pro_res_basic_charged_table else 0]
    res_property2 = [res_weight_table[residue], res_pka_table[residue], res_pkb_table[residue], res_pkx_table[residue],
                     res_pl_table[residue], res_hydrophobic_ph2_table[residue], res_hydrophobic_ph7_table[residue]]
    # print(np.array(res_property1 + res_property2).shape)
    return np.array(res_property1 + res_property2)


# one ont encoding
def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        # print(x)
        raise Exception('input {0} not in allowable set{1}:'.format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))  # 返回一个one-hot向量，长度为len(allowable_set)


def one_of_k_encoding_unk(x, allowable_set):
    '''Maps inputs not in the allowable set to the last element.'''
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))


def seq_feature(pro_seq):
    if 'U' in pro_seq or 'B' in pro_seq:
        print('U or B in Sequence')
    pro_seq = pro_seq.replace('U', 'X').replace('B', 'X')  # 将 'U' 和 'B' 替换为 'X'，以标准化序列。
    pro_hot = np.zeros((len(pro_seq), len(pro_res_table)))
    pro_property = np.zeros((len(pro_seq), 12))
    for i in range(len(pro_seq)):
        # if 'X' in pro_seq:
        #     print(pro_seq)
        pro_hot[i,] = one_of_k_encoding(pro_seq[i], pro_res_table)  # 长度为氨基酸种类数量的张量，表示该残基的 one-hot 向量 n*21
        pro_property[i,] = residue_features(pro_seq[i])  # 长12的张量，表示该残基的理化性质 n*12
    return np.concatenate((pro_hot, pro_property), axis=1)


def contact_map(contact_map_proba, contact_threshold=0.5): #contact_map_proba:蛋白质接触概率矩阵 n*n
    num_residues = contact_map_proba.shape[0] # 蛋白质残基数量
    prot_contact_adj = (contact_map_proba >= contact_threshold).long()  # 得到一个邻接矩阵
    edge_index = prot_contact_adj.nonzero(as_tuple=False).t().contiguous()  # 获取所有非零元素的索引 形状：(2, num_edges)
    row, col = edge_index  # 获取行和列的索引
    edge_weight = contact_map_proba[row, col].float()  # 从原始的接触概率矩阵中获取边的权重（概率值）
    ############### CONNECT ISOLATED NODES - Prevent Disconnected Residues ######################  为了防止图中出现孤立的氨基酸残基（没有连接的节点），添加了额外的边
    seq_edge_head1 = torch.stack([torch.arange(num_residues)[:-1], (torch.arange(num_residues) + 1)[:-1]]) #[[0,1,2,3],[1,2,3,4]]
    seq_edge_tail1 = torch.stack([(torch.arange(num_residues))[1:], (torch.arange(num_residues) - 1)[1:]]) #[[1,2,3,4],[0,1,2,3]]
    seq_edge_weight1 = torch.ones(seq_edge_head1.size(1) + seq_edge_tail1.size(1)) * contact_threshold #生成2 * (num_residues - 1) 值为0.5的张量
    edge_index = torch.cat([edge_index, seq_edge_head1, seq_edge_tail1], dim=-1)
    edge_weight = torch.cat([edge_weight, seq_edge_weight1], dim=-1)

    seq_edge_head2 = torch.stack([torch.arange(num_residues)[:-2], (torch.arange(num_residues) + 2)[:-2]])
    seq_edge_tail2 = torch.stack([(torch.arange(num_residues))[2:], (torch.arange(num_residues) - 2)[2:]])
    seq_edge_weight2 = torch.ones(seq_edge_head2.size(1) + seq_edge_tail2.size(1)) * contact_threshold
    edge_index = torch.cat([edge_index, seq_edge_head2, seq_edge_tail2], dim=-1)
    edge_weight = torch.cat([edge_weight, seq_edge_weight2], dim=-1)
    ############### CONNECT ISOLATED NODES - Prevent Disconnected Residues ######################

    edge_index, edge_weight = coalesce(edge_index, edge_weight, reduce='max')
    edge_index, edge_weight = to_undirected(edge_index, edge_weight, reduce='max')
    edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)
    edge_index, edge_weight = add_self_loops(edge_index, edge_weight, fill_value=1)

    return edge_index, edge_weight


def esm_extract(model, batch_converter, seq, layer=36, approach='mean', dim=2560):
    pro_id = 'A'
    if len(seq) <= 700:
        data = []
        data.append((pro_id, seq))  # 链和序列形成一个元组
        print("Input data:", data)
        batch_labels, batch_strs, batch_tokens = batch_converter(data)  # 将数据转换为模型所需的格式
        batch_tokens = batch_tokens.to(next(model.parameters()).device, non_blocking=True)

        with torch.no_grad():
            results = model(batch_tokens, repr_layers=[i for i in range(1, layer + 1)], return_contacts=True)

        LEN = len(seq) // 2
        #logits = results["logits"][0].cpu().numpy()[1: LEN + 1]
        contact_prob_map = results["contacts"][0].cpu().numpy()
        token_representation = torch.cat([results['representations'][i] for i in range(1, layer + 1)])
        assert token_representation.size(0) == layer

        if approach == 'last':  # 蛋白质使用这个
            token_representation = token_representation[-1]
        elif approach == 'sum':
            token_representation = token_representation.sum(dim=0)
        elif approach == 'mean':
            token_representation = token_representation.mean(dim=0)

        token_representation = token_representation.cpu().numpy()
        token_representation = token_representation[1: LEN + 1]
    else:
        amino_acid_count = len(seq) // 2
        contact_prob_map = np.zeros((amino_acid_count, amino_acid_count))  # global contact map prediction  n*n
        token_representation = np.zeros((amino_acid_count, dim))  # n*dim  每个氨基酸的特征
        logits = np.zeros((amino_acid_count, 446))  # n*layer  每个氨基酸的预测
        interval = 150
        i = math.ceil(amino_acid_count / interval)
        # ======================
        # =                    =
        # =                    =
        # =          ======================
        # =          =*********=          =
        # =          =*********=          =
        # ======================          =
        #            =                    =
        #            =                    =
        #            ======================
        # where * is the overlapping area
        # subsection seq contact map prediction
        for s in range(i):
            start = s * interval  # sub seq predict start
            end = min((s + 2) * interval, amino_acid_count)  # sub seq predict end
            sub_seq_len = end - start

            # prediction
            temp_seq = seq[start * 2:end * 2]
            temp_data = []
            temp_data.append((pro_id, temp_seq))
            batch_labels, batch_strs, batch_tokens = batch_converter(temp_data)
            batch_tokens = batch_tokens.to(next(model.parameters()).device, non_blocking=True)
            with torch.no_grad():
                results = model(batch_tokens, repr_layers=[i for i in range(1, layer + 1)], return_contacts=True)

            # insert into the global contact map
            row, col = np.where(contact_prob_map[start:end, start:end] != 0)  # 得到重叠部分的相对索引
            row = row + start  # 转为绝对索引
            col = col + start
            contact_prob_map[start:end, start:end] = contact_prob_map[start:end, start:end] + results["contacts"][
                0].cpu().numpy()
            contact_prob_map[row, col] = contact_prob_map[row, col] / 2.0

            ## TOKEN
            subtoken_repr = torch.cat([results['representations'][i] for i in range(1, layer + 1)])
            assert subtoken_repr.size(0) == layer
            if approach == 'last':
                subtoken_repr = subtoken_repr[-1]
            elif approach == 'sum':
                subtoken_repr = subtoken_repr.sum(dim=0)
            elif approach == 'mean':
                subtoken_repr = subtoken_repr.mean(dim=0)

            subtoken_repr = subtoken_repr.cpu().numpy()
            subtoken_repr = subtoken_repr[:end - start]

            trow = np.where(token_representation[start:end].sum(axis=-1) != 0)[0]
            trow = trow + start
            token_representation[start:end] = token_representation[start:end] + subtoken_repr
            token_representation[trow] = token_representation[trow] / 2.0

            if end == amino_acid_count:
                break

    return torch.from_numpy(token_representation), torch.from_numpy(contact_prob_map)


def esm_extract_A(model, batch_converter, seq, layer=36, approach='mean', dim=2560):
    pro_id = 'A'
    if len(seq) <= 700:
        data = []
        data.append((pro_id, seq))
        batch_labels, batch_strs, batch_tokens = batch_converter(data)
        batch_tokens = batch_tokens.to(next(model.parameters()).device, non_blocking=True)

        with torch.no_grad():
            results = model(batch_tokens, repr_layers=[i for i in range(1, layer + 1)], return_contacts=True)

        contact_prob_map = results["contacts"][0].cpu().numpy()
        token_representation = torch.cat([results['representations'][i] for i in range(1, layer + 1)])
        assert token_representation.size(0) == layer

        if approach == 'last':
            token_representation = token_representation[-1]
        elif approach == 'sum':
            token_representation = token_representation.sum(dim=0)
        elif approach == 'mean':
            token_representation = token_representation.mean(dim=0)

        token_representation = token_representation.cpu().numpy()
        token_representation = token_representation[1: len(seq) + 1]
    else:
        contact_prob_map = np.zeros((len(seq), len(seq)))  # global contact map prediction
        token_representation = np.zeros((len(seq), dim))
        interval = 350
        i = math.ceil(len(seq) / interval)

        for s in range(i):
            start = s * interval  # sub seq predict start
            end = min((s + 2) * interval, len(seq))  # sub seq predict end

            # prediction
            temp_seq = seq[start:end]
            temp_data = []
            temp_data.append((pro_id, temp_seq))
            batch_labels, batch_strs, batch_tokens = batch_converter(temp_data)
            batch_tokens = batch_tokens.to(next(model.parameters()).device, non_blocking=True)
            with torch.no_grad():
                results = model(batch_tokens, repr_layers=[i for i in range(1, layer + 1)], return_contacts=True)

            # insert into the global contact map
            row, col = np.where(contact_prob_map[start:end, start:end] != 0)
            row = row + start
            col = col + start
            contact_prob_map[start:end, start:end] = contact_prob_map[start:end, start:end] + results["contacts"][
                0].cpu().numpy()
            contact_prob_map[row, col] = contact_prob_map[row, col] / 2.0

            ## TOKEN
            subtoken_repr = torch.cat([results['representations'][i] for i in range(1, layer + 1)])
            assert subtoken_repr.size(0) == layer
            if approach == 'last':
                subtoken_repr = subtoken_repr[-1]
            elif approach == 'sum':
                subtoken_repr = subtoken_repr.sum(dim=0)
            elif approach == 'mean':
                subtoken_repr = subtoken_repr.mean(dim=0)

            subtoken_repr = subtoken_repr.cpu().numpy()
            subtoken_repr = subtoken_repr[1: len(temp_seq) + 1]

            trow = np.where(token_representation[start:end].sum(axis=-1) != 0)[0]
            trow = trow + start
            token_representation[start:end] = token_representation[start:end] + subtoken_repr
            token_representation[trow] = token_representation[trow] / 2.0

            if end == len(seq):
                break

    return torch.from_numpy(token_representation), torch.from_numpy(contact_prob_map)



def generate_ESM_structure(model, filename, sequence):
    model.set_chunk_size(256)
    chunk_size = 256
    output = None

    while output is None:
        try:
            with torch.no_grad():
                output = model.infer_pdb(sequence)

            with open(filename, "w") as f:
                f.write(output)
                print("saved", filename)
        except RuntimeError as e:
            if 'out of memory' in str(e):
                print('| WARNING: ran out of memory on chunk_size', chunk_size)
                for p in model.parameters():
                    if p.grad is not None:
                        del p.grad  # free some memory
                torch.cuda.empty_cache()
                chunk_size = chunk_size // 2
                if chunk_size > 2:
                    model.set_chunk_size(chunk_size)
                else:
                    print("Not enough memory for ESMFold")
                    break
            else:
                raise e
    return output is not None


from Bio.PDB import PDBParser

biopython_parser = PDBParser()

one_to_three = {"A": "ALA",
                "C": "CYS",
                "D": "ASP",
                "E": "GLU",
                "F": "PHE",
                "G": "GLY",
                "H": "HIS",
                "I": "ILE",
                "K": "LYS",
                "L": "LEU",
                "M": "MET",
                "N": "ASN",
                "P": "PRO",
                "Q": "GLN",
                "R": "ARG",
                "S": "SER",
                "T": "THR",
                "V": "VAL",
                "W": "TRP",
                "Y": "TYR",
                "B": "ASX",
                "Z": "GLX",
                "X": "UNK",
                "*": " * "}

three_to_one = {}
for _key, _value in one_to_three.items():
    three_to_one[_value] = _key
three_to_one["SEC"] = "C"
three_to_one["MSE"] = "M"


def extract_pdb_seq(protein_path):
    structure = biopython_parser.get_structure('random_id', protein_path)[0]
    seq = ''
    chain_str = ''
    for i, chain in enumerate(structure):
        for res_idx, residue in enumerate(chain):
            if residue.get_resname() == 'HOH':
                continue
            residue_coords = []
            c_alpha, n, c = None, None, None
            for atom in residue:
                if atom.name == 'CA':
                    c_alpha = list(atom.get_vector())
                if atom.name == 'N':
                    n = list(atom.get_vector())
                if atom.name == 'C':
                    c = list(atom.get_vector())
            if c_alpha != None and n != None and c != None:  # only append residue if it is an amino acid and not
                try:
                    seq += three_to_one[residue.get_resname()]
                    chain_str += str(chain.id)
                except Exception as e:
                    seq += 'X'
                    chain_str += str(chain.id)
                    print("encountered unknown AA: ", residue.get_resname(),
                          ' in the complex. Replacing it with a dash X.')

    return seq, chain_str
