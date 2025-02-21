from rdkit import Chem
from rdkit.Chem.rdchem import BondType

from rdkit.Chem import ChemicalFeatures
from rdkit import RDConfig
import os
import numpy as np
# from AAA.tSmiles import get_tsmiles
import torch

fdefName = os.path.join(RDConfig.RDDataDir,'BaseFeatures.fdef')
factory = ChemicalFeatures.BuildFeatureFactory(fdefName)
import sys

# Check if the code is running in a Jupyter notebook
if 'ipykernel' in sys.modules:
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm




def one_of_k_encoding(x, allowable_set): #X不在范围则抛出异常
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))

def one_of_k_encoding_unk(x, allowable_set): #x不在范围则将其认为范围内最后一个元素
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))


def atom_features(atom):
    #encoding = one_of_k_encoding_unk(atom.GetSymbol(),['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na','Ca', 'Fe', 'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb','Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H','Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr','Cr', 'Pt', 'Hg', 'Pb', 'Unknown'])
    encoding = one_of_k_encoding(atom.GetDegree(), [0,1,2,3,4,5,6,7,8,9,10]) + one_of_k_encoding_unk(atom.GetTotalNumHs(), [0,1,2,3,4,5,6,7,8,9,10])
    encoding += one_of_k_encoding_unk(atom.GetImplicitValence(), [0,1,2,3,4,5,6,7,8,9,10]) 
    encoding += one_of_k_encoding_unk(atom.GetHybridization(), [
                      Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
                      Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.SP3D,
                      Chem.rdchem.HybridizationType.SP3D2, 'other'])
    # encoding += one_of_k_encoding_unk(atom.GetFormalCharge(), [0,-1,1,2,-100]) 
    # encoding += one_of_k_encoding_unk(atom.GetNumRadicalElectrons(), [0,1,2,-100]) 
    encoding += [atom.GetIsAromatic()] #是否为芳香性原子加入特征
    # encoding += [atom.IsInRing()]

    try:
        encoding += one_of_k_encoding_unk(
                    atom.GetProp('_CIPCode'),
                    ['R', 'S']) + [atom.HasProp('_ChiralityPossible')]
    except:
        encoding += [0, 0] + [atom.HasProp('_ChiralityPossible')]
    
    return np.array(encoding)



class MoleculeGraphDataset():
    def __init__(self,atom_classes=None, halogen_detail=False, save_path=None):
        ## ATOM CLASSES ##
        self.ATOM_CODES = {} #空字典，用于将原子类型映射到整数代码
        if atom_classes is None:
            metals = ([3, 4, 11, 12, 13] + list(range(19, 32))
                      + list(range(37, 51)) + list(range(55, 84))
                      + list(range(87, 104))) #定义了一组金属元素的原子序号

            self.FEATURE_NAMES = [] #存储原子类别的名称
            if halogen_detail:
                atom_classes = [
                    (5, 'B'),
                    (6, 'C'),
                    (7, 'N'),
                    (8, 'O'),
                    (15, 'P'),
                    (16, 'S'),
                    (34, 'Se'),
                    ## halogen
                    (9, 'F'),
                    (17, 'Cl'),
                    (35, 'Br'),
                    (53, 'I'),
                    ## halogen
                    (metals, 'metal')
                ]
            else:
                atom_classes = [
                    (5, 'B'),
                    (6, 'C'),
                    (7, 'N'),
                    (8, 'O'),
                    (15, 'P'),
                    (16, 'S'),
                    (34, 'Se'),
                    ## halogen
                    ([9, 17, 35, 53], 'halogen'),
                    ## halogen
                    (metals, 'metal')
                ]
            

        self.NUM_ATOM_CLASSES = len(atom_classes) #原子类型的数量
        for code, (atom, name) in enumerate(atom_classes):
            if type(atom) is list:
                for a in atom:
                    self.ATOM_CODES[a] = code #金属原子均为相同的code
            else:
                self.ATOM_CODES[atom] = code
            self.FEATURE_NAMES.append(name)

        ## Extra atom feature to extract
        self.feat_types = ['Donor', 'Acceptor', 'Hydrophobe', 'LumpedHydrophobe'] #定义了额外的原子特征类型，描述理化性质

        ## Bond feature
        self.edge_dict = {BondType.SINGLE: 1, BondType.DOUBLE: 2,
                     BondType.TRIPLE: 3, BondType.AROMATIC: 4,
                         BondType.UNSPECIFIED: 1} #单键、双键、三键、芳香键和未指定键
        ## File Paths
        self.save_path = save_path

    def hybridization_onehot(self,hybrid_type):
        hybrid_type = str(hybrid_type)
        types = {'S': 0, 'SP': 1, 'SP2': 2, 'SP3': 3, 'SP3D': 4, 'SP3D2': 5}

        encoding = np.zeros(len(types))
        try:
            encoding[types[hybrid_type]] = 1.0
        except:
            pass
        return encoding

    def encode_num(self,atomic_num):
        """Encode atom type with a binary vector. If atom type is not included in
        the `atom_classes`, its encoding is an all-zeros vector.

        Parameters
        ----------
        atomic_num: int
            Atomic number

        Returns
        -------
        encoding: np.ndarray
            Binary vector encoding atom type (one-hot or null).
        """

        if not isinstance(atomic_num, int):
            raise TypeError('Atomic number must be int, %s was given'
                            % type(atomic_num))

        encoding = np.zeros(self.NUM_ATOM_CLASSES)
        try:
            encoding[self.ATOM_CODES[atomic_num]] = 1.0
        except:
            pass
        return encoding

    def atom_feature_extract(self,atom):
        '''
            Atom Feature Extraction:
                0 - Degree
                1 - Total Valency
                2 to 7 - Hybridization Type One-hot Encoding
                8 - Number of Radical Electrons
                9 - Number of Formal Charge
                10 - Aromatic
                11 - Belongs to a Ring
                12 - Final to X belongs to Atom Classes
        '''

        feat = []

        feat.append(atom.GetDegree())
        feat.append(atom.GetTotalValence())
        feat += self.hybridization_onehot(atom.GetHybridization()).tolist()
        feat.append(atom.GetNumRadicalElectrons())
        feat.append(atom.GetFormalCharge())
        feat.append(int(atom.GetIsAromatic()))
        feat.append(int(atom.IsInRing()))
        # Atom class
        #feat += self.encode_num(atom.GetAtomicNum()).tolist()

        return feat

    def mol_feature(self,mol):
        atom_ids = []
        atom_feats = []
        for atom in mol.GetAtoms():
            atom_ids.append(atom.GetIdx())
            feat = self.atom_feature_extract(atom)
            atom_feats.append(feat)

        feature = np.array(list(zip(*sorted(zip(atom_ids, atom_feats))))[-1])

        return feature

    def mol_extra_feature(self, mol):
        atom_num = len(mol.GetAtoms())
        feature = np.zeros((atom_num, len(self.feat_types)))

        fact_feats = factory.GetFeaturesForMol(mol)
        for f in fact_feats:
            f_type = f.GetFamily()
            if f_type in self.feat_types:
                f_index = self.feat_types.index(f_type)
                atom_ids = f.GetAtomIds()
                feature[atom_ids, f_index] = 1

        return feature

    def mol_simplified_feature(self,mol):
        atom_ids = []
        atom_feats = []
        for atom in mol.GetAtoms():
            atom_ids.append(atom.GetIdx())
            atomic_num = atom.GetAtomicNum() #获取原子序数，原子的唯一标识

            if atomic_num in self.ATOM_CODES.keys():
                atom_feats.append([self.ATOM_CODES[atomic_num] + 1])
            else:
                atom_feats.append([0])
        feature = np.array(list(zip(*sorted(zip(atom_ids, atom_feats))))[-1])

        return feature
    
    def mol_sequence_simplified_feature(self,mol):
        
        atom_ids = []
        atom_feats = []
        for atom in mol.GetAtoms():
        
            atom_ids.append(atom.GetIdx())
            onehot_label = one_of_k_encoding_unk(atom.GetSymbol(),
                                  ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'As', 'Al',
                                   'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H',
                                   'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr', 'Pt', 'Hg', 'Pb',
                                   'Unknown'])
            out = np.array(onehot_label).nonzero()[0]
            atom_feats.append(out)
        
        feature = np.array(list(zip(*sorted(zip(atom_ids, atom_feats))))[-1])
        
        return feature




    def mol_full_feature(self, mol):
        atom_ids = [] #存储原子的索引
        atom_feats = []
        for atom in mol.GetAtoms():
            atom_ids.append(atom.GetIdx())
            feature = atom_features(atom) #注意，此处只添加了理化特征，没有原子类型的特征
            atom_feats.append(feature)
        #将原子的索引和特征配对成一个个元组 (id, feature)。根据原子索引对这些元组进行排序。将排序后的元组解压成两个列表，一个是索引列表，另一个是特征列表。将特征列表转换为 NumPy 数组。
        feature = np.array(list(zip(*sorted(zip(atom_ids, atom_feats))))[-1])

        return feature

    def bond_feature(self,mol):
        atom_num = len(mol.GetAtoms())
        adj = np.zeros((atom_num,atom_num))

        for b in mol.GetBonds():
            v1 = b.GetBeginAtomIdx()
            v2 = b.GetEndAtomIdx()
            b_type = self.edge_dict[b.GetBondType()]
            adj[v1 - 1, v2 - 1] = b_type
            adj[v2 - 1, v1 - 1] = b_type

        return adj

    def junction_tree(self,mol):
        tree_edge_index, atom2clique_index, num_cliques, x_clique = tree_decomposition(mol,return_vocab=True)
        ## if weird compounds => each assign the separate cluster
        if atom2clique_index.nelement() == 0:
            num_cliques = len(mol.GetAtoms())
            x_clique = torch.tensor([3]*num_cliques)
            atom2clique_index = torch.stack([torch.arange(num_cliques),
                                             torch.arange(num_cliques)])
        tree = dict(tree_edge_index=tree_edge_index,
             atom2clique_index=atom2clique_index,
             num_cliques=num_cliques,
             x_clique=x_clique)
        
        return tree


    def featurize(self,mol,type='atom_type'):
        if type=='atom_type': #后调用的这个
            atom_feature = self.mol_simplified_feature(mol) #返回分子特征，形状为(原子数量*1)
        elif type =='detailed_atom_type':
            atom_feature = self.mol_sequence_simplified_feature(mol)
        elif type=='atom_feature':
            base_feat = self.mol_feature(mol)
            extra_feat = self.mol_extra_feature(mol)
            atom_feature = np.concatenate((base_feat, extra_feat), axis=1)
           
        elif type=='atom_full_feature': #先调用这个！！！
            atom_feature = self.mol_full_feature(mol)
            #extra_feat = self.mol_extra_feature(mol)
            #atom_feature = np.concatenate((base_feat, extra_feat), axis=1)
        else:
            raise Exception('atom_type or atom_feature')
        bond_feature = self.bond_feature(mol) #邻接矩阵，矩阵的取值为1，2，3，4分别代表单键、双键、三键、芳香键

        return atom_feature, bond_feature




## FILE from pytorch geometric version 2.
from itertools import chain
from typing import Any, Tuple, Union

import torch
from scipy.sparse.csgraph import minimum_spanning_tree
from torch import Tensor

from torch_geometric.utils import (
    from_scipy_sparse_matrix,
    to_scipy_sparse_matrix,
    to_undirected,
)


def tree_decomposition(
    mol: Any,
    return_vocab: bool = False,
) -> Union[Tuple[Tensor, Tensor, int], Tuple[Tensor, Tensor, int, Tensor]]:
    r"""The tree decomposition algorithm of molecules from the
    `"Junction Tree Variational Autoencoder for Molecular Graph Generation"
    <https://arxiv.org/abs/1802.04364>`_ paper.
    Returns the graph connectivity of the junction tree, the assignment
    mapping of each atom to the clique in the junction tree, and the number
    of cliques.

    Args:
        mol (rdkit.Chem.Mol): An :obj:`rdkit` molecule.
        return_vocab (bool, optional): If set to :obj:`True`, will return an
            identifier for each clique (ring, bond, bridged compounds, single).
            (default: :obj:`False`)

    :rtype: :obj:`(LongTensor, LongTensor, int)` if :obj:`return_vocab` is
        :obj:`False`, else :obj:`(LongTensor, LongTensor, int, LongTensor)`
    """
    import rdkit.Chem as Chem

    # Cliques = rings and bonds.
    cliques = [list(x) for x in Chem.GetSymmSSSR(mol)] #获取分子所有环，每个环视为一个团簇。cliques是一个包含列表的列表。每个列表是一个团簇。
    xs = [0] * len(cliques) #xs 列表用于存储团簇的类别，其中 0 表示环，1 表示键。
    for bond in mol.GetBonds():
        if not bond.IsInRing(): #不在环中
            cliques.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()]) #像成一个新团簇
            xs.append(1)

    # Generate `atom2clique` mappings.
    atom2clique = [[] for i in range(mol.GetNumAtoms())]
    for c in range(len(cliques)):
        for atom in cliques[c]:
            atom2clique[atom].append(c) #把该原子所属的团簇索引添加到该原子对应的列表中

    # Merge rings that share more than 2 atoms as they form bridged compounds.如果两个环共享超过两个原子，它们会被合并为一个更大的团簇
    for c1 in range(len(cliques)):
        for atom in cliques[c1]:
            for c2 in atom2clique[atom]:
                if c1 >= c2 or len(cliques[c1]) <= 2 or len(cliques[c2]) <= 2:
                    continue
                if len(set(cliques[c1]) & set(cliques[c2])) > 2:
                    cliques[c1] = set(cliques[c1]) | set(cliques[c2])
                    xs[c1] = 2 #合并后的团簇标签为2
                    cliques[c2] = []
                    xs[c2] = -1
    cliques = [c for c in cliques if len(c) > 0]
    xs = [x for x in xs if x >= 0]

    # Update `atom2clique` mappings. 更新每个原子对应的列表
    atom2clique = [[] for i in range(mol.GetNumAtoms())]
    for c in range(len(cliques)):
        for atom in cliques[c]:
            atom2clique[atom].append(c)

    # Add singleton cliques in case there are more than 2 intersecting 添加单原子团簇（一个原子同时位于多个环或键上）
    # cliques. We further compute the "initial" clique graph.
    edges = {}
    for atom in range(mol.GetNumAtoms()):
        cs = atom2clique[atom] #记录原子所属团簇的列表
        if len(cs) <= 1:
            continue

        # Number of bond clusters that the atom lies in.
        bonds = [c for c in cs if len(cliques[c]) == 2]
        # Number of ring clusters that the atom lies in.
        rings = [c for c in cs if len(cliques[c]) > 4]

        if len(bonds) > 2 or (len(bonds) == 2 and len(cs) > 2):
            cliques.append([atom])
            xs.append(3)
            c2 = len(cliques) - 1
            for c1 in cs:
                edges[(c1, c2)] = 1

        elif len(rings) > 2:
            cliques.append([atom])
            xs.append(3)
            c2 = len(cliques) - 1
            for c1 in cs:
                edges[(c1, c2)] = 99

        else:
            for i in range(len(cs)):
                for j in range(i + 1, len(cs)):
                    c1, c2 = cs[i], cs[j]
                    count = len(set(cliques[c1]) & set(cliques[c2]))
                    edges[(c1, c2)] = min(count, edges.get((c1, c2), 99))

    # Update `atom2clique` mappings.
    atom2clique = [[] for i in range(mol.GetNumAtoms())]
    for c in range(len(cliques)):
        for atom in cliques[c]:
            atom2clique[atom].append(c)

    if len(edges) > 0:
        edge_index_T, weight = zip(*edges.items())
        edge_index = torch.tensor(edge_index_T).t()
        inv_weight = 100 - torch.tensor(weight)
        graph = to_scipy_sparse_matrix(edge_index, inv_weight, len(cliques))
        junc_tree = minimum_spanning_tree(graph)
        edge_index, _ = from_scipy_sparse_matrix(junc_tree)
        edge_index = to_undirected(edge_index, num_nodes=len(cliques))
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)

    rows = [[i] * len(atom2clique[i]) for i in range(mol.GetNumAtoms())]
    row = torch.tensor(list(chain.from_iterable(rows)))
    col = torch.tensor(list(chain.from_iterable(atom2clique)))
    atom2clique = torch.stack([row, col], dim=0).to(torch.long)

    if return_vocab:
        vocab = torch.tensor(xs, dtype=torch.long)
        return edge_index, atom2clique, len(cliques), vocab #edge_index：团簇之间的连接情况(形状2*num_edges)，atom2clique：每个原子所属团簇索引列表，len(cliques)：团簇数量，vocab：团簇类别
    else:
        return edge_index, atom2clique, len(cliques)
    

###

def smiles2graph(m_str):
    mgd = MoleculeGraphDataset(halogen_detail=False)
    mol = Chem.MolFromSmiles(m_str)
    #mol = get_mol(m_str)
    atom_feature, bond_feature = mgd.featurize(mol,'atom_full_feature')
    atom_idx, _ = mgd.featurize(mol,'atom_type')
    tree = mgd.junction_tree(mol)

    out_dict = {
        'smiles':m_str, #SMILES串
        'atom_feature':torch.tensor(atom_feature),#.to(torch.int8), #分子特征
        'atom_types':'|'.join([i.GetSymbol() for i in mol.GetAtoms()]), #用|分割各个原子字符
        'atom_idx':torch.tensor(atom_idx),#.to(torch.int8), #原子索引
        'bond_feature':torch.tensor(bond_feature),#.to(torch.int8), #有权值的邻接矩阵

    }
    tree['tree_edge_index'] = tree['tree_edge_index']#.to(torch.int8) #团簇之间的连接情况(形状2*num_edges)
    tree['atom2clique_index'] = tree['atom2clique_index']#.to(torch.int8) #每个原子所属团簇索引列表
    tree['x_clique'] = tree['x_clique']#.to(torch.int8) #团簇类型

    out_dict.update(tree)
    
    return out_dict 
####

def ligand_init(smiles_list):
    ligand_dict = {}
    for smiles in tqdm(smiles_list):
        ligand_dict[smiles] = smiles2graph(smiles)
        # tsmiles = get_tsmiles(smiles)
        # ligand_dict[smiles]['tsml'] = tsmiles

    return ligand_dict