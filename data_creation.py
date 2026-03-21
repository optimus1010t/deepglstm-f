import argparse
import pandas as pd
import numpy as np
import os
from collections import OrderedDict
from rdkit import Chem
from rdkit.Chem import MolFromSmiles
import networkx as nx
from utils import *
from transformers import AutoTokenizer
from tqdm import tqdm

def atom_features(atom):
    return np.array(one_of_k_encoding_unk(atom.GetSymbol(),['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na','Ca', 'Fe', 'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb','Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H','Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr','Cr', 'Pt', 'Hg', 'Pb', 'Unknown']) +
                    one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6,7,8,9,10]) +
                    one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5, 6,7,8,9,10]) +
                    one_of_k_encoding_unk(atom.GetValence(Chem.ValenceType.IMPLICIT), [0, 1, 2, 3, 4, 5, 6,7,8,9,10]) +
                    [atom.GetIsAromatic()])

def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))

def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))

def smile_to_graph(smile):
    mol = Chem.MolFromSmiles(smile)
    
    c_size = mol.GetNumAtoms()
    
    features = []
    for atom in mol.GetAtoms():
        feature = atom_features(atom)
        features.append( feature / sum(feature) )

    edges = []
    for bond in mol.GetBonds():
        edges.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
    g = nx.Graph(edges).to_directed()
    edge_index = []
    for e1, e2 in g.edges:
        edge_index.append([e1, e2])
        
    return c_size, features, edge_index

def seq_cat(prot):
    x = np.zeros(max_seq_len)
    for i, ch in enumerate(prot[:max_seq_len]): 
        x[i] = seq_dict[ch]
    return x  


seq_voc = "ABCDEFGHIKLMNOPQRSTUVWXYZ"
seq_dict = {v:(i+1) for i,v in enumerate(seq_voc)}
seq_dict_len = len(seq_dict)
max_seq_len = 1000

def main(args):
  dataset = args.dataset

  compound_iso_smiles = []
  opts = ['train','test']
  for opt in opts:
    df = pd.read_csv('data/' + dataset + '_' + opt + '.csv')
    compound_iso_smiles += list( df['compound_iso_smiles'] )
  compound_iso_smiles = set(compound_iso_smiles)
  smile_graph = {}
  for smile in tqdm(compound_iso_smiles, desc="Parsing SMILES to graph"):
    g = smile_to_graph(smile)
    smile_graph[smile] = g

  # convert to torch geometric data
  processed_data_file_train = 'data/processed/' + dataset + '_train.pt'
  processed_data_file_test = 'data/processed/' + dataset + '_test.pt'
  if ((not os.path.isfile(processed_data_file_train)) or (not os.path.isfile(processed_data_file_test))):
    if args.use_esm:
      print("Loading ESM Tokenizer...")
      tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")
    
    df = pd.read_csv('data/' + dataset + '_train.csv')
    train_drugs, train_prots,  train_Y = list(df['compound_iso_smiles']),list(df['target_sequence']),list(df['affinity'])
    train_esm_ids, train_esm_mask = None, None
    if args.use_esm:
      print("Tokenizing train proteins for ESM...")
      train_esm = tokenizer(train_prots, truncation=True, max_length=1024, padding='max_length', return_tensors='pt')
      train_esm_ids = [train_esm['input_ids'][i] for i in tqdm(range(len(train_prots)), desc='Extracting train ESM IDs')]
      train_esm_mask = [train_esm['attention_mask'][i] for i in tqdm(range(len(train_prots)), desc='Extracting train ESM Masks')]
    XT = [seq_cat(t) for t in tqdm(train_prots, desc="Categorizing train sequences")]
    train_drugs, train_prots,  train_Y = np.asarray(train_drugs), np.asarray(XT), np.asarray(train_Y)
    
    df = pd.read_csv('data/' + dataset + '_test.csv')
    test_drugs, test_prots,  test_Y = list(df['compound_iso_smiles']),list(df['target_sequence']),list(df['affinity'])
    test_esm_ids, test_esm_mask = None, None
    if args.use_esm:
      print("Tokenizing test proteins for ESM...")
      test_esm = tokenizer(test_prots, truncation=True, max_length=1024, padding='max_length', return_tensors='pt')
      test_esm_ids = [test_esm['input_ids'][i] for i in tqdm(range(len(test_prots)), desc='Extracting test ESM IDs')]
      test_esm_mask = [test_esm['attention_mask'][i] for i in tqdm(range(len(test_prots)), desc='Extracting test ESM Masks')]
    XT = [seq_cat(t) for t in tqdm(test_prots, desc="Categorizing test sequences")]
    test_drugs, test_prots,  test_Y = np.asarray(test_drugs), np.asarray(XT), np.asarray(test_Y)

    # make data PyTorch Geometric ready
    print('preparing ', dataset + '_train.pt in pytorch format!')
    train_data = TestbedDataset(root='data', dataset=dataset+'_train', xd=train_drugs, xt=train_prots, y=train_Y,smile_graph=smile_graph, esm_ids=train_esm_ids, esm_mask=train_esm_mask)
    print('preparing ', dataset + '_test.pt in pytorch format!')
    test_data = TestbedDataset(root='data', dataset=dataset+'_test', xd=test_drugs, xt=test_prots, y=test_Y,smile_graph=smile_graph, esm_ids=test_esm_ids, esm_mask=test_esm_mask)
    print(processed_data_file_train, ' and ', processed_data_file_test, ' have been created')
    print(processed_data_file_test, ' have been created')        
  else:
    print(processed_data_file_train, ' and ', processed_data_file_test, ' are already created')
    print(processed_data_file_test, ' are already created')


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description="Creation of dataset")
  parser.add_argument("--dataset",type=str,default='davis',help="Dataset Name (davis,kiba,DTC,Metz,ToxCast,Stitch)")
  parser.add_argument("--use_esm",action="store_true",help="Whether to extract ESM features")
  args = parser.parse_args()
  print(args)
  main(args)

