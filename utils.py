import json
import numpy as np
from tqdm import tqdm
from copy import deepcopy
import random
import torch
from torch.utils.data import Dataset

def dict2list(d):
    l = []
    for v in d.values():
        l += v
    return l

def preprocess_data(data, entity2id=None):
    input_text, target_text, lm_tokenizer, args = data
    assert type(input_text) == type(target_text) == list

    concept_positions = []
    concept_mask = []

    if args.fresh_tokenizer:
        pad_token_id = args.pad_token_id
    else:
        pad_token_id = lm_tokenizer.pad_token_id
    assert type(pad_token_id) == int

    # input_ids = lm_tokenizer([input_text], return_tensors="pt")["input_ids"][0]
    input_ids = []
    for jj, patch in enumerate(input_text):
        if args.fresh_tokenizer:
            patch_ids = deepcopy(patch)
        else:
            patch_ids = lm_tokenizer([patch])["input_ids"][0]
        concept_pos = [0 for _ in range(len(patch_ids))]
        concept_pos[-1] = 1
        padding_temp = args.max_patch_length - len(concept_pos)
        assert padding_temp >= 0
        concept_pos.extend([0 for _ in range(padding_temp)])
        if args.wernicke_broca:
            patch_ids.extend([pad_token_id for _ in range(padding_temp)])
        input_ids.extend(patch_ids)
        concept_positions.extend(concept_pos)
        concept_mask.append(0)

    # output_ids = lm_tokenizer([target_text], return_tensors="pt")["input_ids"][0]
    output_ids = []
    for jj, patch in enumerate(target_text):
        if args.fresh_tokenizer:
            patch_ids = deepcopy(patch)
        else:
            patch_ids = lm_tokenizer([patch])["input_ids"][0]
        concept_pos = [0 for _ in range(len(patch_ids))]
        concept_pos[-1] = 1
        padding_temp = args.max_patch_length - len(concept_pos)
        assert padding_temp >= 0
        concept_pos.extend([0 for _ in range(padding_temp)])
        if args.wernicke_broca:
            patch_ids.extend([pad_token_id for _ in range(padding_temp)])
        output_ids.extend(patch_ids)
        concept_positions.extend(concept_pos)
        concept_mask.append(1)
    
    assert concept_positions

    all_ids = input_ids + output_ids

    # padding in normal mode
    if not args.wernicke_broca:
        padding_length = args.max_length - len(all_ids)
        assert padding_length >= 0
        all_ids.extend([pad_token_id for _ in range(padding_length)])
        concept_positions.extend([0 for _ in range(padding_length)])   # not really necessary

    all_ids, concept_positions, concept_mask = torch.tensor(all_ids), torch.tensor(concept_positions), torch.tensor(concept_mask)
    
    if not args.wernicke_broca:
        lm_labels = all_ids.clone()
        lm_labels[lm_labels == pad_token_id] = -100
        lm_labels[:len(input_ids)] = -100  # no loss on input tokens
        return {
            "input_ids": all_ids,
            "labels": lm_labels,
        }

    lm_labels = []
    aux_labels = []   # for in-batch negatives for the concept level loss
    aux_labels_2 = []  # for full negative
    for jj in range(len(input_text)):
        if jj < len(input_text) - 1:
            lm_labels.extend([-100 for _ in range(args.max_patch_length)])
            aux_labels.append(-100)
            aux_labels_2.append(-100)
        if jj == len(input_text) - 1:
            patch = target_text[0]
            if args.fresh_tokenizer:
                patch_ids = deepcopy(patch)
            else:
                patch_ids = lm_tokenizer([patch])["input_ids"][0]
            lm_labels.extend(patch_ids)
            padding_temp = args.max_patch_length - len(patch_ids)
            assert padding_temp >= 0
            lm_labels.extend([-100 for _ in range(padding_temp)])
            aux_labels.append(len(aux_labels)+1)
            if entity2id is None:
                aux_labels_2.append(patch_ids[0])
            else:
                assert len(patch_ids) == 2
                aux_labels_2.append(entity2id[(patch_ids[0], patch_ids[1])])
    for jj in range(len(target_text)):
        if jj < len(target_text) - 1:
            patch = target_text[jj+1]
            if args.fresh_tokenizer:
                patch_ids = deepcopy(patch)
            else:
                patch_ids = lm_tokenizer([patch])["input_ids"][0]
            lm_labels.extend(patch_ids)
            padding_temp = args.max_patch_length - len(patch_ids)
            assert padding_temp >= 0
            lm_labels.extend([-100 for _ in range(padding_temp)])
            aux_labels.append(len(aux_labels)+1)
            if entity2id is None:
                aux_labels_2.append(patch_ids[0])
            else:
                assert len(patch_ids) == 2
                aux_labels_2.append(entity2id[(patch_ids[0], patch_ids[1])])

        if jj == len(target_text) - 1:
            lm_labels.extend([-100 for _ in range(args.max_patch_length)])
            aux_labels.append(-100)
            aux_labels_2.append(-100)
    lm_labels = torch.tensor(lm_labels)
    aux_labels = torch.tensor(aux_labels)
    aux_labels_2 = torch.tensor(aux_labels_2)
    
    wb_labels = []
    for jj in range(len(input_text)):
        patch = input_text[jj]
        if args.fresh_tokenizer:
            patch_ids = deepcopy(patch)
        else:
            patch_ids = lm_tokenizer([patch])["input_ids"][0]
        wb_labels.extend(patch_ids)
        padding_temp = args.max_patch_length - len(patch_ids)
        assert padding_temp >= 0
        wb_labels.extend([-100 for _ in range(padding_temp)])
    for jj in range(len(target_text)):
        patch = target_text[jj]
        if args.fresh_tokenizer:
            patch_ids = deepcopy(patch)
        else:
            patch_ids = lm_tokenizer([patch])["input_ids"][0]
        wb_labels.extend(patch_ids)
        padding_temp = args.max_patch_length - len(patch_ids)
        assert padding_temp >= 0
        wb_labels.extend([-100 for _ in range(padding_temp)])
    wb_labels = torch.tensor(wb_labels)

    return {
        "input_ids": all_ids,
        "labels": lm_labels,
        "wb_labels": wb_labels,
        "aux_labels": aux_labels,
        "aux_labels_2": aux_labels_2,
        "concept_positions": concept_positions,
        "concept_mask": concept_mask,
    }

class SimpleDataset(Dataset):
    def __init__(self, lm_tokenizer, args, data, entity2id=None):
        data_ = [
            (item['input_text'], item['target_text'], lm_tokenizer, args) for item in data
        ]
            
        self.examples = [
            preprocess_data(d, entity2id) for d in tqdm(data_, disable=args.silent)
        ]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return self.examples[index]


def read_json(file_name):
    """
    file_name: a .json file containing a list of items. The items should share the same set of keys.
    """
    with open(file_name, 'r', encoding='utf-8') as f:
        data = json.load(f)
        return data

def flatten(l):
    return_l = []
    for item in l:
        for i in item:
            return_l.append(i)
    return return_l

def build_dicts(entities):
    entity2ind = dict()
    ind2entity = []
    for i in range(len(entities)):
        entity = entities[i]
        if not (entity in ind2entity):
            ind2entity.append(entity)
            entity2ind[entity] = len(ind2entity) - 1
    return ind2entity, entity2ind

def choose(arr, ratio_or_count):
    """
    Note: when the amount to choose is greater, instead of throwing an error, this function returns the entire list.
    """
    if type(ratio_or_count) == float:
        num = round(ratio_or_count*len(arr))
    elif type(ratio_or_count) == int:
        num = ratio_or_count
    else:
        assert False
    if num >= len(arr):
        return arr
    rand_inds = np.random.choice(len(arr), num, replace=False).tolist()
    return [arr[i] for i in rand_inds]

def random_pairing(people_):
    # given a list of entities, generate a list of random pairs among the entities
    assert len(people_) % 2 == 0
    people = deepcopy(people_)
    random.shuffle(people)
    return [(people[2*i], people[2*i+1]) for i in range(len(people)//2)]
    
def split(arr_, ratio_or_count):
    arr = deepcopy(arr_)
    if type(ratio_or_count) == float:
        num = round(ratio_or_count*len(arr))
    elif type(ratio_or_count) == int:
        num = ratio_or_count
    else:
         assert False
    assert num <= len(arr)
    random.shuffle(arr)
    return arr[:num], arr[num:]

def kagebunshin(l, rep):
    if rep == 1:
        return l
    return_l = []
    for _ in range(rep):
        return_l = return_l + l
    return return_l

def merge_dicts(dict_1, dict_2):
    keys_1 = set(dict_1.keys())
    keys_2 = set(dict_2.keys())
    # sanity check
    for key in keys_1 & keys_2:
        assert dict_1[key] == dict_2[key]
    # merge
    new_dict = dict()
    for key in keys_1:
        new_dict[key] = dict_1[key]
    for key in keys_2 - keys_1:
        new_dict[key] = dict_2[key]
    return new_dict


class Bag:
    def __init__(self, l):
        self.bag = dict()
        for key in l:
            self.add(key)
    def add(self, key):
        if key in self.bag:
            self.bag[key] += 1
        else:
            self.bag[key] = 1
    def pop(self, key):
        if key not in self.bag:
            assert False
        if self.bag[key] == 1:
            del self.bag[key]
        else:
            self.bag[key] -= 1
    def keys(self):
        return set(self.bag.keys())

def rand_matching(A_, B_):
    """
    given two list of items, return a list of random pairs (a,b) \in A x B that are different
    """
    return_l = []
    A, B = deepcopy(A_), deepcopy(B_)
    assert len(A) == len(B)

    bag = Bag(B)
    matched = {a: set() for a in set(A)}   # list of items b in B s.t. (a,b) already exists
    for a in tqdm(A):
        options = list(bag.keys() - matched[a])
        if not options:
            print("failed; re-try")
            # TODO: maybe make this more efficient
            return rand_matching(A, B)
        b = np.random.choice(options)
        assert b not in matched[a]
        matched[a].add(b)
        bag.pop(b)
        return_l.append((a, b))
        # print((a,b))
    return return_l

def cosine_similarity(vector1, vector2):
    dot_product = torch.dot(vector1, vector2)
    norm_vector1 = torch.norm(vector1)
    norm_vector2 = torch.norm(vector2)
    cos_similarity_manual = dot_product / (norm_vector1 * norm_vector2)
    return cos_similarity_manual.item()

def tensor_diff(a,b):
    return torch.max(torch.abs(a-b))