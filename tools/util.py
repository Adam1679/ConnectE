
import numpy as np
import scipy.sparse as sp
from collections import OrderedDict
import random
import json
def grad_sum_matrix(idx):
    """
    This function helps us to vectorize the training process.
    So that for each batch-size training samples, we could 
    calculate in parallel.
    """
    # unique_idx: unique entity_id; idx_inverse: index for each entity in idx in the unique_idx
    unique_idx, idx_inverse = np.unique(idx, return_inverse=True)
    # Calculate the number of entities that are nedded for updating.(including duplicates)
    sz = len(idx_inverse)
    # generate a coefficient matrix. M.shape = (num_of_unique_entities, num_of_samples)
    # M = [[1,1,0,1,0],
    #      [0,0,1,0,1],
    #      [1,1,1,1,1]]
    # This means the 1-st sample is used to update the 0-th and 2-ed entity; 
    M = sp.coo_matrix((np.ones(sz), (idx_inverse, np.arange(sz)))).tocsr()  # M.shape = (num_of_unique_idx, tot_sample)
    # normalize summation matrix so that each row sums to one
    tot_update_time = np.array(M.sum(axis=1))  # shape = (num_of_unique_entities, ) 
    return unique_idx, M, tot_update_time



def normalize(M, idx=None):
    """
    Used as a tool function to normalize the matrix M by using column-wise L-2 norm.
    If idx is not None, then only the row specified in idx would be normalized.
    """
    if idx is None:
        M /= np.sqrt(np.sum(M ** 2, axis=1))[:, np.newaxis]
    else:
        nrm = np.sqrt(np.sum(M[idx, :] ** 2, axis=1))[:, np.newaxis]
        M[idx, :] /= nrm
    return M

def loadVectors(path="output/entityVector.txt"):
    """
    Used to load the vector specified in the path.
    """
    vectorDict = {}
    with open(path, 'r') as f:
        for line in f.readlines():
            k, v = line.strip().split("\t")
            v = np.array(eval(v))
            vectorDict[k] = v
    return vectorDict

def encode2id(triplet, *ids):
    """
    Traslate the triplet from string format to unique integer format. 
    (Unkonwn entities/relations/types are ignored.)
    """
    data = []
    for pairs in triplet:
        new_pairs = []
        try:
            for i in range(len(pairs)):
                new_pairs.append(ids[i][pairs[i]])
            data.append(tuple(new_pairs))
        except KeyError:
            pass

    return data

def init_nunif(sz):
    """
    Normalized uniform initialization

    See Glorot X., Bengio Y.: "Understanding the difficulty of training
    deep feedforward neural networks". AISTATS, 2010
    """
    bnd = np.sqrt(6) / np.sqrt(sz[0] + sz[1])
    p = np.random.uniform(low=-bnd, high=bnd, size=sz)
    return np.squeeze(p)

def loadEntityId(path="data/FB15K/entity2id.txt"):
    """
    Used to load the entity unique integer ID specified in the path.
    """
    d = OrderedDict()
    with open(path, 'r') as f:
        for line in f.readlines():
            e, id = line.strip().split("\t")
            d[e] = int(id)
    return d

def loadRelationId(path="data/FB15K/relation2id.txt"):
    """
    Used to load the relation unique integer ID specified in the path.
    """
    d = OrderedDict()
    with open(path, 'r') as f:
        for line in f.readlines():
            r, id = line.strip().split("\t")
            d[r] = int(id)
    return d

def loadTypeId(path="data/FB15K/type2id.txt"):
    """
    Used to load the type unique integer ID specified in the path.
    """
    d = OrderedDict()
    with open(path, 'r') as f:
        for line in f.readlines():
            r, id = line.strip().split("\t")
            d[r] = int(id)
    return d

def loadTriplet(path="data/freebase_mtr100_mte100-train.txt"):
    """
    Used to load the triplets specified in the path.
    """
    triplet = []
    with open(path, 'r') as f:
        for line in f.readlines():
            h, l, t = line.strip().split("\t")
            triplet.append((h,l,t))
    return triplet

def loadEntity2Type(path="data/FB15k_Entity_Type_train.txt"):
    """
    Used to load the entity-type mapping data specified in the path.
    """
    triplet = []
    with open(path, 'r') as f:
        for line in f.readlines():
            h, t = line.strip().split("\t")
            triplet.append((h,t))
    return triplet

def getRandomObj(entity_, sourse, exclude_list=None):
    """
    Used to generate a ramdom negative sample(filtered).
    """
    random_entity_ = entity_
    if not exclude_list:
        exclude_list = set([entity_,])

    while (random_entity_ in exclude_list):
        random_entity_ = random.sample(sourse, 1)[0]
    return random_entity_

def get_config(path):
    """
    Get and parse the configeration file in `path`
    """
    with open(path, "r") as f:
        config = json.load(f)
    return config