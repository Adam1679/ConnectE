import os
from collections import Counter
from collections import defaultdict
from copy import deepcopy
import pandas as pd
import json

#**********************************clean the valid/test triplet data**************************************


train_path = "./YAGO43k/YAGO43k_name_train.txt"
valid_path = "./YAGO43k/YAGO43k_name_val.txt"
test_path = "./YAGO43k/YAGO43k_name_test.txt"
valid_path2 = "./YAGO43k/YAGO43k_name_val_clean.txt"
test_path2 = "./YAGO43k/YAGO43k_name_test_clean.txt"

# 对三元组进行清理
train_entities = set()
train_relations = set()
train_num = 0
with open(train_path, "r") as f:
    for line in f.readlines():
        e1, r, e2 = line.strip().split("\t")
        train_entities.add(e1)
        train_entities.add(e2)
        train_relations.add(r)
        train_num += 1

def check_train(path, output_path):
    valid_count = 0
    valid_num = 0
    f_valid = open(output_path, "w")
    with open(path, "r") as f:
        for line in f.readlines():
            e1, r, e2 = line.strip().split("\t")
            if e1 not in train_entities or e2 not in train_entities:
                valid_count += 1
                continue

            if r not in train_relations:
                valid_count += 1
                continue

            f_valid.write(line)
            valid_num += 1

    f_valid.close()
    return valid_count, valid_num

valid_count, valid_num = check_train(valid_path, valid_path2)
test_count, test_num = check_train(test_path, test_path2)
print(f"CLEAN-TRIPLET: TRAIN/VALID/TEST = {train_num}/{valid_num}(-{valid_count})/{test_num}(-{test_count})")

#**********************************clean the valid/test type data**************************************
train_entity_path = "./YAGO43k/YAGO43k_name_train.txt"

train_path = "./YAGO43k_Entity_Types/YAGO43k_Entity_Type_train.txt"
valid_path = "./YAGO43k_Entity_Types/YAGO43k_Entity_Type_valid.txt"
test_path = "./YAGO43k_Entity_Types/YAGO43k_Entity_Type_test.txt"

valid_path2 = "./YAGO43k_Entity_Types/YAGO43k_Entity_Type_valid_clean.txt"
test_path2 = "./YAGO43k_Entity_Types/YAGO43k_Entity_Type_test_clean.txt"

print(f"********** Cleaning Type data **********")

# clean the triplet
tot_train_entities = set()
tot_train_relations = set()

with open(train_entity_path, "r") as f:
    for line in f.readlines():
        e1, r, e2 = line.strip().split("\t")
        tot_train_entities.add(e1)
        tot_train_entities.add(e2)
        tot_train_relations.add(r)

train_entities2 = set()
train_types = set()
train_num = 0
with open(train_path, "r") as f:
    for line in f.readlines():
        e, et = line.strip().split("\t")
        train_entities2.add(e)
        train_types.add(et)
        train_num += 1

with open("./type2id.txt", "w") as f:
    count = 0
    for et in train_types:
        f.write("\t".join([et, str(count)]))
        f.write("\n")
        count += 1
    
with open("./entity2id.txt", "w") as f:
    count = 0
    for et in tot_train_entities:
        f.write("\t".join([et, str(count)]))
        f.write("\n")
        count += 1

with open("./relation2id.txt", "w") as f:
    count = 0
    for et in tot_train_relations:
        f.write("\t".join([et, str(count)]))
        f.write("\n")
        count += 1

def check(path, train_entities):
    """
    Delete the samples, the entities of which existing in valid/test dataset while not existing
    in YAGO43k_name_train.txt.
    """
    elements = path.split(r"/")
    new_path = os.path.join(*elements[:-1], elements[-1][:-4] + "_clean.txt")
    f2 = open(new_path, "w")
    with open(path, "r") as f:
        count = 0
        new_count = 0
        for line in f.readlines():
            e, et = line.strip().split("\t")
            if e not in train_entities:
                count += 1
                continue

            f2.write(line)
            new_count += 1

        # print(f"Cleaned samples because of invalid entities(compared to the triplet data): {new_count}(-{count})")

    f2.close()
    return count, new_count

def check_train2(path, output_path):
    """
    Delete the samples of which either types or entities existing in valid/test dataset while not existing
    in YAGO43k_Entity_Type_train.txt.
    """
    valid_count = 0
    valid_num = 0
    f_valid = open(output_path, "w")
    with open(path, "r") as f:
        for line in f.readlines():
            e, et = line.strip().split("\t")
            if e not in train_entities2:
                valid_count += 1
                continue

            if et not in train_types:
                valid_count += 1
                continue

            f_valid.write(line)
            valid_num += 1
    f_valid.close()
    return valid_count, valid_num

valid_count, valid_num = check_train2(valid_path, valid_path2)
test_count, test_num = check_train2(test_path, test_path2)

valid_count, valid_num = check(valid_path2, tot_train_entities)
test_count, test_num = check(test_path2, tot_train_entities)
train_count, train_num = check(train_path, tot_train_entities)


print(f"CLEAN-TYPE: TRAIN/VALID/TEST = {train_num}(-{train_count})/{valid_num}(-{valid_count})/{test_num}(-{test_count})")

#********************************************************************************************

trainTriplePaths = ["./YAGO43k/YAGO43k_name_train.txt",
                    "./YAGO43k/YAGO43k_name_val_clean.txt",
                    "./YAGO43k/YAGO43k_name_test_clean.txt"
                    ]
trainTypePaths = ["./YAGO43k_Entity_Types/YAGO43k_Entity_Type_train_clean.txt",
                  "./YAGO43k_Entity_Types/YAGO43k_Entity_Type_valid_clean_clean.txt",
                  "./YAGO43k_Entity_Types/YAGO43k_Entity_Type_test_clean_clean.txt"
                  ]
savingTypeRelationTypePaths = ["./type-relation-type-train.txt",
                               "./type-relation-type-valid.txt",
                               "./type-relation-type-test.txt"
                               ]

# savingTypeRelationTypeUniquePaths = ["./type-relation-type-unique-with-frequency-train.txt",
#                                      "./type-relation-type-unique-with-frequency-valid.txt",
#                                      "./type-relation-type-unique-with-frequency-test.txt"]



total_types = set()
total_entities = set()
print(f"********** Generating TRT data **********")

for i in range(3):
    trainTriplePath = trainTriplePaths[i]
    trainTypePath = trainTypePaths[i]
    savingTypeRelationTypePath = savingTypeRelationTypePaths[i]
    # savingTypeRelationTypeUniquePath = savingTypeRelationTypeUniquePaths[i]
    print("Processing ... "+trainTriplePath)
    triplesEntityRelationEntity = []
    with open(trainTriplePath, "r") as f:
        for line in f.readlines():
            triplesEntityRelationEntity.append(tuple(line.strip().split("\t")))
    print(f"In total, there are {len(triplesEntityRelationEntity)} samples")

    types  = set()
    entityEntityTypeDict = defaultdict(list)
    with open(trainTypePath, "r") as f:
        for line in f.readlines():
            entityId, entityType = line.strip().split("\t")
            entityEntityTypeDict[entityId].append(entityType)
            total_types.add(entityType)
            types.add(entityType)

    totalTypes2 = sum(map(lambda x: len(x), entityEntityTypeDict.values()))

    totalEntity = len(entityEntityTypeDict.keys())
    print(f"Based on type data, there are {totalEntity} entities and {len(types)} types.")

    triplesTypeRelationType = []
    entityEntityTypeDict1 = deepcopy(entityEntityTypeDict)
    entityEntityTypeDict2 = deepcopy(entityEntityTypeDict)
    count = 0
    for spo in triplesEntityRelationEntity:
        leftEntity, relation, rightEntity = spo
        total_entities.add(leftEntity)
        total_entities.add(rightEntity)
        while(entityEntityTypeDict1[leftEntity]):
            leftType = entityEntityTypeDict1[leftEntity].pop()
            tmp = deepcopy(entityEntityTypeDict2[rightEntity])
            while(tmp):
                rightType = tmp.pop()
                triplesTypeRelationType.append(tuple([leftType, relation, rightType]))
                count += 1

    print(f"In total, there are {len(triplesTypeRelationType)} type2type samples")

    with open(savingTypeRelationTypePath, 'w') as f:
        for line in triplesTypeRelationType:
            f.write("\t".join(line)+"\n")

print(f"Based on triplets, there are {len(total_types)}/{len(total_entities)} kinds of types/entities")


















