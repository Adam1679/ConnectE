import logging
import sys
import os
from tools.trainer import *
from tools.util import *
from tools.optimizer import *
from models.transE_e2t import transE_e2t

logger = logging.getLogger(__name__)
logger.setLevel(level=logging.DEBUG)
handler = logging.FileHandler(f"output/FB15K/E2T/log_ere_e2t.txt")
handler.setLevel(logging.INFO)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.addHandler(console)
logger.info("Start print log")


if __name__ == "__main__":
    entity2Id = loadEntityId()
    relation2Id = loadRelationId()
    type2Id = loadTypeId()
    triplet = loadTriplet("data/FB15K/origin/freebase_mtr100_mte100-train.txt")
    e2t = loadEntity2Type(path="data/FB15K/origin/FB15k_Entity_Type_train.txt")
    valid_e2t = loadEntity2Type(path="data/FB15K/FB15k_Entity_Type_valid_clean.txt")
    test_e2t = loadEntity2Type(path="data/FB15K/FB15k_Entity_Type_test_clean.txt")
    trt = loadTriplet("data/FB15K/type-relation-type-train.txt")
    config = get_config("./config/FB15K/E2T/config.json")
    model = transE_e2t(triplet, e2t, trt, relation2Id, entity2Id, type2Id, logger,
                           valid_e2t=valid_e2t,
                           test_e2t=test_e2t,
                           **config
                           )

    model.train(801)
    model.save(directory="output/FB15K/E2T")