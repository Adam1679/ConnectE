import logging
import os
from tools.trainer import *
from tools.util import *
from tools.optimizer import *
from models.transE_e2t import transE_e2t

logger = logging.getLogger(__name__)
logger.setLevel(level=logging.DEBUG)
# handler = logging.FileHandler(f"output/FB15K/E2T/log_ere_e2t.txt")
handler = logging.FileHandler(f"output/YAGO/E2T/log_ere_e2t.txt")
handler.setLevel(logging.INFO)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.addHandler(console)

logger.info("Start print log")


if __name__ == "__main__":
    # 训练模型
    entity2Id = loadEntityId(path="data/YAGO/entity2id.txt")
    relation2Id = loadRelationId("data/YAGO/relation2id.txt")
    type2Id = loadTypeId("data/YAGO/type2id.txt")
    # 加载三元组
    triplet = loadTriplet("data/YAGO/YAGO43k/YAGO43k_name_train.txt")
    # 加载entity, entity_type 二元组
    e2t = loadEntity2Type(path="data/YAGO/YAGO43k_Entity_Types/YAGO43k_Entity_Type_train_clean.txt")
    valid_e2t = loadEntity2Type(path="data/YAGO/YAGO43k_Entity_Types/YAGO43k_Entity_Type_valid_clean_clean.txt")
    test_e2t = loadEntity2Type(path="data/YAGO/YAGO43k_Entity_Types/YAGO43k_Entity_Type_test_clean_clean.txt")

    # 加载type_relation_type 三元组
    trt = loadTriplet("data/YAGO/type-relation-type-train.txt")
    config = get_config("./config/YAGO/E2T/config.json")
    model = transE_e2t(triplet, e2t, trt, relation2Id, entity2Id, type2Id, logger,
                           valid_e2t=valid_e2t,
                           test_e2t=test_e2t,
                           **config
                       )

    model.loadModel("output/YAGO/E2T/entityVector_e250d_et125_m1.txt",
                    "output/YAGO/E2T/relationVector_e250d_et125_m1.txt",
                    'output/YAGO/E2T/typeVector_e250d_et125_m1.txt',
                    'output/YAGO/E2T/entity2typeMat_e250d_et125_m1.npy')
    # model.evaluator(model)
    model.train(200)
    model.save(directory="output/YAGO/E2T")