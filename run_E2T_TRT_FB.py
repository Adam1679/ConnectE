import logging
import os
from tools.trainer import *
from tools.util import *
from tools.optimizer import *
from models.transE_e2t_trt import transE_e2t_trt

logger = logging.getLogger(__name__)
logger.setLevel(level=logging.DEBUG)
handler = logging.FileHandler(f"output/FB15K/E2T_TRT/log_ere_e2t_trt.txt")
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
    config = get_config("./config/FB15K/E2T_TRT/config.json")

    model = transE_e2t_trt(triplet, e2t, trt, relation2Id, entity2Id, type2Id, logger,
                           valid_e2t=valid_e2t,
                           test_e2t=test_e2t,
                           **config
                           )

    model.loadModel("output/FB15K/E2T_TRT/entityVector_e200d_et100_m2.txt",
                    "output/FB15K/E2T_TRT/relationVector_e200d_et100_m2.txt",
                    'output/FB15K/E2T_TRT/typeVector_e200d_et100_m2.txt',
                    'output/FB15K/E2T_TRT/entity2typeMat_e200d_et100_m2.npy')
    model.train(801)
    
    model.save(directory="output/FB15K/E2T_TRT")

    e = Type_Evaluator_trt(encode2id(test_e2t, model.entityId, model.typeId),
                           encode2id(valid_e2t + e2t + test_e2t, model.entityId, model.typeId),
                           encode2id(triplet, model.entityId, model.relationId, model.entityId),
                           logger=logger)

    e(model)
    e.save_prediction(path="output/FB15K/E2T_TRT/pos_e2t_trt", fpath="output/FB15K/E2T_TRT/fpos_e2t_trt")