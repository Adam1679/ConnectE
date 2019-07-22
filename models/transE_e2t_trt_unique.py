import logging
import sys
import os
from tools.trainer import *
from tools.util import *
from tools.optimizer import *




class transE_e2t_trt_unique(object):
    name = "E2T_TRT_U"
    def __init__(self, train_ere: list,
                 train_e2t: list,
                 train_trt: list,
                 relationId: dict,
                 entityId: dict,
                 typeId: dict,
                 logger,
                 nbatch=64,
                 learning_rate=0.01,
                 entity_dim=100,
                 type_dim=50,
                 norm='L2',
                 margin=2,
                 seed=52,
                 valid_ere=None,
                 test_ere=None,
                 valid_e2t=None,
                 test_e2t=None,
                 valid_trt=None,
                 test_trt=None,
                 optimizer='AdaGrad',
                 evaluator='Relation_Evaluator',
                 evaluator_time = 5,
                 ):

        self.nbatch = nbatch
        self.learning_rate = learning_rate
        self.norm = norm
        self.margin = margin  # 论文中{1, 2, 10}
        self.entity_dim = entity_dim  # 论文中{20, 50}
        self.max_loss = 100
        self.type_dim = type_dim
        self.test_epoth_freq = evaluator_time
        self.logger=logger

        self.ere_raw_data = train_ere
        self.train_ere = []

        self.e2t_raw_data = train_e2t
        self.train_e2t = []

        self.trt_raw_data = train_trt
        self.train_trt = []

        self.relationId = relationId  # {entity: id, ...}
        self.relationList = list(relationId.values())
        self.relationIdReverse = {v: k for k, v in relationId.items()}

        self.entityId = entityId
        self.entityList = list(entityId.values())
        self.entityIdReverse = {v: k for k, v in entityId.items()}

        self.typeId = typeId
        self.typeList = list(typeId.values())
        self.typeIdReverse = {v: k for k, v in typeId.items()}


        # 可以训练的参数
        self.entityMat = np.zeros((len(entityId), entity_dim))
        self.relationMat = np.zeros((len(relationId), entity_dim))
        self.typeMat = np.zeros((len(typeId), type_dim))
        self.entity2typeMat = np.zeros((type_dim, entity_dim))

        self.train_ere = encode2id(self.ere_raw_data, self.entityId, self.relationId, self.entityId)
        self.train_e2t = encode2id(self.e2t_raw_data, self.entityId, self.typeId)
        self.train_trt = encode2id(self.trt_raw_data, self.typeId, self.relationId, self.typeId)

        mod = sys.modules[__name__]
        opt = getattr(mod, optimizer)
        self.optimizers = {'relationMat': opt(self.relationMat, learning_rate),
                           'entityMat': opt(self.entityMat, learning_rate),
                           'typeMat': opt(self.typeMat, learning_rate),
                           'entity2typeMat': opt(self.entity2typeMat, learning_rate)}

        self.transE_trainer = transE_trainer(self.train_ere,
                                             self.nbatch,
                                             self.optimizers,
                                             self.margin,
                                             self.entityList,
                                             self.norm,
                                             logger=logger
                                             )

        self.e2t_trainer = e2t_trainer(self.train_e2t,
                                       self.nbatch,
                                       self.optimizers,
                                       self.margin,
                                       self.typeList,
                                       self.norm,
                                       logger=logger,
                                       )

        self.trt_trainer = trt_trainer(self.train_trt,
                                       self.nbatch,
                                       self.optimizers,
                                       self.margin,
                                       self.typeList,
                                       self.norm,
                                       logger=logger,
                                       )
        self.initialize()

        if valid_e2t and test_e2t:
            eval = getattr(mod, evaluator)
            # self.evaluator = eval(encode2id(test_ere, self.entityId, self.relationId),
            #                       encode2id(valid_ere + train_ere + test_ere, self.entityId, self.relationId),
            #                       logger)
            self.evaluator = eval(encode2id(test_e2t, self.entityId, self.typeId),
                                  encode2id(valid_e2t + train_e2t + test_e2t, self.entityId, self.typeId),
                                  logger)

        elif valid_ere and test_ere:
            eval = getattr(mod, evaluator)
            self.evaluator = eval(encode2id(test_ere, self.entityId, self.relationId),
                                  encode2id(valid_ere + train_ere + test_ere, self.entityId, self.relationId),
                                  logger)

        else:
            self.evaluator = None

        self.epoch = 1
        self.seed = seed


    def initialize(self):
        # 初始化向量

        bnd = np.sqrt(6) / np.sqrt(self.entityMat.shape[0] + self.entityMat.shape[1])
        for e_ in self.entityList:
            self.entityMat[e_] = \
                np.random.uniform(-bnd, bnd, self.entity_dim)
        normalize(self.entityMat)
        self.logger.info(f"entityMat:{self.entityMat.shape}")

        bnd = np.sqrt(6) / np.sqrt(self.relationMat.shape[0] + self.relationMat.shape[1])
        for p_ in self.relationList:
            self.relationMat[p_] = \
                np.random.uniform(-bnd, bnd, self.entity_dim)
        self.relationMat = normalize(self.relationMat)
        self.logger.info(f"relationMat: {self.relationMat.shape}")

        bnd = np.sqrt(6) / np.sqrt(self.typeMat.shape[0] + self.typeMat.shape[1])
        for p_ in self.typeList:
            self.typeMat[p_] = \
                np.random.uniform(-bnd, bnd, self.type_dim)
        self.typeMat = normalize(self.typeMat)
        self.logger.info(f"typeMat: {self.typeMat.shape}")

        bnd = np.sqrt(6) / np.sqrt(self.entity2typeMat.shape[0] + self.entity2typeMat.shape[1])
        self.entity2typeMat[:] = np.random.uniform(-bnd, bnd, self.entity2typeMat.shape)
        # self.entity2typeMat = normalize(self.entity2typeMat)
        self.logger.info(f"entity2typeMat: {self.entity2typeMat.shape}")

        ########################

        self.et_t_set, self.t_et_set = self.construct(self.train_trt)

    def construct(self, trt):
        et_t_set = defaultdict(list)
        t_et_set = defaultdict(list)
        self.trt_set = set()
        for et1, r, et2 in trt:
            et_t_set[(et1, r)].append(et2)
            t_et_set[(r, et2)].append(et1)
            self.trt_set.add((et1, r, et2))

        return et_t_set, t_et_set

    def train(self, epoch=1000):
        self.logger.info('Start')
        # np.random.seed(self.seed)
        past_fmrr = []
        time = 0
        for self.epoch in range(1, epoch):

            transE_loss = self.transE_trainer.update_one_epoch(self.epoch, train_relation=True)
            e2t_loss = self.e2t_trainer.update_one_epoch(self.epoch)
            trt_loss = self.trt_trainer.update_one_epoch(self.epoch)

            # self.x = self.logger.info(f"完成{self.epoch}轮训练，损失函数为{self.loss/count}, 超过margin的样本数量为{self.violations}")
            if self.evaluator and self.epoch % self.test_epoth_freq == 0:
                fmrr = self.evaluator(self)
                past_fmrr.append(fmrr)

        return past_fmrr

    def save(self, directory="output"):
        self.logger.info("Saving Type Vector")
        dir = os.path.join(directory, f"typeVector_e{self.entityMat.shape[1]}d_et{self.typeMat.shape[1]}_m{self.margin}.txt")
        with open(dir, 'w') as f:
            for et_ in self.typeList:
                f.write(self.typeIdReverse[et_] + "\t")
                f.write(str(self.typeMat[et_].tolist()))
                f.write("\n")

        self.logger.info("Saving entity2type Matrices")
        dir = os.path.join(directory, f"entity2typeMat_e{self.entityMat.shape[1]}d_et{self.typeMat.shape[1]}_m{self.margin}")
        np.save(dir, self.entity2typeMat)
        self.logger.info("Saving Entity Vector")
        dir = os.path.join(directory, f"entityVector_e{self.entityMat.shape[1]}d_et{self.typeMat.shape[1]}_m{self.margin}.txt")
        with open(dir, 'w') as f:
            for e_ in self.entityList:
                f.write(self.entityIdReverse[e_] + "\t")
                f.write(str(self.entityMat[e_].tolist()))
                f.write("\n")
        self.logger.info("Saving Relation Vector")
        dir = os.path.join(directory, f"relationVector_e{self.entityMat.shape[1]}d_et{self.typeMat.shape[1]}_m{self.margin}.txt")
        with open(dir, 'w') as f:
            for e_ in self.relationList:
                f.write(self.relationIdReverse[e_] + "\t")
                f.write(str(self.relationMat[e_].tolist()))
                f.write("\n")

    def loadModel(self, entity_path, replation_path, type_path=None, entity2type_path=None):
        entityVectors = loadVectors(path=entity_path)
        relationVectors = loadVectors(path=replation_path)
        for e, v in entityVectors.items():
            self.entityMat[self.entityId[e]] = v
        for r, v in relationVectors.items():
            self.relationMat[self.relationId[r]] = v

        if type_path:
            typeVectors = loadVectors(type_path)
            for et, v in typeVectors.items():
                self.typeMat[self.typeId[et]] = v

        if entity2type_path:
            with open(entity2type_path, 'rb') as f:
                self.entity2typeMat[:] = np.load(f)

    def _scores_et(self, e, et=None):
        # 给定一个三元组，求替换了label的
        if et is None:
            score = self.optimizers['entityMat'].param[e].dot(self.optimizers['entity2typeMat'].param.T) - \
                    self.optimizers['typeMat'].param
            if self.norm == 'L1':
                score = np.abs(score)
            else:
                score = score ** 2
        else:
            score = self.optimizers['entityMat'].param[e].dot(self.optimizers['entity2typeMat'].param.T) - \
                    self.optimizers['typeMat'].param[et]
            score = score.reshape(1,-1)
            if self.norm == 'L1':
                score = np.abs(score)
            else:
                score = score ** 2

        return np.sum(score, axis=1)

    def _scores_trt(self, et, r, change_head=True, et2=None):

        T = self.optimizers['typeMat'].param
        R = self.optimizers['relationMat'].param
        M = self.optimizers['entity2typeMat'].param
        if et2 is None:
            if change_head:
                score = T + R[r].dot(M.T) - T[et]
                tmp = np.mean(score, axis=0)
                tmp = np.tile(tmp,(score.shape[0], 1))
                tmp[self.t_et_set[(et, r)], :] = score[self.t_et_set[(et, r)], :]
                score = tmp
            else:
                score = T + R[r].dot(M.T) - T[et]
                tmp = np.mean(score, axis=0)
                tmp = np.tile(tmp,(score.shape[0], 1))
                tmp[self.t_et_set[(et, r)], :] = score[self.t_et_set[(et, r)], :]
                score = tmp

            if self.norm == 'L1':
                score = np.abs(score)
            else:
                score = score ** 2
        else:
            if change_head:
                if (et2, r, et) in self.trt_set:
                    score = T[et2] + R[r].dot(M.T) - T[et]
                else:
                    score = np.array([self.max_loss,])
            else:
                if (et2, r, et) in self.trt_set:
                    score = T[et] + R[r].dot(M.T) - T[et2]
                else:
                    score = np.array([self.max_loss,])
                
            score = score.reshape(1, -1)
            if self.norm == 'L1':
                score = np.abs(score)
            else:
                score = score ** 2

        return np.sum(score, axis=1)

    def get_type_size(self):
        return self.typeMat.shape[0]

    def get_entity_size(self):
        return self.entityMat.shape[0]



