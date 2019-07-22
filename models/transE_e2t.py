from models.transE_e2t_trt import transE_e2t_trt
import pickle
import numpy as np
import os
import json

class transE_e2t(transE_e2t_trt):
    name = "E2T Model"
    def train(self, epoch=1000):
        self.logger.info('Start')
        # np.random.seed(self.seed)
        self.past_fmrr = []
        for self.epoch in range(1, epoch+1):

            transE_loss = self.transE_trainer.update_one_epoch(self.epoch, train_relation=True)
            e2t_loss = self.e2t_trainer.update_one_epoch(self.epoch)

            # self.x = self.logger.info(f"完成{self.epoch}轮训练，损失函数为{self.loss/count}, 超过margin的样本数量为{self.violations}")
            if self.evaluator and self.epoch % self.test_epoth_freq == 0:
                fmrr = self.evaluator(self)
                self.past_fmrr.append(fmrr)
                # if len(self.past_fmrr) > 10:
                #     if fmrr <= max(self.past_fmrr):
                #         time += 1
                #     else:
                #         time = 0
                # if time >= 3:
                #     self.logger.info("early_stopping")
                #     break

        return self.past_fmrr

class transE_e2t_loaded(object):
    def __init__(self, name, ete_model_path=None, kg_model_path=None, entityIdReverse=None, typeIdReverse=None):
        self.name = name
        with open(os.path.join(ete_model_path, "kg_model.model.ent.json"), "r") as f:
            s = f.read()
            self.entityId = json.loads(s)['ent_id']

        with open(os.path.join(ete_model_path, "ete_model.model.et.json"), "r") as f:
            s = f.read()
            self.typeId = json.loads(s)['et_id']
        
        self.entityIdReverse = entityIdReverse
        self.typeIdReverse = typeIdReverse
        self.ET = self.load_e2t_model(ete_model_path)
        self.E = self.load_kg_model(kg_model_path)

    def load_e2t_model(self, model_path):
        ET = np.load(os.path.join(model_path, "ete_model.model.nry"), fix_imports=True)
        return ET

    def load_kg_model(self, model_path):
        ET = np.load(os.path.join(model_path, "kg_model.model.nry"), fix_imports=True)
        return ET

    def _scores_et(self, e, et):
        e = self.entityId[self.entityIdReverse[e]]
        et = self.typeId[self.typeIdReverse[et]]
        return self.scores_t(e, et)

    def scores_t(self, e, t):
        score = (self.ET[t] - self.E[e])**2
        return np.sum(score)

    def get_type_size(self):
        return self.ET.shape[1]



