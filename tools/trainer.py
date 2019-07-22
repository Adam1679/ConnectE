from tools.util import *

class trainer(object):
    def __init__(self, train_data, nbatch, opt, margin, norm, **kwargs):
        self.optimizers = opt
        self.train_data = train_data
        self.nbatch = nbatch
        self.batch_size = len(train_data) // nbatch + 1
        self.margin = margin
        self.loss = 0
        self.violations = 0
        self.norm = norm
        self.logger = kwargs.get('logger', None)

    def before_epoch(self):

        self.loss = 0
        self.violations = 0
        random.shuffle(self.train_data)

    def output(self, content):
        if self.logger is None:
            print(content)
        else:
            self.logger.info(content)

class transE_trainer(trainer):
    def __init__(self, train_data, batch_size, opt, margin, entityList, norm, **kwargs):
        super().__init__(train_data, batch_size, opt, margin, norm, **kwargs)
        self.entityList = entityList
        self.set = set(train_data)

    def update_one_epoch(self, epoch,
                         train_entity=True,
                         normalize_entity=True,
                         train_relation=False,
                         normalize_relation=False):
        self.before_epoch()
        count = 0
        for j in range(0, len(self.train_data), self.batch_size):
            count += 1
            self.batch_loss = 0
            Sbatch = self.train_data[j: j + self.batch_size]
            pxs, nxs = [], []  # 初始化positive sample和负样本negative sample
            for h_, l_, t_ in Sbatch:
                new_h_ = getRandomObj(h_, self.entityList)
                new_t_ = getRandomObj(t_, self.entityList)
                while ((new_h_, l_, t_) in self.set):
                    new_h_ = getRandomObj(h_, self.entityList)

                while ((h_, l_, new_t_) in self.set):
                    new_t_ = getRandomObj(t_, self.entityList)

                pxs.append((h_, l_, t_))
                pxs.append((h_, l_, t_))
                nxs.append((new_h_, l_, t_))
                nxs.append((h_, l_, new_t_))

            self.update(pxs, nxs,
                         train_entity,
                         normalize_entity,
                         train_relation,
                         normalize_relation)
            # self.output(f"{self.__class__.__name__}: [{count}] minibatch loss {self.batch_loss}")

        self.output(f"{self.__class__.__name__}: epoch:{epoch},loss{self.loss/count}, "
                    f"margin : {100*self.violations/len(self.train_data):.2f}%")
        return self.loss

    def update(self, pxs, nxs, train_entity=True,
                         normalize_entity=True,
                         train_relation=False,
                         normalize_relation=False):

        """pxs: [(head_id, label_id, tail_id)]"""
        h_p, l_p, t_p = np.array(list(zip(*pxs)))
        h_n, l_n, t_n = np.array(list(zip(*nxs)))
        # distance_p.shape = (batch_size,)
        distance_p = self.distance(h_p, l_p, t_p)
        distance_n = self.distance(h_n, l_n, t_n)
        # 需要更新的sample的index array
        loss = np.maximum(self.margin + distance_p - distance_n, 0)
        loss_ = np.mean(loss)
        self.loss += loss_
        self.batch_loss = loss_
        ind = np.where(loss > 0)[0]

        self.violations += len(ind)
        if len(ind) == 0:  # 若没有样本需要更新向量,则返回
            return

        h_p2, l_p2, t_p2 = list(h_p[ind]), list(l_p[ind]), list(t_p[ind])
        h_n2, l_n2, t_n2 = list(h_n[ind]), list(l_n[ind]), list(t_n[ind])

        # step 1 : 计算d = (head + label - tail), gradient_p = (len(ind), dim)
        gradient_p = self.optimizers['entityMat'].param[h_p2]\
                     + self.optimizers['relationMat'].param[l_p2]\
                     - self.optimizers['entityMat'].param[t_p2]

        gradient_n = self.optimizers['entityMat'].param[h_n2]\
                     + self.optimizers['relationMat'].param[l_n2]\
                     - self.optimizers['entityMat'].param[t_n2]

        if self.norm == 'L1':
            # L1正则化的次梯度
            gradient_p = np.sign(gradient_p)
            gradient_n = np.sign(gradient_n)
        else:
            gradient_p = gradient_p*2
            gradient_n = gradient_n*2

        # 所有需要更新的entity_id list
        if train_entity:
            tot_entity = h_p2 + t_p2 + h_n2 + t_n2
            # step 2 : 计算一个中间矩阵M,方便整合positive和negative sample的所有entity（有重复的）到一个unique_entity
            unique_idx_e, M_e, tot_update_time_e = grad_sum_matrix(tot_entity)

            # step 3 : 计算每个entity的梯度
            # M.shape = (num_of_unique_entities, num_of_samples),
            # M2.shape = (num_of_samples, dim)
            # gradient.shape = (num_of_unique_entities, dim)除以n表示一个batch中的平均梯度,例如gradient.shape = (4, 3)
            # M = [[1,1,0,1],
            #      [0,0,1,0],
            #      [1,1,1,1]]
            # M2 = [[0.1, 0.3, 0.1],
            #       [-0.1, -0.3, -0.1],
            #       [0.18, 0.11, 0.43],
            #       [-0.18, -0.11, -0.43]]
            # gradient[0,0] = (0.1-0.1-0.18) / 3, 即0号entity在所有的正(负)样本的head(tail)中出现了3次（应该是偶数,我只是举例)
            M2_e = np.vstack((gradient_p, # 正样本的head的梯度
                            -gradient_p, # 正样本的tail的梯度
                            -gradient_n, # 负样本的head的梯度
                            gradient_n)) # 负样本的tail的梯度
            gradient_e = M_e.dot(M2_e) / tot_update_time_e
            self.optimizers['entityMat']._update(gradient_e, unique_idx_e)
            if normalize_entity:
                normalize(self.optimizers['entityMat'].param, unique_idx_e)

        # step 4 : 计算每个relation的梯度
        if train_relation:
            tot_relations = l_p2 + l_n2
            unique_idx_r, M_r, tot_update_time_r = grad_sum_matrix(tot_relations)
            M2_r = np.vstack((gradient_p, # 正样本的relation的梯度
                            -gradient_n) # 负样本的relation的梯度
                           )
            gradient_r = M_r.dot(M2_r) / tot_update_time_r

            self.optimizers['relationMat']._update(gradient_r, unique_idx_r)
            if normalize_relation:
                normalize(self.optimizers['relationMat'].param, unique_idx_r)

    def distance(self, ss, ps, os):
        score = self.optimizers['entityMat'].param[ss]\
                + self.optimizers['relationMat'].param[ps]\
                - self.optimizers['entityMat'].param[os]
        if self.norm == 'L1':
            score = np.abs(score)
        else:
            score = score ** 2
        return np.sum(score, axis=1)

class e2t_trainer(trainer):
    def __init__(self, train_data, nbatch, opt, margin, typeList, norm, **kwargs):
        super().__init__(train_data, nbatch, opt, margin, norm, **kwargs)
        self.typeList = typeList
        self.set = set(train_data)

    def update_one_epoch(self, epoch,
                         train_entity=False,
                         normalize_entity=True,
                         train_type=True,
                         normalize_type=False,
                         train_M=True):
        count = 0
        self.before_epoch()
        for j in range(0, len(self.train_data), self.batch_size):
            count += 1
            self.batch_loss = 0
            Sbatch = self.train_data[j: j + self.batch_size]
            pxs, nxs = [], []  # 初始化positive sample和负样本negative sample
            for e, et in Sbatch:
                new_et = getRandomObj(et, self.typeList)
                while((e, new_et) in self.set):
                    new_et = getRandomObj(et, self.typeList)
                pxs.append((e, et))
                nxs.append((e, new_et))

            self.update(pxs, nxs,
                         train_type,
                         normalize_type,
                         train_M)
            # self.output(f"{self.__class__.__name__}: [{count}] minibatch loss {self.batch_loss}")

        self.output(f"{self.__class__.__name__}: epoch:{epoch}, loss{self.loss/count}, "
                    f"margin : {100*self.violations/len(self.train_data):.2f}%")
        return self.loss

    def update(self, pxs, nxs,
               train_type=True,
               normalize_type=False,
               train_M=True):

        e_p, et_p = np.array(list(zip(*pxs)))
        e_n, et_n = np.array(list(zip(*nxs)))
        distance_p = self.distance(e_p, et_p)
        distance_n = self.distance(e_n, et_n)
        loss = np.maximum(self.margin + distance_p - distance_n, 0)
        loss_ = np.mean(loss)
        self.loss += loss_
        self.batch_loss = loss_
        ind = np.where(loss > 0)[0]
        self.violations += len(ind)
        if len(ind) == 0:  # 若没有样本需要更新向量,则返回
            return
        e_p2, et_p2 = list(e_p[ind]), list(et_p[ind])
        e_n2, et_n2 = list(e_n[ind]), list(et_n[ind])

        gradient_p = self.optimizers['entityMat'].param[e_p2].dot(self.optimizers['entity2typeMat'].param.T) - \
                self.optimizers['typeMat'].param[et_p2]

        gradient_n = self.optimizers['entityMat'].param[e_n2].dot(self.optimizers['entity2typeMat'].param.T) - \
                self.optimizers['typeMat'].param[et_n2]

        if self.norm == 'L1':
            # L1正则化的次梯度
            gradient_p = np.sign(gradient_p)
            gradient_n = np.sign(gradient_n)
        else:
            gradient_p = gradient_p*2
            gradient_n = gradient_n*2
        if train_M:
            gradient_M_p = gradient_p.T.dot(self.optimizers['entityMat'].param[e_p2])
            gradient_M_n = gradient_n.T.dot(self.optimizers['entityMat'].param[e_n2])
            gradient_M = (gradient_M_p - gradient_M_n) / len(e_p2)
            self.optimizers['entity2typeMat']._update(gradient_M)

        if train_type:
            tot_types = et_p2 + et_n2
            unique_idx_e, M_e, tot_update_time_e = grad_sum_matrix(tot_types)
            M2_et = np.vstack((-gradient_p,
                            gradient_n))
            gradient_et = M_e.dot(M2_et) / tot_update_time_e
            self.optimizers['typeMat']._update(gradient_et, unique_idx_e)
            if normalize_type:
                normalize(self.optimizers['typeMat'].param, unique_idx_e)

    def distance(self, e, et):
        # score = E*M-ET
        score = self.optimizers['entityMat'].param[e].dot(self.optimizers['entity2typeMat'].param.T) - \
                self.optimizers['typeMat'].param[et]

        if self.norm == 'L1':
            score = np.abs(score)
        else:
            score = score ** 2

        return np.sum(score, axis=1)

class trt_trainer(trainer):
    def __init__(self, train_data, batch_size, opt, margin, typeList, norm, **kwargs):
        super().__init__(train_data, batch_size, opt, margin, norm, **kwargs)
        self.typeList = typeList
        self.set = set(train_data)

    def update_one_epoch(self, epoch,
                         train_type=True,
                         normalize_type=False,
                         train_relation=True,
                         normalize_relation=False):
        self.before_epoch()
        count = 0
        for j in range(0, len(self.train_data), self.batch_size):
            count += 1
            self.batch_loss = 0
            Sbatch = self.train_data[j: j + self.batch_size]
            pxs, nxs = [], []
            for h_, l_, t_ in Sbatch:
                new_h_ = getRandomObj(h_, self.typeList)
                new_t_ = getRandomObj(t_, self.typeList)
                while((new_h_, l_, t_) in self.set):
                    new_h_ = getRandomObj(h_, self.typeList)

                while ((h_, l_, new_t_) in self.set):
                    new_t_ = getRandomObj(t_, self.typeList)

                pxs.append((h_, l_, t_))
                pxs.append((h_, l_, t_))
                nxs.append((new_h_, l_, t_))
                nxs.append((h_, l_, new_t_))

            self.update(pxs, nxs,
                        train_type,
                        normalize_type,
                        train_relation,
                        normalize_relation
                        )
            # self.output(f"{self.__class__.__name__}: [{count}] minibatch loss {self.batch_loss}")

        self.output(f"{self.__class__.__name__}: epoch:{epoch},loss{self.loss/count}, "
                    f"margin : {100*self.violations/len(self.train_data):.2f}%")
        return self.loss


    def update(self, pxs, nxs,
               train_type=True,
               normalize_type=False,
               train_relation=True,
               normalize_relation=False):

        """pxs: [(head_id, label_id, tail_id)]"""
        h_p, l_p, t_p = np.array(list(zip(*pxs)))
        h_n, l_n, t_n = np.array(list(zip(*nxs)))
        # distance_p.shape = (batch_size,)
        distance_p = self.distance(h_p, l_p, t_p)
        distance_n = self.distance(h_n, l_n, t_n)
        # 需要更新的sample的index array
        loss = np.maximum(self.margin + distance_p - distance_n, 0)

        loss_ = np.mean(loss)
        self.loss += loss_
        self.batch_loss = loss_
        ind = np.where(loss > 0)[0]

        self.violations += len(ind)
        if len(ind) == 0:  # 若没有样本需要更新向量,则返回
            return

        h_p2, l_p2, t_p2 = list(h_p[ind]), list(l_p[ind]), list(t_p[ind])
        h_n2, l_n2, t_n2 = list(h_n[ind]), list(l_n[ind]), list(t_n[ind])

        # step 1 : 计算d = (head + label - tail), gradient_p = (len(ind), dim)
        gradient_p = self.optimizers['typeMat'].param[h_p2]\
                     + self.optimizers['relationMat'].param[l_p2].dot(self.optimizers['entity2typeMat'].param.T)\
                     - self.optimizers['typeMat'].param[t_p2]

        gradient_n = self.optimizers['typeMat'].param[h_n2]\
                     + self.optimizers['relationMat'].param[l_n2].dot(self.optimizers['entity2typeMat'].param.T)\
                     - self.optimizers['typeMat'].param[t_n2]

        if self.norm == 'L1':
            # L1正则化的次梯度
            gradient_p = np.sign(gradient_p)
            gradient_n = np.sign(gradient_n)
        else:
            gradient_p = gradient_p*2
            gradient_n = gradient_n*2

        # 所有需要更新的entity_id list
        if train_type:
            tot_entity = h_p2 + t_p2 + h_n2 + t_n2
            # step 2 : 计算一个中间矩阵M,方便整合positive和negative sample的所有entity（有重复的）到一个unique_entity
            unique_idx_e, M_e, tot_update_time_e = grad_sum_matrix(tot_entity)
            # step 3 : 计算每个type的梯度
            M2_e = np.vstack((gradient_p, # 正样本的head的梯度
                            -gradient_p, # 正样本的tail的梯度
                            -gradient_n, # 负样本的head的梯度
                            gradient_n)) # 负样本的tail的梯度

            gradient_e = M_e.dot(M2_e) / tot_update_time_e
            self.optimizers['typeMat']._update(gradient_e, unique_idx_e)
            if normalize_type:
                normalize(self.optimizers['typeMat'].param, unique_idx_e)

        # step 4 : 计算每个relation的梯度
        if train_relation:
            tot_relations = l_p2 + l_n2
            unique_idx_r, M_r, tot_update_time_r = grad_sum_matrix(tot_relations)
            M2_r = np.vstack((gradient_p, # 正样本的relation的梯度
                            -gradient_n) # 负样本的relation的梯度
                           ).dot(self.optimizers['entity2typeMat'].param)

            gradient_r = M_r.dot(M2_r) / tot_update_time_r

            self.optimizers['relationMat']._update(gradient_r, unique_idx_r)
            if normalize_relation:
                normalize(self.optimizers['relationMat'].param, unique_idx_r)

    def distance(self, ss, ps, os):
        score = self.optimizers['typeMat'].param[ss]\
                + self.optimizers['relationMat'].param[ps].dot(self.optimizers['entity2typeMat'].param.T)\
                - self.optimizers['typeMat'].param[os]

        if self.norm == 'L1':
            score = np.abs(score)
        else:
            score = score ** 2
        return np.sum(score, axis=1)

