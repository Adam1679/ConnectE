import numpy as np
from collections import defaultdict, Counter
from sklearn.metrics import accuracy_score
import pickle

class optimizer(object):
    def __init__(self, param, learning_rate, post=None):
        self.param = param
        self.learning_rate = learning_rate

class SGD(optimizer):
    """
    SGD updates on a parameter
    """

    def _update(self, g, idx=None):
        if idx:
            self.param[idx] -= self.learning_rate * g
        else:
            self.param -= self.learning_rate * g

class AdaGrad(optimizer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.p2 = np.zeros_like(self.param)

    def _update(self, g, idx=None):
        self.p2[idx] += g * g
        H = np.maximum(np.sqrt(self.p2[idx]), 1e-7)
        self.param[idx] -= self.learning_rate * g / H

class Evaluator(object):
    def __init__(self, test_triplet, logger=None):
        self.logger = logger
        self.xs = test_triplet
        self.tot = len(test_triplet)
        self.pos = None
        self.fpos = None

    def __call__(self, model):
        pos_v, fpos_v = self.positions(model)
        self.pos = pos_v
        self.fpos = fpos_v
        fmrr = self.p_ranking_scores(pos_v, fpos_v, model.epoch, 'VALID')
        return fmrr

    def positions(self, mdl):
        raise NotImplementedError

    def p_ranking_scores(self, pos, fpos, epoch, txt):
        rpos = [p for k in pos.keys() for p in pos[k]]
        frpos = [p for k in fpos.keys() for p in fpos[k]]
        fmrr = self._print_pos(
            np.array(rpos),
            np.array(frpos),
            epoch, txt)
        return fmrr

    def _print_pos(self, pos, fpos, epoch, txt):
        mrr, mean_pos, hits = self.compute_scores(pos)
        fmrr, fmean_pos, fhits = self.compute_scores(fpos)
        if self.logger:
            self.logger.info(
                f"[{epoch: 3d}] {txt}: MRR = {mrr:.2f}/{fmrr:.2f}, "
                f"Mean Rank = {mean_pos:.2f}/{fmean_pos:.2f}, "
                f"Hits@1 = {hits[0]:.2f}/{fhits[0]:.2f}, "
                f"Hits@3 = {hits[1]:.2f}/{fhits[1]:.2f}, "
                f"Hits@10 = {hits[2]:.2f}/{fhits[2]:.2f}"
            )
        else:
            print(
                f"[{epoch: 3d}] {txt}: MRR = {mrr:.2f}/{fmrr:.2f}, "
                f"Mean Rank = {mean_pos:.2f}/{fmean_pos:.2f}, "
                f"Hits@1 = {hits[0]:.2f}/{fhits[0]:.2f}, "
                f"Hits@3 = {hits[1]:.2f}/{fhits[1]:.2f}, "
                f"Hits@10 = {hits[2]:.2f}/{fhits[2]:.2f}"
            )
        return fmrr

    def compute_scores(self, pos, hits=None):
        if hits is None:
            hits = [1, 3, 10]
        mrr = np.mean(1.0 / pos)
        mean_pos = np.mean(pos)
        hits_results = []
        for h in range(0, len(hits)):
            k = np.mean(pos <= hits[h])
            k2 = k.sum()
            hits_results.append(k2 * 100)
        return mrr, mean_pos, hits_results

    def save_prediction(self, path="output/pos", fpath="output/fpos_e2t_trt"):
        with open(path, 'wb') as f:
            pickle.dump(self.pos, f)
        with open(fpath, 'wb') as f:
            pickle.dump(self.fpos, f)

    def load_prediction(self,path="output/pos", fpath="output/fpos_e2t_trt"):
        with open(path, 'rb') as f:
            self.pos = pickle.load(f)

        with open(fpath, 'rb') as f:
            self.fpos = pickle.load(f)

class Type_Evaluator(Evaluator):

    def __init__(self, xs, true_tuples, logger=None):
        super(Type_Evaluator, self).__init__(xs, logger)
        self.idx = defaultdict(list)   # defaultdict
        self.tt = defaultdict(list)  # true tuples
        self.sz = len(xs)

        for e, t in xs:
            self.idx[e].append((t))

        for e, t in true_tuples:
            self.tt[e].append(t)

        self.idx = dict(self.idx)
        self.tt = dict(self.tt)


    def __call__(self, model, path=None, fpath=None):
        if path and fpath:
            self.load_prediction(path, fpath)
        else:
            pos_v, fpos_v = self.positions(model)
            self.pos = pos_v
            self.fpos = fpos_v

        return self.et_ranking_scores(self.pos, self.fpos, model.epoch, 'VALID')

    def et_ranking_scores(self, pos, fpos, epoch, txt):
        tpos = [p for k in pos.keys() for p in pos[k]['type']]
        tfpos = [p for k in fpos.keys() for p in fpos[k]['type']]
        fmrr = self._print_pos(
            np.array(tpos),
            np.array(tfpos),
            epoch, txt)
        return fmrr

    def positions(self, mdl):
        pos = {}    # Raw Positions
        fpos = {}   # Filtered Positions

        for e, ts in self.idx.items():

            ppos = {'type': []}
            pfpos = {'type': []}

            for t in ts:

                scores_t = mdl._scores_et(e).flatten()
                sortidx_t = np.argsort(np.argsort(scores_t))
                ppos['type'].append(sortidx_t[t] + 1)

                rm_idx = self.tt[e]
                rm_idx = [i for i in rm_idx if i != t]
                scores_t[rm_idx] = np.Inf
                sortidx_t = np.argsort(np.argsort(scores_t))
                pfpos['type'].append(sortidx_t[t] + 1)


            pos[e] = ppos
            fpos[e] = pfpos

        return pos, fpos

class Type_Evaluator_trt(Type_Evaluator):
    def __init__(self, xs, true_tuples, train_triplets, logger=None):
        super().__init__(xs, true_tuples, logger)
        self.tt_h_l, self.tt_t_l = self.convert_triple_into_dict(
            train_triplets)  # {head: {tail: [relation1, relation2, ...]}}

    def convert_triple_into_dict(self, triplet):
        h_l_dict = {}
        t_l_dict = {}
        for head, label, tail in triplet:
            if head in h_l_dict.keys():
                if label in h_l_dict[head].keys():
                    h_l_dict[head][label].append(tail)
                else:
                    h_l_dict[head][label] = [tail]
            else:
                h_l_dict[head] = {label: [tail]}

            if tail in t_l_dict.keys():
                if label in t_l_dict[tail].keys():
                    t_l_dict[tail][label].append(head)
                else:
                    t_l_dict[tail][label] = [head]
            else:
                t_l_dict[tail] = {label: [head]}

        return h_l_dict, t_l_dict

    def positions(self, mdl):
        pos = {}    # Raw Positions
        fpos = {}   # Filtered Positions
        count = 0
        for e, ts in self.idx.items():

            ppos = {'type': []}
            pfpos = {'type': []}

            for t in ts:

                score1, score2 = self._score_trt(mdl, e)
                scores_t = mdl._scores_et(e).flatten()
                scores_t = scores_t + 0.5*score1 + 0.5*score2
                # scores_t = 0.5 * score1 + 0.5 * score2
                sortidx_t = np.argsort(np.argsort(scores_t))
                ppos['type'].append(sortidx_t[t] + 1)

                rm_idx = self.tt[e]
                rm_idx = [i for i in rm_idx if i != t]
                scores_t[rm_idx] = np.Inf
                sortidx_t = np.argsort(np.argsort(scores_t))
                pfpos['type'].append(sortidx_t[t] + 1)
                count += 1
                if count % 200 == 0:
                    self.logger.info(f"Processing: {100*count/self.tot:.2f}%")

            pos[e] = ppos
            fpos[e] = pfpos

        return pos, fpos

    def _score_trt(self, mdl, e):
        # e, r, e2
        scores1 = []
        scores2 = []
        if e in self.tt_h_l.keys():
            for r, related_e in self.tt_h_l[e].items():
                for entity in related_e:
                    if entity in self.tt.keys():
                        types = self.tt[entity]
                        for et in types:
                            scores1.append(mdl._scores_trt(et, r, change_head=True))


        if e in self.tt_t_l.keys():
            for r, related_e in self.tt_t_l[e].items():
                for entity in related_e:
                    if entity in self.tt.keys():
                        types = self.tt[entity]
                        for et in types:
                            scores2.append(mdl._scores_trt(et, r, change_head=False))

        if scores1:
            score1 = np.nanmean(np.vstack(scores1), axis=0)
        else:
            score1 = np.zeros(mdl.typeMat.shape[0])

        if scores2:
            score2 = np.nanmean(np.vstack(scores2), axis=0)
        else:
            score2 = np.zeros(mdl.typeMat.shape[0])

        return score1, score2

# """only use the most frequent related entities"""
# class Type_Evaluator_trt_2(Type_Evaluator_trt):
#     def _score_trt(self, mdl, e):
#         # e, r, e2
#         scores1 = []
#         scores2 = []
#         if e in self.tt_h_l.keys():
#             for r, related_e in self.tt_h_l[e].items():
#                 ranks_e = Counter(related_e)
#                 entity = ranks_e.most_common(1)[0][0]
#                 if entity in self.tt.keys():
#                     types = self.tt[entity]
#                     for et in types:
#                         scores1.append(mdl._scores_trt(et, r, change_head=True))


#         if e in self.tt_t_l.keys():
#             for r, related_e in self.tt_t_l[e].items():
#                 ranks_e = Counter(related_e)
#                 entity = ranks_e.most_common(1)[0][0]
#                 if entity in self.tt.keys():
#                     types = self.tt[entity]
#                     for et in types:
#                         scores2.append(mdl._scores_trt(et, r, change_head=False))

#         if scores1:
#             score1 = np.vstack(scores1).mean(axis=0)
#         else:
#             score1 = np.zeros(mdl.typeMat.shape[0])

#         if scores2:
#             score2 = np.vstack(scores2).mean(axis=0)
#         else:
#             score2 = np.zeros(mdl.typeMat.shape[0])

#         return score1, score2


# """only use trt for high degree entitities"""
# class Type_Evaluator_trt_3(Type_Evaluator):

#     def __init__(self, xs, true_tuples, train_triplets, train_tuples, logger=None):
#         super().__init__(xs, true_tuples, logger)
#         self.tt_h_l, self.tt_t_l = self.convert_triple_into_dict(
#             train_triplets)  # {head: {tail: [relation1, relation2, ...]}}
#         def count_types(e2t):
#             d = defaultdict(set)
#             for e, et in e2t:
#                 d[e].add(et)
#             return {k: len(v) for k, v in d.items()}

#         self.tot_types = count_types(train_tuples)
#         # 出度
#         self.chudu_dict = defaultdict(int)
#         # 入度
#         self.rudu_dict = defaultdict(int)
#         # 双向
#         self.shuangxiang_dict = defaultdict(int)
#         for e1, r, e2 in train_triplets:
#             self.chudu_dict[e1] += 1
#             self.rudu_dict[e2] += 1
#             self.shuangxiang_dict[e1] += 1
#             self.shuangxiang_dict[e2] += 1


#     def convert_triple_into_dict(self, triplet):
#         h_l_dict = {}
#         t_l_dict = {}
#         for head, label, tail in triplet:
#             if head in h_l_dict.keys():
#                 if label in h_l_dict[head].keys():
#                     h_l_dict[head][label].append(tail)
#                 else:
#                     h_l_dict[head][label] = [tail]
#             else:
#                 h_l_dict[head] = {label: [tail]}

#             if tail in t_l_dict.keys():
#                 if label in t_l_dict[tail].keys():
#                     t_l_dict[tail][label].append(head)
#                 else:
#                     t_l_dict[tail][label] = [head]
#             else:
#                 t_l_dict[tail] = {label: [head]}

#         return h_l_dict, t_l_dict

#     def positions(self, mdl):
#         pos = {}    # Raw Positions
#         fpos = {}   # Filtered Positions
#         count = 0

#         for e, ts in self.idx.items():

#             ppos = {'type': []}
#             pfpos = {'type': []}

#             for t in ts:
#                 scores_t = mdl._scores_et(e).flatten()
#                 if self.chudu_dict[e] * self.tot_types[e] > 700:
#                     score1, score2 = self._score_trt(mdl, e)
#                     scores_t = scores_t + 0.5*score1 + 0.5*score2

#                 # scores_t = 0.5 * score1 + 0.5 * score2
#                 sortidx_t = np.argsort(np.argsort(scores_t))
#                 ppos['type'].append(sortidx_t[t] + 1)

#                 rm_idx = self.tt[e]
#                 rm_idx = [i for i in rm_idx if i != t]
#                 scores_t[rm_idx] = np.Inf
#                 sortidx_t = np.argsort(np.argsort(scores_t))
#                 pfpos['type'].append(sortidx_t[t] + 1)
#                 count += 1
#                 if count % 200 == 0:
#                     self.logger.info(f"Processing: {100*count/self.tot:.2f}%")

#             pos[e] = ppos
#             fpos[e] = pfpos

#         return pos, fpos

#     def _score_trt(self, mdl, e):
#         # e, r, e2
#         scores1 = []
#         scores2 = []
#         if e in self.tt_h_l.keys():
#             for r, related_e in self.tt_h_l[e].items():
#                 for entity in related_e:
#                     if entity in self.tt.keys():
#                         types = self.tt[entity]
#                         for et in types:
#                             scores1.append(mdl._scores_trt(et, r, change_head=True))


#         if e in self.tt_t_l.keys():
#             for r, related_e in self.tt_t_l[e].items():
#                 for entity in related_e:
#                     if entity in self.tt.keys():
#                         types = self.tt[entity]
#                         for et in types:
#                             scores2.append(mdl._scores_trt(et, r, change_head=False))

#         if scores1:
#             score1 = np.vstack(scores1).mean(axis=0)
#         else:
#             score1 = np.zeros(mdl.typeMat.shape[0])

#         if scores2:
#             score2 = np.vstack(scores2).mean(axis=0)
#         else:
#             score2 = np.zeros(mdl.typeMat.shape[0])

#         return score1, score2


# """rank based blending"""
# class Type_Evaluator_trt2(Type_Evaluator_trt):
#     def positions(self, mdl):
#         pos = {}    # Raw Positions
#         fpos = {}   # Filtered Positions
#         count = 0
#         for e, ts in self.idx.items():

#             ppos = {'type': []}
#             pfpos = {'type': []}

#             for t in ts:

#                 scores_t = mdl._scores_et(e).flatten()
#                 score1, score2 = self._score_trt(mdl, e)
#                 scores_t = rank(scores_t)
#                 score2_t2 = rank(score1+score2)
#                 scores_t = 0.5*scores_t + 0.5*score2_t2
#                 sortidx_t = rank(scores_t)
#                 ppos['type'].append(sortidx_t[t] + 1)

#                 rm_idx = self.tt[e]
#                 rm_idx = [i for i in rm_idx if i != t]
#                 scores_t[rm_idx] = np.Inf
#                 sortidx_t = np.argsort(np.argsort(scores_t))
#                 pfpos['type'].append(sortidx_t[t] + 1)
#                 count += 1
#                 if count % 200 == 0:
#                     self.logger.info(f"Processing: {100*count/self.tot:.2f}%")

#             pos[e] = ppos
#             fpos[e] = pfpos

#         return pos, fpos

# class Type_Evaluator_trt2_2(Type_Evaluator_trt2):
#     def _score_trt(self, mdl, e):
#         # e, r, e2
#         scores1 = []
#         scores2 = []
#         if e in self.tt_h_l.keys():
#             for r, related_e in self.tt_h_l[e].items():
#                 ranks_e = Counter(related_e)
#                 entity = ranks_e.most_common(1)[0][0]
#                 if entity in self.tt.keys():
#                     types = self.tt[entity]
#                     for et in types:
#                         scores1.append(mdl._scores_trt(et, r, change_head=True))


#         if e in self.tt_t_l.keys():
#             for r, related_e in self.tt_t_l[e].items():
#                 ranks_e = Counter(related_e)
#                 entity = ranks_e.most_common(1)[0][0]
#                 if entity in self.tt.keys():
#                     types = self.tt[entity]
#                     for et in types:
#                         scores2.append(mdl._scores_trt(et, r, change_head=False))

#         if scores1:
#             score1 = np.vstack(scores1).mean(axis=0)
#         else:
#             score1 = np.zeros(mdl.typeMat.shape[0])

#         if scores2:
#             score2 = np.vstack(scores2).mean(axis=0)
#         else:
#             score2 = np.zeros(mdl.typeMat.shape[0])

#         return score1, score2

class Classification_Evaluator(Type_Evaluator_trt):
    def __init__(self, validate_set, test_set, true_tuples, train_triplets, model, logger=None,
                 save_data=False, load_data=False, load_path=None, save_path=None):
        super().__init__(test_set, true_tuples, train_triplets, logger)
        self.validate = validate_set
        self.test = test_set
        self.model = model
        self.name = model.name
        self.type_size = self.model.get_type_size()
        self.load_data = load_data
        self.load_path = load_path
        self.save_data = save_data
        self.save_path = save_path
        self.logger.info(f"Start to handel {self.name} model")

    def sample(self, p_sample):
        n_sample = []
        for e, _ in p_sample:
            p_et = np.random.randint(0, self.type_size)

            # while p_et in self.tt[e]:
            #     p_et = np.random.randint(0, self.type_size)
            n_sample.append((e, p_et))
        return n_sample

    def __call__(self):
        saving_data = {}
        # validate
        if self.load_data:
            results1 = self.load(self.load_path, type='valid')
        else:
            results1 = self.evaluate(self.validate)

        best_dis, best_score = self.get_optimal_distance(results1)
        if self.save_data:
            saving_data = {'valid': results1}

        self.logger.info("%s: The best distance/accuracy in VALID data is %f/%f"%(self.name, best_dis, best_score))


        # test
        if self.load_data:
            results = self.load(self.load_path, type='test')

        else:
            results = self.evaluate(self.test)

        if self.save_data:
            saving_data['test'] = results
            self.save(saving_data, path=self.save_path)

        acc = self.cal_accuracy(results, best_dis)
        self.logger.info("%s: The accuracy in TEST is %f"%(self.name, acc))
        return results

    def save(self, data, path):
        with open(path, "wb") as f:
            pickle.dump(data, f)

    def load(self, path, type):
        with open(path, "rb") as f:
            return pickle.load(f)[type]

    def normalize(self, data):
        max_ = max([d for d, _ in data])
        return [(1-d/max_, flag) for d, flag in data]

    def get_optimal_distance(self, results):
        distances = []
        y_true = []
        for d, pos in results:
            distances.append(d)
            y_true.append(pos)

        distances = np.array(distances)
        y_true = np.array(y_true)
        best_accuracy = 0
        best_distance = distances[0]
        for dis in distances:
            y_pred = np.ones_like(y_true)
            y_pred[distances>=dis] = -1
            acc = accuracy_score(y_true, y_pred)
            if acc > best_accuracy:
                best_accuracy = acc
                best_distance = dis

        return best_distance, best_accuracy

    def cal_accuracy(self, results, dis):
        distances = []
        y_true = []
        for d, pos in results:
            distances.append(d)
            y_true.append(pos)

        distances = np.array(distances)
        y_true = np.array(y_true)
        y_pred = np.ones_like(y_true)
        y_pred[distances >= dis] = -1
        acc = accuracy_score(y_true, y_pred)
        return acc

    def evaluate(self, p_sample):
        n_sample = self.sample(p_sample)
        p_sample = [(k, 1) for k in p_sample]
        n_sample = [(k, -1) for k in n_sample]
        tot_sample = p_sample + n_sample
        tot_scores = []
        trt_scores = []
        count = 0
        for m, flag in tot_sample:
            e, et = m
            scores_t = self.model._scores_et(e, et).flatten()[0]
            score1, score2 = self._score_trt(self.model, e, et)
            trt_score = 0.5*score1[0] + 0.5*score2[0]
            avg_score = scores_t + trt_score
            trt_scores.append((trt_score, flag))
            tot_scores.append((avg_score, flag))
            count += 1
            # if count % 1000 == 0:
            #     print(f"已经完成{count/len(tot_sample)*100:.2f}%")

        return tot_scores


    def _score_trt(self, mdl, e, et2):
        # e, r, e2
        scores1 = []
        scores2 = []
        if e in self.tt_h_l.keys():
            for r, related_e in self.tt_h_l[e].items():
                for entity in related_e:
                    if entity in self.tt.keys():
                        types = self.tt[entity]
                        for et in types:
                            scores1.append(mdl._scores_trt(et, r, change_head=True, et2=et2))


        if e in self.tt_t_l.keys():
            for r, related_e in self.tt_t_l[e].items():
                for entity in related_e:
                    if entity in self.tt.keys():
                        types = self.tt[entity]
                        for et in types:
                            scores2.append(mdl._scores_trt(et, r, change_head=False, et2=et2))

        if scores1:
            score1 = np.vstack(scores1).mean(axis=0)
        else:
            score1 = np.zeros(mdl.typeMat.shape[0])

        if scores2:
            score2 = np.vstack(scores2).mean(axis=0)
        else:
            score2 = np.zeros(mdl.typeMat.shape[0])

        return score1, score2


class Classification_Evaluator_E2T(Classification_Evaluator):
    def evaluate(self, p_sample):
        n_sample = self.sample(p_sample)
        p_sample = [(k, 1) for k in p_sample]
        n_sample = [(k, -1) for k in n_sample]
        tot_sample = p_sample + n_sample
        tot_scores = []
        count = 0
        for m, flag in tot_sample:
            e, et = m
            scores_t = self.model._scores_et(e, et).flatten()[0]
            avg_score = scores_t
            tot_scores.append((avg_score, flag))
            count += 1
            # if count % 1000 == 0:
            #     print(f"已经完成{count/len(tot_sample)*100:.2f}%")

        return tot_scores

# class Classification_Evaluator_TRT(Classification_Evaluator):
#     name = "TRT Model"
#     def evaluate(self, p_sample):
#         n_sample = self.sample(p_sample)
#         p_sample = [(k, 1) for k in p_sample]
#         n_sample = [(k, -1) for k in n_sample]
#         tot_sample = p_sample + n_sample
#         tot_scores = []
#         count = 0
#         for m, flag in tot_sample:
#             e, et = m
#             score1, score2 = self._score_trt(self.model, e, et)
#             avg_score = 0.5 * score1[0] + 0.5 * score2[0]
#             tot_scores.append((avg_score, flag))
#             count += 1
#             if count % 1000 == 0:
#                 print(f"已经完成{count/len(tot_sample)*100:.2f}%")

#         return tot_scores


def rank(arr):
    return np.argsort(np.argsort(arr))






