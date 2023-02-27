import operator
import random
import time
from functools import cmp_to_key
from tqdm import tqdm

from algorithms.basic_mondrian.models.numrange import NumRange
from algorithms.basic_mondrian.utils.utility import (cmp_str, get_num_list_from_str,
                                          qid_to_key)

__DEBUG = False
ATT_TREES = []
LEN_DATA = 0
QI_LEN = 0
QI_RANGE = []
IS_CAT = []
LCA_CACHE = []
NCP_CACHE = {}


class Cluster(object):

    def __init__(self, member, gen_result, information_loss=0.0):
        self.information_loss = information_loss
        self.member = member
        self.gen_result = gen_result[:]
        self.center = gen_result[:]
        for i in range(QI_LEN):
            if IS_CAT[i] is False:
                self.center[i] = str(sum([float(t[i]) for t in self.member]) * 1.0 / len(self.member))

    def add_record(self, record):
        self.member.append(record)
        self.update_gen_result(record, record)

    def update_cluster(self):
        self.gen_result = cluster_generalization(self.member)
        for i in range(QI_LEN):
            if IS_CAT[i]:
                self.center[i] = self.gen_result[i]
            else:
                self.center[i] = str(sum([float(t[i]) for t in self.member]) * 1.0 / len(self.member))
        self.information_loss = len(self.member) * NCP(self.gen_result)

    def update_gen_result(self, merge_gen_result, center, num=1):
        self.gen_result = generalization(self.gen_result, merge_gen_result)
        current_len = len(self.member)
        for i in range(QI_LEN):
            if IS_CAT[i]:
                self.center[i] = self.gen_result[i]
            else:
                self.center[i] = str((float(self.center[i]) * (current_len - num) +
                                      float(center[i]) * num) / current_len)
        self.information_loss = len(self.member) * NCP(self.gen_result)

    def add_same_record(self, record):
        self.member.append(record)

    def merge_cluster(self, cluster):
        self.member.extend(cluster.member)
        self.update_gen_result(cluster.gen_result, cluster.center, len(cluster))

    def __getitem__(self, item):
        return self.gen_result[item]

    def __len__(self):
        return len(self.member)

    def __str__(self):
        return str(self.gen_result)


def r_distance(source, target):
    source_gen = source
    target_gen = target
    source_len = 1
    target_len = 1
    # check if target is Cluster
    if isinstance(target, Cluster):
        target_gen = target.gen_result
        target_len = len(target)
    # check if souce is Cluster
    if isinstance(source, Cluster):
        source_gen = source.gen_result
        source_len = len(source)
    if source_gen == target_gen:
        return 0
    gen = generalization(source_gen, target_gen)
    # len should be taken into account
    distance = (source_len + target_len) * NCP(gen)
    return distance


def diff_distance(record, cluster):
    gen_after = generalization(record, cluster.gen_result)
    return NCP(gen_after) * (len(cluster) + 1) - cluster.information_loss


def NCP(record):

    ncp = 0.0
    list_key = qid_to_key(record)
    try:
        return NCP_CACHE[list_key]
    except KeyError:
        pass
    for i in range(QI_LEN):
        # if leaf_num of numerator is 1, then NCP is 0
        width = 0.0
        if IS_CAT[i] is False:
            try:
                float(record[i])
            except ValueError:
                temp = record[i].split(',')
                width = float(temp[1]) - float(temp[0])
        else:
            width = len(ATT_TREES[i][record[i]]) * 1.0
        width /= QI_RANGE[i]
        ncp += width
    NCP_CACHE[list_key] = ncp
    return ncp


def get_LCA(index, item1, item2):
    # get parent list from
    if item1 == item2:
        return item1
    try:
        return LCA_CACHE[index][item1 + item2]
    except KeyError:
        pass
    parent1 = ATT_TREES[index][item1].parent[:]
    parent2 = ATT_TREES[index][item2].parent[:]
    parent1.insert(0, ATT_TREES[index][item1])
    parent2.insert(0, ATT_TREES[index][item2])
    min_len = min(len(parent1), len(parent2))
    last_LCA = parent1[-1]
    for i in range(1, min_len + 1):
        if parent1[-i].value == parent2[-i].value:
            last_LCA = parent1[-i]
        else:
            break
    LCA_CACHE[index][item1 + item2] = last_LCA.value
    return last_LCA.value


def generalization(record1, record2):
    gen = []
    for i in range(QI_LEN):
        if IS_CAT[i] is False:
            split_number = []
            split_number.extend(get_num_list_from_str(record1[i]))
            split_number.extend(get_num_list_from_str(record2[i]))
            split_number = list(set(split_number))
            if len(split_number) == 1:
                gen.append(split_number[0])
            else:
                split_number.sort(key=cmp_to_key(cmp_str))
                gen.append(split_number[0] + ',' + split_number[-1])
        else:
            gen.append(get_LCA(i, record1[i], record2[i]))
    return gen


def cluster_generalization(records):
    len_r = len(records)
    gen = records[0]
    for i in range(1, len_r):
        gen = generalization(gen, records[i])
    return gen


def find_best_knn(index, k, data):
    dist_dict = {}
    record = data[index]
    # add random seed to cluster
    for i, t in enumerate(data):
        if i == index:
            continue
        dist = r_distance(record, t)
        dist_dict[i] = dist
    sorted_dict = sorted(dist_dict.items(), key=operator.itemgetter(1))
    knn = sorted_dict[:k - 1]
    knn.append((index, 0))
    record_index = [t[0] for t in knn]
    elements = [data[t[0]] for t in knn]
    gen = cluster_generalization(elements)
    cluster = Cluster(elements, gen, k * NCP(gen))
    # delete multiple elements from data according to knn index list
    return cluster, record_index


def find_best_cluster_iloss(record, clusters):
    min_distance = 1000000000000
    min_index = 0
    best_cluster = clusters[0]
    for i, t in enumerate(clusters):
        distance = r_distance(record, t.gen_result)
        if distance < min_distance:
            min_distance = distance
            min_index = i
            best_cluster = t
    return min_index


def find_best_cluster_iloss_increase(record, clusters):
    min_diff = 1000000000000
    min_index = 0
    best_cluster = clusters[0]
    for i, t in enumerate(clusters):
        IF_diff = diff_distance(record, t)
        if IF_diff < min_diff:
            min_distance = IF_diff
            min_index = i
            best_cluster = t
    return min_index


def find_furthest_record(record, data):
    max_distance = 0
    max_index = -1
    for index in range(len(data)):
        current_distance = r_distance(record, data[index])
        if current_distance >= max_distance:
            max_distance = current_distance
            max_index = index
    return max_index


def find_best_record_iloss_increase(cluster, data):
    min_diff = 1000000000000
    min_index = 0
    for index, record in enumerate(data):
        IF_diff = diff_distance(record, cluster)
        if IF_diff < min_diff:
            min_diff = IF_diff
            min_index = index
    return min_index


def clustering_knn(data, k=25):
    clusters = []
    while len(data) >= k:
        index = random.randrange(len(data))
        cluster, record_index = find_best_knn(index, k, data)
        data = [t for i, t in enumerate(data[:]) if i not in set(record_index)]
        clusters.append(cluster)
    while len(data) > 0:
        t = data.pop()
        cluster_index = find_best_cluster_iloss(t, clusters)
        clusters[cluster_index].add_record(t)
    return clusters


def clustering_kmember(data, k=25):
    clusters = []
    r_pos = random.randrange(len(data))
    r_i = data[r_pos]
    while len(data) >= k:
        r_pos = find_furthest_record(r_i, data)
        r_i = data.pop(r_pos)
        cluster = Cluster([r_i], r_i)
        while len(cluster) < k:
            r_pos = find_best_record_iloss_increase(cluster, data)
            r_j = data.pop(r_pos)
            cluster.add_record(r_j)
        clusters.append(cluster)
    # residual assignment
    while len(data) > 0:
        t = data.pop()
        cluster_index = find_best_cluster_iloss_increase(t, clusters)
        clusters[cluster_index].add_record(t)
    return clusters


def adjust_cluster(cluster, residual, k):
    center = cluster.center
    dist_dict = {}
    for i, t in enumerate(cluster.member):
        dist = r_distance(center, t)
        dist_dict[i] = dist
    sorted_dict = sorted(dist_dict.iteritems(), key=operator.itemgetter(1))
    need_adjust_index = [t[0] for t in sorted_dict[k:]]
    need_adjust = [cluster.member[t] for t in need_adjust_index]
    residual.extend(need_adjust)
    cluster.member = [t for i, t in enumerate(cluster.member)
                      if i not in set(need_adjust_index)]
    cluster.update_cluster()


def clustering_oka(data, k=25):
    clusters = []
    can_clusters = []
    less_clusters = []
    # randomly choose seed and find k-1 nearest records to form cluster with size k
    seed_index = random.sample(range(len(data)), len(data) / k)
    for index in seed_index:
        record = data[index]
        can_clusters.append(Cluster([record], record))
    data = [t for i, t in enumerate(data[:]) if i not in set(seed_index)]
    while len(data) > 0:
        record = data.pop()
        index = find_best_cluster_iloss(record, can_clusters)
        can_clusters[index].add_record(record)
    residual = []
    for cluster in can_clusters:
        if len(cluster) < k:
            less_clusters.append(cluster)
        else:
            if len(cluster) > k:
                adjust_cluster(cluster, residual, k)
            clusters.append(cluster)
    while len(residual) > 0:
        record = residual.pop()
        if len(less_clusters) > 0:
            index = find_best_cluster_iloss(record, less_clusters)
            less_clusters[index].add_record(record)
            if less_clusters[index] >= k:
                clusters.append(less_clusters.pop(index))
        else:
            index = find_best_cluster_iloss(record, clusters)
            clusters[index].add_record(record)
    return clusters


def init(att_trees, data, SA_num, QI_num=-1):
    global ATT_TREES, DATA_BACKUP, LEN_DATA, QI_RANGE, IS_CAT, QI_LEN, LCA_CACHE, NCP_CACHE, SA_INDEX
    SA_INDEX = SA_num
    ATT_TREES = att_trees
    QI_RANGE = []
    IS_CAT = []
    LEN_DATA = len(data)
    LCA_CACHE = []
    NCP_CACHE = {}
    if QI_num <= 0:
        QI_LEN = len(data[0]) - 1
    else:
        QI_LEN = QI_num
    for i in range(QI_LEN):
        LCA_CACHE.append(dict())
        if isinstance(ATT_TREES[i], NumRange):
            IS_CAT.append(False)
            QI_RANGE.append(ATT_TREES[i].range)
        else:
            IS_CAT.append(True)
            QI_RANGE.append(len(ATT_TREES[i]['*']))


def clustering_based_k_anon(att_trees, data, k, QI_num, SA_num, type_alg):
    init(att_trees, data, SA_num, QI_num)
    result = []
    start_time = time.time()
    if type_alg == 'knn':
        print("Begin to KNN Cluster based on NCP")
        clusters = clustering_knn(data, k)
    elif type_alg == 'kmember':
        print("Begin to K-Member Cluster based on NCP")
        clusters = clustering_kmember(data, k)
    elif type_alg == 'oka':
        print("Begin to OKA Cluster based on NCP")
        clusters = clustering_oka(data, k)
    else:
        print("Please choose merge algorithm types")
        print("knn | kmember")
        return (0, (0, 0))
    rtime = float(time.time() - start_time)
    for cluster in tqdm(clusters):
        final_result = []
        for i in range(len(cluster)):
            # Custom! For non QI Values
            tmp = []
            for s in range(len(cluster.member[i]) - len(SA_INDEX), len(cluster.member[i])):
                tmp += [cluster.member[i][s]]
            final_result.append(cluster.gen_result + tmp)
        result.extend(final_result)

    return (result, rtime)
