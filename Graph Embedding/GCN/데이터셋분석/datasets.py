import networkx as nx
import numpy as np
import os
import pickle
import torch
import numpy as np
import scipy.sparse as sp
import torch


def encode_onehot_KIHOON(labels):
    classes = set(labels) #중복되지 않은 원소 집합
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in #객체 고유값 메모리에있는지 체크
                    enumerate(classes)} #인덱스와 원소를 동시에 접근하면서 루프돌림
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot


def load_data_KIHOON(dataset): #citeseer / cora / pubmed  택1
    #데이터 얻어오는 함수
    data_path = 'C:/Users/LeeKihoon/PycharmProjects/KKK/Graph Embedding/planetoid-master/data'
    suffixs = ['x', 'y', 'allx', 'ally', 'tx', 'ty', 'graph'] #접미사(마지막에 붙는것들)
    objects = [] #객체
    for suffix in suffixs:
        file = data_path + '/ind.%s.%s'%(dataset, suffix)  #파일 경로 설정
        objects.append(pickle.load(open(file, 'rb'), encoding='latin1')) #리스트에 추가함. pickle은 텍스트데이터X 바이너리파일로 저장.
    x, y, allx, ally, tx, ty, graph = objects
    x, allx, tx = x.toarray(), allx.toarray(), tx.toarray()

    # 테스트 인덱스들
    test_index_file = os.path.join(data_path, 'ind.%s.test.index'%dataset)
    with open(test_index_file, 'r') as f:
        lines = f.readlines()
    indices = [int(line.strip()) for line in lines]
    min_index, max_index = min(indices), max(indices)

    # 테스트 인덱스들 처리 및 모든 데이터 결합
    tx_extend = np.zeros((max_index - min_index + 1, tx.shape[1]))
    features = np.vstack([allx, tx_extend])
    features[indices] = tx
    ty_extend = np.zeros((max_index - min_index + 1, ty.shape[1]))
    labels = np.vstack([ally, ty_extend])
    labels[indices] = ty

    # 인접 매트릭스 가져오기
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph)).toarray()

    idx_train = range(len(y))
    idx_val = range(len(y), len(y) + 500)
    idx_test = indices

    train_mask = sample_mask(idx_train, labels.shape[0])
    val_mask = sample_mask(idx_val, labels.shape[0])
    test_mask = sample_mask(idx_test, labels.shape[0])
    zeros = np.zeros(labels.shape)
    y_train = zeros.copy()
    y_val = zeros.copy()
    y_test = zeros.copy()
    y_train[train_mask, :] = labels[train_mask, :]
    y_val[val_mask, :] = labels[val_mask, :]
    y_test[test_mask, :] = labels[test_mask, :]
    features = torch.from_numpy(process_features(features))
    y_train, y_val, y_test, train_mask, val_mask, test_mask = \
        torch.from_numpy(y_train), torch.from_numpy(y_val), torch.from_numpy(y_test), \
        torch.from_numpy(train_mask), torch.from_numpy(val_mask), torch.from_numpy(test_mask)

    return adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask


def process_features(features):
    row_sum_diag = np.sum(features, axis=1)
    row_sum_diag_inv = np.power(row_sum_diag, -1)
    row_sum_diag_inv[np.isinf(row_sum_diag_inv)] = 0.
    row_sum_inv = np.diag(row_sum_diag_inv)
    return np.dot(row_sum_inv, features)


def sample_mask(idx, l):
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)


def load_data(dataset):
    ## get data
    data_path = '/Graph Embedding/planetoid-master/data'
    suffixs = ['x', 'y', 'allx', 'ally', 'tx', 'ty', 'graph']
    objects = []
    for suffix in suffixs:
        file = os.path.join(data_path, 'ind.%s.%s'%(dataset, suffix))
        objects.append(pickle.load(open(file, 'rb'), encoding='latin1'))
    x, y, allx, ally, tx, ty, graph = objects
    x, allx, tx = x.toarray(), allx.toarray(), tx.toarray()

    # test indices
    test_index_file = os.path.join(data_path, 'ind.%s.test.index'%dataset)
    with open(test_index_file, 'r') as f:
        lines = f.readlines()
    indices = [int(line.strip()) for line in lines]
    min_index, max_index = min(indices), max(indices)

    # preprocess test indices and combine all data
    tx_extend = np.zeros((max_index - min_index + 1, tx.shape[1]))
    features = np.vstack([allx, tx_extend])
    features[indices] = tx
    ty_extend = np.zeros((max_index - min_index + 1, ty.shape[1]))
    labels = np.vstack([ally, ty_extend])
    labels[indices] = ty

    # get adjacency matrix
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph)).toarray()

    idx_train = range(len(y))
    idx_val = range(len(y), len(y) + 500)
    idx_test = indices

    train_mask = sample_mask(idx_train, labels.shape[0])
    val_mask = sample_mask(idx_val, labels.shape[0])
    test_mask = sample_mask(idx_test, labels.shape[0])
    zeros = np.zeros(labels.shape)
    y_train = zeros.copy()
    y_val = zeros.copy()
    y_test = zeros.copy()
    y_train[train_mask, :] = labels[train_mask, :]
    y_val[val_mask, :] = labels[val_mask, :]
    y_test[test_mask, :] = labels[test_mask, :]
    features = torch.from_numpy(process_features(features))
    y_train, y_val, y_test, train_mask, val_mask, test_mask = \
        torch.from_numpy(y_train), torch.from_numpy(y_val), torch.from_numpy(y_test), \
        torch.from_numpy(train_mask), torch.from_numpy(val_mask), torch.from_numpy(test_mask)

    return adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask
