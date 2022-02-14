import torch
def hierarchical_softmax(inp, tree):
    x1 = torch.multiply(tree.decision_matrix, input)
    x1 = tree.base + x1
    x1 = torch.log(x1)                                   #extra step #1
    x1 = torch.sum(x1, axis=1)               #reduce_prod is replaced by reduce_sum
    return torch.exp(x1)                            #extra step #2