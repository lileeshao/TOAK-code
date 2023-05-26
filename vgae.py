import os
import time
import dgl
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
import torch
import torch.nn.functional as F
from dgl.nn.pytorch import GraphConv
import torch.nn as nn
import networkx as nx

class VGAEModel(nn.Module):
    def __init__(self, in_dim, hidden1_dim, hidden2_dim):
        super(VGAEModel, self).__init__()
        self.in_dim = in_dim
        self.hidden1_dim = hidden1_dim
        self.hidden2_dim = hidden2_dim

        layers = [GraphConv(self.in_dim, self.hidden1_dim, activation=F.relu, allow_zero_in_degree=True),
                  GraphConv(self.hidden1_dim, self.hidden2_dim, activation=lambda x: x, allow_zero_in_degree=True),
                  GraphConv(self.hidden1_dim, self.hidden2_dim, activation=lambda x: x, allow_zero_in_degree=True)]
        self.layers = nn.ModuleList(layers)

    def encoder(self, g, features, device):
        h = self.layers[0](g, features)
        self.mean = self.layers[1](g, h)
        self.log_std = self.layers[2](g, h)
        gaussian_noise = torch.randn(features.size(0), self.hidden2_dim).to(device)
        sampled_z = self.mean + gaussian_noise * torch.exp(self.log_std).to(device)
        return sampled_z

    def decoder(self, z):
        adj_rec = torch.sigmoid(torch.matmul(z, z.t()))
        return adj_rec

    def forward(self, g, features, device):
        z = self.encoder(g, features, device)
        adj_rec = self.decoder(z)
        return adj_rec,z

def mask_test_edges_dgl(graph, adj):
    src, dst = graph.edges()
    edges_all = torch.stack([src, dst], dim=0)
    edges_all = edges_all.t().cpu().numpy()
    num_test = int(np.floor(edges_all.shape[0] / 10.))
    num_val = int(np.floor(edges_all.shape[0] / 20.))

    all_edge_idx = list(range(edges_all.shape[0]))
    np.random.shuffle(all_edge_idx)
    val_edge_idx = all_edge_idx[:num_val]
    test_edge_idx = all_edge_idx[num_val:(num_val + num_test)]
    train_edge_idx = all_edge_idx 
    test_edges = edges_all[test_edge_idx]
    val_edges = edges_all[val_edge_idx]
    train_edges = edges_all 

    def ismember(a, b, tol=5):
        rows_close = np.all(np.round(a - b[:, None], tol) == 0, axis=-1)
        return np.any(rows_close)

    test_edges_false = []
    while len(test_edges_false) < len(test_edges):
        idx_i = np.random.randint(0, adj.shape[0])
        idx_j = np.random.randint(0, adj.shape[0])
        if idx_i == idx_j:
            continue
        if ismember([idx_i, idx_j], edges_all):
            continue
        if test_edges_false:
            if ismember([idx_j, idx_i], np.array(test_edges_false)):
                continue
            if ismember([idx_i, idx_j], np.array(test_edges_false)):
                continue
        test_edges_false.append([idx_i, idx_j])

    val_edges_false = []
    while len(val_edges_false) < len(val_edges):
        idx_i = np.random.randint(0, adj.shape[0])
        idx_j = np.random.randint(0, adj.shape[0])
        if idx_i == idx_j:
            continue
        if ismember([idx_i, idx_j], train_edges):
            continue
        if ismember([idx_j, idx_i], train_edges):
            continue
        #if ismember([idx_i, idx_j], val_edges):
            #continue
        #if ismember([idx_j, idx_i], val_edges):
            #continue
        if val_edges_false:
            if ismember([idx_j, idx_i], np.array(val_edges_false)):
                continue
            if ismember([idx_i, idx_j], np.array(val_edges_false)):
                continue
        val_edges_false.append([idx_i, idx_j])

    #assert ~ismember(test_edges_false, edges_all)
    #assert ~ismember(val_edges_false, edges_all)
    #assert ~ismember(val_edges, train_edges)
    #assert ~ismember(test_edges, train_edges)
    #assert ~ismember(val_edges, test_edges)

    # NOTE: these edge lists only contain single direction of edge!
    return train_edge_idx, val_edges, val_edges_false, test_edges, test_edges_false

def compute_loss_para(adj, device):
    pos_weight = ((adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum())
    norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)
    weight_mask = adj.view(-1) == 1
    weight_tensor = torch.ones(weight_mask.size(0)).to(device)
    weight_tensor[weight_mask] = pos_weight
    return weight_tensor, norm


def get_acc(adj_rec, adj_label):
    labels_all = adj_label.view(-1).long()
    preds_all = (adj_rec > 0.5).view(-1).long()
    accuracy = (preds_all == labels_all).sum().float() / labels_all.size(0)
    return accuracy


def get_scores(edges_pos, edges_neg, adj_rec):
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    adj_rec = adj_rec.cpu()
    # Predict on test set of edges
    preds = []
    for e in edges_pos:
        preds.append(sigmoid(adj_rec[e[0], e[1]].item()))

    preds_neg = []
    for e in edges_neg:
        preds_neg.append(sigmoid(adj_rec[e[0], e[1]].data))

    preds_all = np.hstack([preds, preds_neg])
    labels_all = np.hstack([np.ones(len(preds)), np.zeros(len(preds_neg))])
    roc_score = roc_auc_score(labels_all, preds_all)
    ap_score = average_precision_score(labels_all, preds_all)

    return roc_score, ap_score

def VGAE(graph, id2idx, target_dir, args):
    if not os.path.exists('./emb/'):
        os.mkdir('./emb/')
    device = torch.device("cuda:{}".format(args.gpu_id) if torch.cuda.is_available() else "cpu")
    
    # Extract node features
    feats = np.load(os.path.join(target_dir, "feats.npy"), allow_pickle=True)
    feats = torch.Tensor(feats).to(device)
    in_dim = feats.shape[-1]

    # generate input
    graph = dgl.from_networkx(graph)
    adj_orig = graph.adjacency_matrix().to_dense()
    
    train_edge_idx, val_edges, val_edges_false, test_edges, test_edges_false = mask_test_edges_dgl(graph, adj_orig)
    graph = graph.to(device)

    # create train graph
    train_edge_idx = torch.tensor(train_edge_idx).to(device)
    train_graph = dgl.edge_subgraph(graph, train_edge_idx, relabel_nodes=False)
    train_graph = train_graph.to(device)
    adj = train_graph.adjacency_matrix().to_dense().to(device)

    # compute loss parameters
    weight_tensor, norm = compute_loss_para(adj, device)

    # create model
    vgae_model = VGAEModel(in_dim, args.hidden1, args.hidden2)
    vgae_model = vgae_model.to(device)

    # create training component
    optimizer = torch.optim.Adam(vgae_model.parameters(), lr=args.learning_rate)
    print('VGAE Total Parameters:', sum([p.nelement() for p in vgae_model.parameters()]))

    # create training epoch
    for epoch in range(args.epochs):
        t = time.time()

        # Training and validation using a full graph
        vgae_model.train()

        logits,z = vgae_model.forward(graph, feats, device)
        #print(logits.shape, adj.shape)

        # compute loss
        loss = norm * F.binary_cross_entropy(logits.view(-1), adj.view(-1), weight=weight_tensor)
        kl_divergence = 0.5 / logits.size(0) * (
                1 + 2 * vgae_model.log_std - vgae_model.mean ** 2 - torch.exp(vgae_model.log_std) ** 2).sum(
            1).mean()
        loss -= kl_divergence

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_acc = get_acc(logits, adj)

        val_roc, val_ap = get_scores(val_edges, val_edges_false, logits)

        # Print out performance
        if (epoch+1)%20==0:
            print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(loss.item()), "train_acc=",
                "{:.5f}".format(train_acc), "val_roc=", "{:.5f}".format(val_roc), "val_ap=", "{:.5f}".format(val_ap),
                "time=", "{:.5f}".format(time.time() - t))

    np.save('./emb/'+args.dataset+'_vgae_emb',z.cpu().detach().numpy())
    test_roc, test_ap = get_scores(test_edges, test_edges_false, logits)
    print("End of training!", "test_roc=", "{:.5f}".format(test_roc), "test_ap=", "{:.5f}".format(test_ap))

    return z.cpu().detach().numpy()