import json
import os
import argparse
import numpy as np
import networkx as nx
from networkx.readwrite import json_graph
import random
import numpy as np
from vgae import VGAE
import numba
import scipy.sparse as sp
import time
import ot
import warnings
warnings.filterwarnings("ignore")

def parse_args():
    parser = argparse.ArgumentParser(description="Attack UIL model")
    parser.add_argument('--dataset', default="douban", help='Dataset')
    parser.add_argument('--ratio', default=0.1, type=float, help="Flipped ratio")

    #Rand Walk and EMD parameters
    parser.add_argument('--walks_per_node', default=1000, type=int, help="random walk numbers per node")
    parser.add_argument('--walk_len', default=5, type=int, help="random walk length")
    parser.add_argument('--lamda', type=float, default=1, help='lambda parameter')

    #VGAE parameters
    parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate used in VGAE.')
    parser.add_argument('--epochs', '-e', type=int, default=1000, help='Number of epochs to train.')
    parser.add_argument('--hidden1', '-h1', type=int, default=32, help='Number of units in hidden layer 1.')
    parser.add_argument('--hidden2', '-h2', type=int, default=16, help='Number of units in hidden layer 2.')
    parser.add_argument('--gpu_id', type=int, default=0, help='GPU id to use.')

    return parser.parse_args()

def load_id2idx(base_dir):
    '''
    mapping nodes in a network to id.
    '''
    id2idx_file = os.path.join(base_dir, 'id2idx.json')
    id2idx = {}
    id2idx = json.load(open(id2idx_file)) 
    for k, v in id2idx.items():
        id2idx[str(k)] = v
    return id2idx

def load_graph(base_dir):
    '''
    load the graph data
    '''
    G_data = json.load(open(os.path.join(base_dir, "G.json")))
    G = json_graph.node_link_graph(G_data)
    id2idx = load_id2idx(base_dir)
    mapping = {k: int(id2idx[k]) for k in G.nodes()}
    G = nx.relabel_nodes(G, mapping)
    return G

def load_H(dataset):
    '''
    load prior knowledge matrix H, here we futher generate the indicator I(H_{i,:}) for conveniently calculating the edge distribution
    '''
    h_file = './dataset/'+str(dataset)+'/h.npy'
    h = np.load(h_file,allow_pickle=True)
    #here we attack the target network, so H heeds transpose
    h = h.T
    node_num = h.shape[0]
    h_indicator = np.zeros(node_num)
    for i in range(node_num):
        if np.sum(h[i,:])>=1:
            h_indicator[i] = 1
    return h_indicator


def cal_EMD(D, weight, index):
    '''
    Calculate accurate EMD
    '''
    p = weight/weight.sum()
    weight1 = weight.copy()
    weight1[index] = 0
    q = weight1/weight1.sum()   
    return ot.emd2(p,q,D)

def cal_approximate_EMD(emb, weight, index):
    '''
    Calculate approximate EMD with its lower bound
    '''
    if weight.shape[0]==1:
        return 1
    p_g = weight/weight.sum()
    weight1 = weight.copy()
    weight1[index] = 0
    p_g_star = weight1/weight1.sum() 
    d_norm = np.linalg.norm(np.average(emb,axis=0, weights=p_g)-np.average(emb,axis=0, weights=p_g_star))
    return np.square(d_norm)


def construct_adjacency(G):
    '''
    generating the adjacency matrix of graph G
    '''
    adjacency = np.zeros((len(G.nodes()), len(G.nodes())))
    for src_id, trg_id in G.edges():
        adjacency[src_id, trg_id] = 1
        adjacency[trg_id, src_id] = 1
    return adjacency

#random walk process
@numba.jit(nopython=True)
def random_walk(indptr, indices, walk_length, walks_per_node, seed=333):
    np.random.seed(seed)
    N = len(indptr) - 1
    walks = []

    for _ in range(walks_per_node):
        for n in range(N):
            if indptr[n]==indptr[n + 1]:
                continue
            one_walk = []
            for _ in range(walk_length+1):
                one_walk.append(n)
                n = np.random.choice(indices[indptr[n]:indptr[n + 1]])
            walks.append(one_walk)

    return walks

def calculate_score(graph, z, h_indicator, args):
    edge_num = graph.number_of_edges()
    node_num = graph.number_of_nodes()
    candidate_num = int(edge_num*args.ratio*3)

    #calculating edge embedding 
    e2eid = {}
    eid = 0
    edge_emb = np.zeros((edge_num, z.shape[1]*2),dtype=np.float32) 
    for e in graph.edges():
        if e[0]>e[1]: e = (e[1],e[0])
        factor = np.exp(args.lamda*np.clip(h_indicator[e[0]]+h_indicator[e[1]],0,1))
        e2eid[(e[0],e[1])] = [eid, factor]
        e_vector = np.hstack((z[e[0]],z[e[1]]))
        edge_emb[eid] = e_vector/np.linalg.norm(e_vector)
        eid+=1
    print('\n'+'*'*20)
    print('Calculate edge embedding has done!!!')
    print('*'*20)

    #random walk on clean graph
    adj_matrix = sp.csr_matrix(construct_adjacency(graph))
    walks_on_clean_graph = random_walk(adj_matrix.indptr,adj_matrix.indices,args.walk_len,args.walks_per_node)
    print('\n'+'*'*20)
    print('Random walk process on clean ego networks has done!!!')
    print('*'*20,)

    #calculate the edge weight on clean graph
    rw_on_clean_ego = {i:{} for i in range(node_num)}
    total_rw = {v[0]:0 for v in e2eid.values()}
    for wk in walks_on_clean_graph:
        start_node = wk[0]
        for i in range(len(wk)-1):
            if wk[i]>wk[i+1]:
                e = e2eid[(wk[i+1], wk[i])]
            else:
                e = e2eid[(wk[i], wk[i+1])]

            total_rw[e[0]] += e[1]

            if e[0] in rw_on_clean_ego[start_node].keys():
                rw_on_clean_ego[start_node][e[0]] += e[1]
            else:
                rw_on_clean_ego[start_node][e[0]] = e[1]
    print('\n'+'*'*20)
    print('Calculate edge distribution on clean ego networks has done!!!')
    print('*'*20)

    #genarate add and remove edge candidates
    total_rw_list = [[k,v] for k,v in total_rw.items()]
    candidate_for_del = sorted(total_rw_list, key=lambda x:x[1],reverse=True)[:candidate_num]
    candidate_for_del = set([item[0] for item in candidate_for_del])
    candidate_for_add = set()
    deg = graph.degree()
    endpoints1 = sorted([(k,deg[k]) for k in graph.nodes()], key=lambda x:x[1], reverse=True)[:30]
    endpoints2 = sorted([(int(k),deg[int(k)]) for k in np.flatnonzero(h_indicator)], key=lambda x:x[1], reverse=True)
    add_edge = set()
    for i in endpoints1:
        for j in endpoints2:
            if i[0]<j[0]:
                e=(i[0],j[0])
            else:
                e=(j[0],i[0])
            if (e[0]!=e[1]) and (not graph.has_edge(e[0],e[1])) and ((e[0],e[1]) not in add_edge):
                add_edge.add((e[0],e[1]))
                if len(add_edge)==candidate_num:break
        if len(add_edge)==candidate_num:break
    edge_emb = np.vstack((edge_emb,np.zeros((candidate_num, z.shape[1]*2),dtype=np.float32)))
    add_edge = list(add_edge)
    for e in add_edge:
        if e[0]>e[1]: e=(e[1],e[0])
        graph.add_edge(e[0],e[1])
        factor = np.exp(args.lamda*np.clip(h_indicator[e[0]]+h_indicator[e[1]],0,1))
        eid = len(e2eid)
        e2eid[(e[0],e[1])] = [eid,factor]
        candidate_for_add.add(eid)
        e_vector = np.hstack((z[e[0]],z[e[1]]))
        edge_emb[eid] = e_vector/np.linalg.norm(e_vector)

    print('\n'+'*'*20)
    print('Selecting candidates has done!!!')
    print(' Remove candidates num: ',len(candidate_for_del), '\n Add candidates num: ',len(candidate_for_add))
    print('*'*20)


    #random walk on poisoned graph
    adj_matrix = sp.csr_matrix(construct_adjacency(graph))
    walks_on_poisoned = random_walk(adj_matrix.indptr,adj_matrix.indices,args.walk_len,args.walks_per_node)
    print('\n'+'*'*20)
    print('Random walk process on candidate poisoned ego networks has done!!!')
    print('*'*20)

    #calculate the edge weight on poisoned graph
    rw_on_poisoned_ego = {i:{} for i in range(node_num)}
    for wk in walks_on_poisoned:
        start_node = wk[0]
        for i in range(len(wk)-1):
            if wk[i]>wk[i+1]:
                e = e2eid[(wk[i+1], wk[i])]
            else:
                e = e2eid[(wk[i], wk[i+1])]

            if e[0] in rw_on_poisoned_ego[start_node].keys():
                rw_on_poisoned_ego[start_node][e[0]] += e[1]
            else:
                rw_on_poisoned_ego[start_node][e[0]] = e[1]
    print('\n'+'*'*20)
    print('Calculate edge distribution on poisoned ego networks has done!!!')
    print('*'*20)

    #calculate the socre(e) for each e.
    all_num = 0
    cnt = 0
    edge_score = np.zeros((len(e2eid),))
    for node in graph.nodes():
        e1 = rw_on_clean_ego[node]
        e2 = rw_on_poisoned_ego[node]
        num = len(e1)
        edge = []
        weight = []
        for e,w in e1.items():
            edge.append(e)
            weight.append(w)
        
        emb = edge_emb[edge]
        weight = np.asarray(weight)
        # Most kernel values equal to 1, which means flipping edge e will not affect the k ego-network of a node v.
        # So, we minus 1 to each kernel value items, such that the trivial kernel values equal 0, and only the flipped part are added.
        # This is why we add (temp-1) to edge_score and only loops on edges that appears at k ego-network of a node v.
        for index in range(num):
            if edge[index] in candidate_for_del:
                temp = np.exp(-np.abs(cal_approximate_EMD(emb, weight, index)))
                edge_score[edge[index]] += (temp-1)
                all_num+=1

        for ed,wed in e2.items():
            if ed in candidate_for_add:
                emb1 = np.vstack((emb,edge_emb[ed]))
                weight1 = np.append(weight,wed)
                temp = np.exp(-np.abs(cal_approximate_EMD(emb1, weight1, num)))
                edge_score[ed] += (temp-1)
                all_num+=1

        cnt+=1
        if cnt%500==0:
            print(cnt)
    print('\n'+'*'*20)
    print('Calculating candidate edge score has done!!!')
    print('*'*20+'\n')
    #print(np.max(edge_score),np.min(edge_score))
    return edge_score,e2eid

def generate_attack(edge_score, e2eid, id2idx, args, flip_num):
    if not os.path.exists('./attack_graph/'):
        os.mkdir('./attack_graph/')
    if not os.path.exists('./attack_graph/'+args.dataset):
        os.mkdir('./attack_graph/'+args.dataset)
    if not os.path.exists('./attack_graph/'+args.dataset+'/toak'):
        os.mkdir('./attack_graph/'+args.dataset+'/toak')

    fname = '_'.join(['toak', args.dataset, str(args.ratio),'l',str(args.walk_len),'n',str(args.walks_per_node),'lmd',str(args.lamda),'1'])
    save = './attack_graph/'+args.dataset+'/toak/'+fname

    edge = []
    idx2id = {v:k for k,v in id2idx.items()}
    for k in e2eid.keys():
        if edge_score[e2eid[k][0]]!=0:
            edge.append([[k[0], k[1]], edge_score[e2eid[k][0]]])

    #sort edges according to the score
    edge = sorted(edge, key=lambda x:x[1])
    try:
        np.save(args.dataset+str(args.ratio)+'_edge_score', edge)
    except:
        pass

    #save flipped edges
    file = open(save, 'w')
    for e in edge[:flip_num]:
        file.write("{0} {1}\n".format(idx2id[e[0][0]], idx2id[e[0][1]]))
    file.close()
    print('Done. Flipped  Edges are saving at ', save)
    return 0


if __name__ == '__main__':
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    args = parse_args()
    print(args)
    target_dir = './dataset/'+args.dataset+'/target'

    print("START ATTACK THE GRAPH!!!")
    print("Removing {:3.0%} edges, dataset={}, walks per nodes={:5d}, walk length={:3d}, lambda={:3d}".format(args.ratio, args.dataset, args.walks_per_node, args.walk_len, args.lamda))
    graph  = load_graph(target_dir)
    flip_num = int(graph.number_of_edges()*args.ratio)
    print("Total Edges:", len(list(graph.edges())))
    id2idx = load_id2idx(target_dir)

    #generate VGAE embedding
    t = time.time()
    if os.path.exists('./emb/'+args.dataset+'_vgae_emb.npy'):
        print("VGAE embedding already exist! Load it!")
        z = np.load('./emb/'+args.dataset+'_vgae_emb.npy', allow_pickle=True)
    else:
        print('Train VGAE!')
        z = VGAE(graph, id2idx, target_dir, args)

    H = load_H(args.dataset)

    #calculate score
    edge_score,e2eid = calculate_score(graph, z, H, args)

    #generate flipped edges with score
    generate_attack(edge_score,e2eid,id2idx,args,flip_num)
    print("ATTACK FINISHED!!! Spend Time={:.3f}".format(time.time() - t))
