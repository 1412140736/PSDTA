import torch.nn as nn
import torch.nn.functional as F
import torch
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.utils import to_dense_adj


class GIBGCN(torch.nn.Module):
    def __init__(self, protein_dim3,hidden_dim):
        super(GIBGCN, self).__init__()

        self.cluster1 = Linear(protein_dim3+hidden_dim, protein_dim3)
        self.cluster=nn.ModuleList()

        self.cluster2 = Linear(protein_dim3, 2)
        self.mse_loss = nn.MSELoss()
        self.fc1=Linear(protein_dim3,protein_dim3)

        self.fc2=Linear(protein_dim3,hidden_dim)

    def reset_parameters(self):
        """Reset the parameters of all layers to their initial state."""
        self.cluster1.reset_parameters()
        self.cluster2.reset_parameters()
        self.fc1.reset_parameters()
        self.fc2.reset_parameters()

    def assignment(self,x):
        x = F.relu(self.cluster1(x))
        x=self.cluster2(x)
        return x

    def aggregate(self, assignment, x, batch, edge_index):
        if assignment.get_device()<0:
            batch = torch.tensor(batch, device="cpu")
            max_id = torch.max(batch)
            EYE = torch.ones(2, device="cpu")
        else:
            batch=torch.tensor(batch,device="cuda:"+str(assignment.get_device()))
            max_id = torch.max(batch)
            EYE = torch.ones(2,device="cuda:"+str(assignment.get_device()))


        all_pos_penalty = 0
        all_graph_embedding = []
        all_pos_embedding = []

        st = 0
        end = 0

        for i in range(int(max_id + 1)):
            j = 0
            while batch[st + j] == i and st + j <= len(batch) - 2:
                j += 1

            end = st + j

            if end == len(batch) - 1:
                end += 1
            one_batch_x = x[st:end]
            one_batch_assignment = assignment[st:end]

            #sum_assignment = torch.sum(x, dim=0, keepdim=False)[0]
            group_features=one_batch_assignment[:,:,0].unsqueeze(-1)
            pos_embedding=group_features*one_batch_x
            pos_embedding=torch.mean(pos_embedding,dim=1)

            Adj = to_dense_adj(edge_index)[0]
            new_adj = torch.matmul(torch.transpose(one_batch_assignment, 1, 2), Adj)
            new_adj = torch.matmul(new_adj, one_batch_assignment)

            normalize_new_adj = F.normalize(new_adj, p=1, dim=2,eps = 0.00001)

            if assignment.get_device() < 0:
                pos_penalty = torch.tensor(0.0, device="cpu")
            else:
                pos_penalty=torch.tensor(0.0,device="cuda:"+str(assignment.get_device()))
            for p in range(normalize_new_adj.shape[0]):
                norm_diag = torch.diag(normalize_new_adj[p])
                pos_penalty += self.mse_loss(norm_diag, EYE)/normalize_new_adj.shape[0]

            graph_embedding = torch.mean(one_batch_x, dim=1)

            all_pos_embedding.append(pos_embedding)
            all_graph_embedding.append(graph_embedding)

            all_pos_penalty = all_pos_penalty + pos_penalty
            st = end

        all_pos_embedding = torch.cat(tuple(all_pos_embedding), dim=0)
        all_graph_embedding = torch.cat(tuple(all_graph_embedding), dim=0)
        all_pos_penalty = all_pos_penalty / (max_id + 1)

        return all_pos_embedding,all_graph_embedding, all_pos_penalty

    def forward(self, emb, edge_index, batch, prot_feature):
        assignment = torch.nn.functional.softmax(self.assignment(emb), dim=-1)
        all_subgraph_embedding, all_graph_embedding,all_pos_penalty = self.aggregate(assignment, prot_feature, batch, edge_index)


        out = F.relu(self.fc1(all_subgraph_embedding))
        out = F.dropout(out, p=0.5, training=self.training)
        out = self.fc2(out)
        return out,all_subgraph_embedding, all_graph_embedding, all_pos_penalty,assignment  #注意此处修改了out形状，多了第二个维度


class Discriminator(torch.nn.Module):
    def __init__(self, hidden_size):
        super(Discriminator, self).__init__()

        self.input_size = 2 * hidden_size
        self.hidden_size = hidden_size
        self.lin1 = torch.nn.Linear(self.input_size,self.hidden_size)
        self.lin2 = torch.nn.Linear(self.hidden_size, 1)

    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()


    def forward(self, embeddings,positive):
        cat_embeddings = torch.cat((embeddings, positive),dim = -1)

        pre = F.relu(self.lin1(cat_embeddings))
        pre = F.dropout(pre,training=self.training)
        pre = self.lin2(pre)

        return pre