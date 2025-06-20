from turtle import position
import torch
from torch import nn
import numpy as np
import math


from src.models.ext.tgn.model.temporal_attention import TemporalAttentionLayer
# from model.temporal_attention import TemporalAttentionLayer


class EmbeddingModule(nn.Module):
    def __init__(self, node_features, edge_features, memory, neighbor_finder, time_encoder, n_layers,
                             n_node_features, n_edge_features, n_time_features, embedding_dimension, device,
                             dropout):
        super(EmbeddingModule, self).__init__()
        self.node_features = node_features
        self.edge_features = edge_features
        # self.memory = memory
        self.neighbor_finder = neighbor_finder
        self.time_encoder = time_encoder
        self.n_layers = n_layers
        self.n_node_features = n_node_features
        self.n_edge_features = n_edge_features
        self.n_time_features = n_time_features
        self.dropout = dropout
        self.embedding_dimension = embedding_dimension
        self.device = device

    def compute_embedding(self, memory, source_nodes, timestamps, n_layers, n_neighbors=20, time_diffs=None,
                                                use_time_proj=True):
        pass


class IdentityEmbedding(EmbeddingModule):
    def compute_embedding(self, memory, source_nodes, timestamps, n_layers, n_neighbors=20, time_diffs=None,
                                                use_time_proj=True):
        return memory[source_nodes, :], None


class TimeEmbedding(EmbeddingModule):
    def __init__(self, node_features, edge_features, memory, neighbor_finder, time_encoder, n_layers,
                             n_node_features, n_edge_features, n_time_features, embedding_dimension, device,
                             n_heads=2, dropout=0.1, use_memory=True, n_neighbors=1):
        super(TimeEmbedding, self).__init__(node_features, edge_features, memory,
                                                                                neighbor_finder, time_encoder, n_layers,
                                                                                n_node_features, n_edge_features, n_time_features,
                                                                                embedding_dimension, device, dropout)

        class NormalLinear(nn.Linear):
            # From Jodie code
            def reset_parameters(self):
                stdv = 1. / math.sqrt(self.weight.size(1))
                self.weight.data.normal_(0, stdv)
                if self.bias is not None:
                    self.bias.data.normal_(0, stdv)

        self.embedding_layer = NormalLinear(1, self.n_node_features)

    def compute_embedding(self, memory, source_nodes, timestamps, n_layers, n_neighbors=20, time_diffs=None,
                                                use_time_proj=True):
        source_embeddings = memory[source_nodes, :] * (1 + self.embedding_layer(time_diffs.unsqueeze(1)))

        return source_embeddings, None


class GraphEmbedding(EmbeddingModule):
    def __init__(self, node_features, edge_features, memory, neighbor_finder, time_encoder, n_layers,
                             n_node_features, n_edge_features, n_time_features, embedding_dimension, device,
                             n_heads=2, dropout=0.1, use_memory=True):
        super(GraphEmbedding, self).__init__(node_features, edge_features, memory,
                                                                                 neighbor_finder, time_encoder, n_layers,
                                                                                 n_node_features, n_edge_features, n_time_features,
                                                                                 embedding_dimension, device, dropout)

        self.use_memory = use_memory
        self.device = device
        self.atten_weights_list = []
        self.n_heads = n_heads


    def compute_embedding(self, memory, source_nodes, timestamps, n_layers, n_neighbors=20, time_diffs=None, use_time_proj=True,
                                                edge_idx_preserve_list=None,
                                                candidate_weights_dict=None,
                                                num_neighbors=None):
        """Recursive implementation of curr_layers temporal graph attention layers.

        src_idx_l [batch_size]: users / items input ids.
        cut_time_l [batch_size]: scalar representing the instant of the time where we want to extract the user / item representation.
        curr_layers [scalar]: number of temporal convolutional layers to stack.
        num_neighbors [scalar]: number of temporal neighbor to consider in each convolutional layer.
        """

        assert (n_layers >= 0)
        if num_neighbors is None:
            num_neighbors = n_neighbors

        source_nodes_torch = torch.from_numpy(source_nodes).long().to(self.device)
        timestamps_torch = torch.unsqueeze(torch.from_numpy(timestamps).float().to(self.device), dim=1)

        # query node always has the start time -> time span == 0
        source_nodes_time_embedding = self.time_encoder(torch.zeros_like(
            timestamps_torch))

        source_node_features = self.node_features[source_nodes_torch, :]

        if self.use_memory:
            source_node_features = memory[source_nodes, :] + source_node_features

        if n_layers == 0:
            return source_node_features
        else:

            neighbors, edge_idxs, edge_times = self.neighbor_finder.get_temporal_neighbor(
                                                                                    source_nodes,
                                                                                    timestamps,
                                                                                    num_neighbors=num_neighbors,
                                                                                    edge_idx_preserve_list=edge_idx_preserve_list,
                                                                                    )

            neighbors_torch = torch.from_numpy(neighbors).long().to(self.device)

            edge_idxs = torch.from_numpy(edge_idxs).long().to(self.device)


            edge_deltas = timestamps[:, np.newaxis] - edge_times

            # edge_times = torch.from_numpy(edge_times).float().to(self.device)


            edge_deltas_torch = torch.from_numpy(edge_deltas).float().to(self.device)

            neighbors = neighbors.flatten()
            # import ipdb; ipdb.set_trace()
            neighbor_embeddings = self.compute_embedding(memory,
                                                        neighbors,
                                                        # np.repeat(timestamps, n_neighbors),
                                                        edge_times.flatten(), # NOTE: important! otherwise igh_finder cannot find some neighbors that it should find.
                                                        n_layers=n_layers - 1,
                                                        n_neighbors=n_neighbors,
                                                        edge_idx_preserve_list=edge_idx_preserve_list,
                                                        candidate_weights_dict=candidate_weights_dict,
                                                        )

            effective_n_neighbors = n_neighbors if n_neighbors > 0 else 1
            neighbor_embeddings = neighbor_embeddings.view(len(source_nodes), effective_n_neighbors, -1)
            edge_time_embeddings = self.time_encoder(edge_deltas_torch)

            edge_features = self.edge_features[edge_idxs, :]

            # mask = edge_idxs == 0 # NOTE: True to be masked out, i.e., 0 positions
            position0 = edge_idxs == 0
            mask = torch.zeros_like(position0).to(dtype=torch.float32)
            mask[position0] = -1e10

            # import ipdb; ipdb.set_trace()
            if candidate_weights_dict is not None:
                # TODO: support for pg explainer.
                # import ipdb; ipdb.set_trace()
                position0 = edge_idxs == 0
                mask = torch.zeros_like(position0).to(dtype=torch.float32)

                event_idxs = candidate_weights_dict['candidate_events']
                event_weights = candidate_weights_dict['edge_weights']
                for i, e_idx in enumerate(event_idxs):
                    indices = edge_idxs == e_idx
                    mask[indices] = event_weights[i]

                mask[position0] = -1e10 # because the addition in torch's multi-head attention implementation
                # import ipdb; ipdb.set_trace()

            source_embedding, atten_weights = self.aggregate(n_layers, source_node_features,
                                            source_nodes_time_embedding,
                                            neighbor_embeddings,
                                            edge_time_embeddings,
                                            edge_features,
                                            mask)



            # preserve_mask = edge_idxs != 0
            # edge_idxs = edge_idxs[preserve_mask]
            # atten_weights = atten_weights[preserve_mask].reshape((self.n_heads, neighbors_torch.shape[1]))
            self.atten_weights_list.append({
                'layer': n_layers,
                'src_nodes': source_nodes_torch[source_nodes_torch!=0],
                'src_ngh_nodes': neighbors_torch[neighbors_torch!=0],
                'src_ngh_eidx': edge_idxs[edge_idxs!=0],
                'attn_weight': atten_weights[edge_idxs!=0],
            })
            # import ipdb; ipdb.set_trace()

            return source_embedding

    def aggregate(self, n_layers, source_node_features, source_nodes_time_embedding,
                                neighbor_embeddings,
                                edge_time_embeddings, edge_features, mask):
        return None


class GraphSumEmbedding(GraphEmbedding):
    def __init__(self, node_features, edge_features, memory, neighbor_finder, time_encoder, n_layers,
                             n_node_features, n_edge_features, n_time_features, embedding_dimension, device,
                             n_heads=2, dropout=0.1, use_memory=True):
        super(GraphSumEmbedding, self).__init__(node_features=node_features,
                                                edge_features=edge_features,
                                                memory=memory,
                                                neighbor_finder=neighbor_finder,
                                                time_encoder=time_encoder, n_layers=n_layers,
                                                n_node_features=n_node_features,
                                                n_edge_features=n_edge_features,
                                                n_time_features=n_time_features,
                                                embedding_dimension=embedding_dimension,
                                                device=device,
                                                n_heads=n_heads, dropout=dropout,
                                                use_memory=use_memory)
        self.linear_1 = torch.nn.ModuleList([torch.nn.Linear(embedding_dimension + n_time_features +
                                                                n_edge_features, embedding_dimension)
                                                                                 for _ in range(n_layers)])
        self.linear_2 = torch.nn.ModuleList(
            [torch.nn.Linear(embedding_dimension + n_node_features + n_time_features,
                                             embedding_dimension) for _ in range(n_layers)])

    def aggregate(self, n_layer, source_node_features, source_nodes_time_embedding,
                                neighbor_embeddings,
                                edge_time_embeddings, edge_features, mask):
        neighbors_features = torch.cat([neighbor_embeddings, edge_time_embeddings, edge_features],
                                                                     dim=2)
        neighbor_embeddings = self.linear_1[n_layer - 1](neighbors_features)
        neighbors_sum = torch.nn.functional.relu(torch.sum(neighbor_embeddings, dim=1))

        source_features = torch.cat([source_node_features,
                                                                 source_nodes_time_embedding.squeeze()], dim=1)
        source_embedding = torch.cat([neighbors_sum, source_features], dim=1)
        source_embedding = self.linear_2[n_layer - 1](source_embedding)

        return source_embedding, None


class GraphAttentionEmbedding(GraphEmbedding): #! actually use this
    def __init__(self, node_features, edge_features, memory, neighbor_finder, time_encoder, n_layers,
                             n_node_features, n_edge_features, n_time_features, embedding_dimension, device,
                             n_heads=2, dropout=0.1, use_memory=True):
        super(GraphAttentionEmbedding, self).__init__(node_features, edge_features, memory,
                                                    neighbor_finder, time_encoder, n_layers,
                                                    n_node_features, n_edge_features,
                                                    n_time_features,
                                                    embedding_dimension, device,
                                                    n_heads, dropout,
                                                    use_memory)

        self.attention_models = torch.nn.ModuleList([TemporalAttentionLayer(
            n_node_features=n_node_features,
            n_neighbors_features=n_node_features,
            n_edge_features=n_edge_features,
            time_dim=n_time_features,
            n_head=n_heads,
            dropout=dropout,
            output_dimension=n_node_features)
            for _ in range(n_layers)])


    def aggregate(self, n_layer, source_node_features, source_nodes_time_embedding,
                                neighbor_embeddings,
                                edge_time_embeddings, edge_features, mask):
        attention_model = self.attention_models[n_layer - 1]

        source_embedding, atten_weights = attention_model(source_node_features,
                                            source_nodes_time_embedding,
                                            neighbor_embeddings,
                                            edge_time_embeddings,
                                            edge_features,
                                            mask)

        return source_embedding, atten_weights


def get_embedding_module(module_type, node_features, edge_features, memory, neighbor_finder,
                                                 time_encoder, n_layers, n_node_features, n_edge_features, n_time_features,
                                                 embedding_dimension, device,
                                                 n_heads=2, dropout=0.1, n_neighbors=None,
                                                 use_memory=True):
    if module_type == "graph_attention":
        return GraphAttentionEmbedding(node_features=node_features,
                                    edge_features=edge_features,
                                    memory=memory,
                                    neighbor_finder=neighbor_finder,
                                    time_encoder=time_encoder,
                                    n_layers=n_layers,
                                    n_node_features=n_node_features,
                                    n_edge_features=n_edge_features,
                                    n_time_features=n_time_features,
                                    embedding_dimension=embedding_dimension,
                                    device=device,
                                    n_heads=n_heads, dropout=dropout, use_memory=use_memory)
    elif module_type == "graph_sum":
        return GraphSumEmbedding(node_features=node_features,
                                edge_features=edge_features,
                                memory=memory,
                                neighbor_finder=neighbor_finder,
                                time_encoder=time_encoder,
                                n_layers=n_layers,
                                n_node_features=n_node_features,
                                n_edge_features=n_edge_features,
                                n_time_features=n_time_features,
                                embedding_dimension=embedding_dimension,
                                device=device,
                                n_heads=n_heads, dropout=dropout, use_memory=use_memory)

    elif module_type == "identity":
        return IdentityEmbedding(node_features=node_features,
                                edge_features=edge_features,
                                memory=memory,
                                neighbor_finder=neighbor_finder,
                                time_encoder=time_encoder,
                                n_layers=n_layers,
                                n_node_features=n_node_features,
                                n_edge_features=n_edge_features,
                                n_time_features=n_time_features,
                                embedding_dimension=embedding_dimension,
                                device=device,
                                dropout=dropout)
    elif module_type == "time":
        return TimeEmbedding(node_features=node_features,
                            edge_features=edge_features,
                            memory=memory,
                            neighbor_finder=neighbor_finder,
                            time_encoder=time_encoder,
                            n_layers=n_layers,
                            n_node_features=n_node_features,
                            n_edge_features=n_edge_features,
                            n_time_features=n_time_features,
                            embedding_dimension=embedding_dimension,
                            device=device,
                            dropout=dropout,
                            n_neighbors=n_neighbors)
    else:
        raise ValueError("Embedding Module {} not supported".format(module_type))


