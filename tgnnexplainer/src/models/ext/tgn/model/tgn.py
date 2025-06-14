import logging
from turtle import pos
import numpy as np
import torch
from collections import defaultdict

from src.models.ext.tgn.utils.utils import MergeLayer
from src.models.ext.tgn.modules.memory import Memory
from src.models.ext.tgn.modules.message_aggregator import get_message_aggregator
from src.models.ext.tgn.modules.message_function import get_message_function
from src.models.ext.tgn.modules.memory_updater import get_memory_updater
from src.models.ext.tgn.model.time_encoding import TimeEncode


# from utils.utils import MergeLayer
# from modules.memory import Memory
# from modules.message_aggregator import get_message_aggregator
# from modules.message_function import get_message_function
# from modules.memory_updater import get_memory_updater
# from modules.embedding_module import get_embedding_module
# from model.time_encoding import TimeEncode


class TGN(torch.nn.Module):
    def __init__(self, neighbor_finder, node_features, edge_features, device, n_layers=2,
                             n_heads=2, dropout=0.1, use_memory=False, forbidden_memory_update=False,
                             memory_update_at_start=True, message_dimension=100,
                             memory_dimension=500, embedding_module_type="graph_attention",
                             message_function="mlp",
                             mean_time_shift_src=0, std_time_shift_src=1, mean_time_shift_dst=0,
                             std_time_shift_dst=1, n_neighbors=None, aggregator_type="last",
                             memory_updater_type="gru",
                             use_destination_embedding_in_message=False,
                             use_source_embedding_in_message=False,
                             dyrep=False,
                             mode='tgnne'):
        super(TGN, self).__init__()

        self.num_layers = n_layers
        self.ngh_finder = neighbor_finder
        self.device = device
        self.logger = logging.getLogger(__name__)

        self.node_raw_features = torch.from_numpy(node_features.astype(np.float32)).to(self.device)
        print('Node features shape: ', self.node_raw_features.shape, self.node_raw_features.dtype)
        self.edge_raw_features = torch.from_numpy(edge_features.astype(np.float32)).to(self.device)
        self.node_raw_embed = self.node_raw_features # just a copy for compatiblility in PGExplainerExt._create_explainer_input()
        self.edge_raw_embed = self.edge_raw_features


        self.n_node_features = self.node_raw_features.shape[1]
        self.n_nodes = self.node_raw_features.shape[0]
        self.n_edge_features = self.edge_raw_features.shape[1]
        self.embedding_dimension = self.n_node_features
        self.num_neighbors = n_neighbors
        self.embedding_module_type = embedding_module_type
        self.use_destination_embedding_in_message = use_destination_embedding_in_message
        self.use_source_embedding_in_message = use_source_embedding_in_message
        self.dyrep = dyrep

        self.use_memory = use_memory
        self.forbidden_memory_update = forbidden_memory_update
        self.time_encoder = TimeEncode(dimension=self.n_node_features)
        self.memory = None

        self.mean_time_shift_src = mean_time_shift_src
        self.std_time_shift_src = std_time_shift_src
        self.mean_time_shift_dst = mean_time_shift_dst
        self.std_time_shift_dst = std_time_shift_dst

        if self.use_memory:
            self.memory_dimension = memory_dimension
            self.memory_update_at_start = memory_update_at_start
            raw_message_dimension = 2 * self.memory_dimension + self.n_edge_features + \
                                                            self.time_encoder.dimension
            message_dimension = message_dimension if message_function != "identity" else raw_message_dimension
            self.memory = Memory(n_nodes=self.n_nodes,
                                memory_dimension=self.memory_dimension,
                                input_dimension=message_dimension,
                                message_dimension=message_dimension,
                                device=self.device)
            self.message_aggregator = get_message_aggregator(aggregator_type=aggregator_type,
                                                            device=self.device)
            self.message_function = get_message_function(module_type=message_function,
                                                        raw_message_dimension=raw_message_dimension,
                                                        message_dimension=message_dimension)
            self.memory_updater = get_memory_updater(module_type=memory_updater_type,
                                                    memory=self.memory,
                                                    message_dimension=message_dimension,
                                                    memory_dimension=self.memory_dimension,
                                                    device=self.device)

        self.embedding_module_type = embedding_module_type

        if mode == 'tgnne':

            from src.models.ext.tgn.modules.embedding_module import get_embedding_module
        else:
            from src.models.ext.tgn.modules.temp_me_embedding_module import get_embedding_module

        self.embedding_module = get_embedding_module(module_type=embedding_module_type,
                                                    node_features=self.node_raw_features,
                                                    edge_features=self.edge_raw_features,
                                                    memory=self.memory,
                                                    neighbor_finder=self.ngh_finder,
                                                    time_encoder=self.time_encoder,
                                                    n_layers=self.num_layers,
                                                    n_node_features=self.n_node_features,
                                                    n_edge_features=self.n_edge_features,
                                                    n_time_features=self.n_node_features,
                                                    embedding_dimension=self.embedding_dimension,
                                                    device=self.device,
                                                    n_heads=n_heads, dropout=dropout,
                                                    use_memory=use_memory,
                                                    n_neighbors=self.num_neighbors)

        # MLP to compute probability on an edge given two node embeddings
        self.affinity_score = MergeLayer(self.n_node_features, self.n_node_features,
                                                                         self.n_node_features,
                                                                         1)

    def compute_temporal_embeddings(self, source_nodes, destination_nodes, negative_nodes, edge_times,
                                                                    edge_idxs, n_neighbors=20,
                                                                    edge_idx_preserve_list=None,
                                                                    candidate_weights_dict=None,
                                                                    ):
        """
        Compute temporal embeddings for sources, destinations, and negatively sampled destinations.

        source_nodes [batch_size]: source ids.
        :param destination_nodes [batch_size]: destination ids
        :param negative_nodes [batch_size]: ids of negative sampled destination
        :param edge_times [batch_size]: timestamp of interaction
        :param edge_idxs [batch_size]: index of interaction
        :param n_neighbors [scalar]: number of temporal neighbor to consider in each convolutional
        layer
        :return: Temporal embeddings for sources, destinations and negatives
        """

        n_samples = len(source_nodes)
        nodes = np.concatenate([source_nodes, destination_nodes, negative_nodes])
        positives = np.concatenate([source_nodes, destination_nodes])
        timestamps = np.concatenate([edge_times, edge_times, edge_times])

        memory = None
        time_diffs = None

        # assert self.memory_update_at_start is False, 'We only want to use memory_update_at_end'

        if self.use_memory:
            if self.memory_update_at_start:
                # Update memory for all nodes with messages stored in previous batches
                memory, last_update = self.get_updated_memory(list(range(self.n_nodes)),
                                                            self.memory.messages)
            else: #! We want use this
                memory = self.memory.get_memory(list(range(self.n_nodes)))
                last_update = self.memory.last_update

            ### Compute differences between the time the memory of a node was last updated,
            ### and the time for which we want to compute the embedding of a node
            source_time_diffs = torch.LongTensor(edge_times).to(self.device) - last_update[
                source_nodes].long()
            source_time_diffs = (source_time_diffs - self.mean_time_shift_src) / self.std_time_shift_src
            destination_time_diffs = torch.LongTensor(edge_times).to(self.device) - last_update[
                destination_nodes].long()
            destination_time_diffs = (destination_time_diffs - self.mean_time_shift_dst) / self.std_time_shift_dst
            negative_time_diffs = torch.LongTensor(edge_times).to(self.device) - last_update[
                negative_nodes].long()
            negative_time_diffs = (negative_time_diffs - self.mean_time_shift_dst) / self.std_time_shift_dst

            time_diffs = torch.cat([source_time_diffs, destination_time_diffs, negative_time_diffs],
                                                         dim=0)

        # Compute the embeddings using the embedding module

        node_embedding = self.embedding_module.compute_embedding(memory=memory,
                                                                source_nodes=nodes,
                                                                timestamps=timestamps,
                                                                n_layers=self.num_layers,
                                                                n_neighbors=n_neighbors,
                                                                time_diffs=time_diffs,
                                                                edge_idx_preserve_list=edge_idx_preserve_list,
                                                                candidate_weights_dict=candidate_weights_dict,
                                                                )

        source_node_embedding = node_embedding[:n_samples]
        destination_node_embedding = node_embedding[n_samples: 2 * n_samples]
        negative_node_embedding = node_embedding[2 * n_samples:]

        #! We want to comment this, because want to use memory_update_at_end and don't update memory when computing scores.
        if self.use_memory and (not self.forbidden_memory_update):
            if self.memory_update_at_start:
                # Persist the updates to the memory only for sources and destinations (since now we have
                # new messages for them)
                self.update_memory(positives, self.memory.messages)

                assert torch.allclose(memory[positives], self.memory.get_memory(positives), atol=1e-5), \
                    "Something wrong in how the memory was updated"

                # Remove messages for the positives since we have already updated the memory using them
                self.memory.clear_messages(positives)

            unique_sources, source_id_to_messages = self.get_raw_messages(source_nodes,
                                                                        source_node_embedding,
                                                                        destination_nodes,
                                                                        destination_node_embedding,
                                                                        edge_times, edge_idxs)
            unique_destinations, destination_id_to_messages = self.get_raw_messages(destination_nodes,
                                                                                    destination_node_embedding,
                                                                                    source_nodes,
                                                                                    source_node_embedding,
                                                                                    edge_times, edge_idxs)
            if self.memory_update_at_start:
                self.memory.store_raw_messages(unique_sources, source_id_to_messages)
                self.memory.store_raw_messages(unique_destinations, destination_id_to_messages)
            else:
                #! always using memory_update_at_end
                self.update_memory(unique_sources, source_id_to_messages)
                self.update_memory(unique_destinations, destination_id_to_messages)

            if self.dyrep:
                source_node_embedding = memory[source_nodes]
                destination_node_embedding = memory[destination_nodes]
                negative_node_embedding = memory[negative_nodes]

        return source_node_embedding, destination_node_embedding, negative_node_embedding

    def get_prob(self, src_idx_l, target_idx_l, cut_time_l, edge_idxs=None, logit=False, edge_idx_preserve_list=None, candidate_weights_dict=None, num_neighbors=None):
        """
        src_idx_l, target_idx_l, cut_time_l: np.array
        edge_idxs: actually can be None... Because in self.compute_temporal_embeddings(), we will skip self.get_raw_messages() function.
        edge_idx_preserve_list: support for masking out some edges
        candidate_weights_dict: support for pg explainer

        """
        if hasattr(self.embedding_module, 'atten_weights_list'): #! avoid cuda memory leakage
            self.embedding_module.atten_weights_list = []

        if num_neighbors is None:
            num_neighbors = self.num_neighbors

        n_samples = len(src_idx_l)
        negative_nodes = np.array([0, ])
        edge_idxs = None
        source_node_embedding, destination_node_embedding, negative_node_embedding = self.compute_temporal_embeddings(
            src_idx_l, target_idx_l, negative_nodes, cut_time_l, edge_idxs, num_neighbors,
            edge_idx_preserve_list=edge_idx_preserve_list,
            candidate_weights_dict=candidate_weights_dict,
            )

        score = self.affinity_score(torch.cat([source_node_embedding, source_node_embedding], dim=0),
                                                                torch.cat([destination_node_embedding,
                                                                                     negative_node_embedding])).squeeze(dim=0)

        pos_score = score[:n_samples]

        if logit:
            return pos_score
        else:
            return pos_score.sigmoid()



    def compute_edge_probabilities(self, source_nodes, destination_nodes, negative_nodes, edge_times,
                                                                 edge_idxs, n_neighbors=20):
        """
        Compute probabilities for edges between sources and destination and between sources and
        negatives by first computing temporal embeddings using the TGN encoder and then feeding them
        into the MLP decoder.
        :param destination_nodes [batch_size]: destination ids
        :param negative_nodes [batch_size]: ids of negative sampled destination
        :param edge_times [batch_size]: timestamp of interaction
        :param edge_idxs [batch_size]: index of interaction
        :param n_neighbors [scalar]: number of temporal neighbor to consider in each convolutional
        layer
        :return: Probabilities for both the positive and negative edges
        """
        if hasattr(self.embedding_module, 'atten_weights_list'): #! avoid cuda memory leakage
            self.embedding_module.atten_weights_list = []

        n_samples = len(source_nodes)
        source_node_embedding, destination_node_embedding, negative_node_embedding = self.compute_temporal_embeddings(
            source_nodes, destination_nodes, negative_nodes, edge_times, edge_idxs, n_neighbors)

        score = self.affinity_score(torch.cat([source_node_embedding, source_node_embedding], dim=0),
                                                                torch.cat([destination_node_embedding,
                                                                                     negative_node_embedding])).squeeze(dim=0)
        pos_score = score[:n_samples]
        neg_score = score[n_samples:]

        return pos_score.sigmoid(), neg_score.sigmoid()

    def update_memory(self, nodes, messages):
        # Aggregate messages for the same nodes
        unique_nodes, unique_messages, unique_timestamps = \
            self.message_aggregator.aggregate(
                nodes,
                messages)

        if len(unique_nodes) > 0:
            unique_messages = self.message_function.compute_message(unique_messages)

        # Update the memory with the aggregated messages
        self.memory_updater.update_memory(unique_nodes, unique_messages,
                                                                            timestamps=unique_timestamps)

    def get_updated_memory(self, nodes, messages):
        # Aggregate messages for the same nodes
        unique_nodes, unique_messages, unique_timestamps = self.message_aggregator.aggregate(nodes, messages)

        if len(unique_nodes) > 0:
            unique_messages = self.message_function.compute_message(unique_messages)

        updated_memory, updated_last_update = self.memory_updater.get_updated_memory(unique_nodes,
                                                                                    unique_messages,
                                                                                    timestamps=unique_timestamps)

        return updated_memory, updated_last_update

    def get_raw_messages(self, source_nodes, source_node_embedding, destination_nodes,
                                             destination_node_embedding, edge_times, edge_idxs):
        edge_times = torch.from_numpy(edge_times).float().to(self.device)
        edge_features = self.edge_raw_features[edge_idxs]

        source_memory = self.memory.get_memory(source_nodes) if not \
            self.use_source_embedding_in_message else source_node_embedding
        destination_memory = self.memory.get_memory(destination_nodes) if \
            not self.use_destination_embedding_in_message else destination_node_embedding

        source_time_delta = edge_times - self.memory.last_update[source_nodes]
        source_time_delta_encoding = self.time_encoder(source_time_delta.unsqueeze(dim=1)).view(len(
            source_nodes), -1)

        source_message = torch.cat([source_memory, destination_memory, edge_features,
                                                                source_time_delta_encoding],
                                                             dim=1)
        messages = defaultdict(list)
        unique_sources = np.unique(source_nodes)

        for i in range(len(source_nodes)):
            messages[source_nodes[i]].append((source_message[i], edge_times[i]))

        return unique_sources, messages

    def set_neighbor_finder(self, neighbor_finder):
        self.ngh_finder = neighbor_finder
        self.embedding_module.neighbor_finder = neighbor_finder

    def set_neighbor_sampler(self, neighbor_finder):
        self.embedding_module.neighbor_sampler = neighbor_finder

    def grab_subgraph(self, src_idx_l, cut_time_l):
        subgraph = self.embedding_module.neighbor_sampler.find_k_hop(2, src_idx_l, cut_time_l, num_neighbors=self.num_neighbors, e_idx_l=None)
        return subgraph

    def get_node_emb(self, src_idx, tgt_idx, bgd_idx, cut_time, e_idx,
                     subgraph_src, subgraph_tgt, subgraph_bgd, explain_weights=None, edge_attr=None):
        """
        Compute temporal embeddings for sources, destinations, and negatively sampled destinations.
        """
        n_samples = len(src_idx)
        nodes_0 = np.expand_dims(np.concatenate([src_idx, tgt_idx, bgd_idx]), axis=-1)  #[3 * bsz, 1]
        nodes_1 = np.concatenate([subgraph_src[0][0], subgraph_tgt[0][0], subgraph_bgd[0][0]], axis=0)  #[3 * bsz, n]
        nodes_2 = np.concatenate([subgraph_src[0][1], subgraph_tgt[0][1], subgraph_bgd[0][1]], axis=0)  #[3* bsz, n**2]
        node_list = [nodes_0, nodes_1, nodes_2]

        edge_1 = np.concatenate([subgraph_src[1][0], subgraph_tgt[1][0], subgraph_bgd[1][0]], axis=0)  #[3 * bsz, n]
        edge_2 = np.concatenate([subgraph_src[1][1], subgraph_tgt[1][1], subgraph_bgd[1][1]], axis=0)  #[3* bsz, n**2]
        edge_list = [edge_1, edge_2]

        time_1 = np.concatenate([subgraph_src[2][0], subgraph_tgt[2][0], subgraph_bgd[2][0]], axis=0)  #[3 * bsz, n]
        time_2 = np.concatenate([subgraph_src[2][1], subgraph_tgt[2][1], subgraph_bgd[2][1]], axis=0)  #[3* bsz, n**2]
        time_list = [time_1, time_2]
        positives = np.concatenate([src_idx, tgt_idx])

        memory = None
        time_diffs = None

        if self.use_memory:
            if self.memory_update_at_start:
                # Update memory for all nodes with messages stored in previous batches
                memory, last_update = self.get_updated_memory(list(range(self.n_nodes)),self.memory.messages)
            else:
                memory = self.memory.get_memory(list(range(self.n_nodes)))
                last_update = self.memory.last_update

            source_time_diffs = torch.LongTensor(cut_time).to(self.device) - last_update[
                src_idx].long()
            source_time_diffs = (source_time_diffs - self.mean_time_shift_src) / self.std_time_shift_src
            destination_time_diffs = torch.LongTensor(cut_time).to(self.device) - last_update[tgt_idx].long()
            destination_time_diffs = (destination_time_diffs - self.mean_time_shift_dst) / self.std_time_shift_dst
            negative_time_diffs = torch.LongTensor(cut_time).to(self.device) - last_update[bgd_idx].long()
            negative_time_diffs = (negative_time_diffs - self.mean_time_shift_dst) / self.std_time_shift_dst

            time_diffs = torch.cat([source_time_diffs, destination_time_diffs, negative_time_diffs],
                                                         dim=0)

        # Compute the embeddings using the embedding module
        if edge_attr is not None:
            node_embedding = self.embedding_module.embedding_update_attr(memory=memory,
                                                                    node_list=node_list,
                                                                    edge_list=edge_list,
                                                                    time_list=time_list,
                                                                    cut_time=cut_time,
                                                                    n_layers=self.num_layers,
                                                                    edge_features=edge_attr,
                                                                    explain_weights=explain_weights
                                                                    )  # [3*bs, node_feature]
        else:
            node_embedding = self.embedding_module.embedding_update(memory=memory,
                                                                    node_list=node_list,
                                                                    edge_list=edge_list,
                                                                    time_list=time_list,
                                                                    cut_time=cut_time,
                                                                    n_layers=self.num_layers,
                                                                    explain_weights=explain_weights
                                                                    )  #[3*bs, node_feature]
        source_node_embedding = node_embedding[:n_samples]
        destination_node_embedding = node_embedding[n_samples: 2 * n_samples]
        negative_node_embedding = node_embedding[2 * n_samples:]

        #! We want to comment this, because want to use memory_update_at_end and don't update memory when computing scores.
        if self.use_memory and (not self.forbidden_memory_update):
            if self.memory_update_at_start:
                # Persist the updates to the memory only for sources and destinations (since now we have
                # new messages for them)
                self.update_memory(positives, self.memory.messages)

                # assert torch.allclose(memory[positives], self.memory.get_memory(positives), atol=1e-5), \
                #     "Something wrong in how the memory was updated"

                # Remove messages for the positives since we have already updated the memory using them
                self.memory.clear_messages(positives)

            unique_sources, source_id_to_messages = self.get_raw_messages(src_idx,
                                                                        source_node_embedding,
                                                                        tgt_idx,
                                                                        destination_node_embedding,
                                                                        cut_time, e_idx)
            unique_destinations, destination_id_to_messages = self.get_raw_messages(tgt_idx,
                                                                                    destination_node_embedding,
                                                                                    src_idx,
                                                                                    source_node_embedding,
                                                                                    cut_time, e_idx)
            if self.memory_update_at_start:
                self.memory.store_raw_messages(unique_sources, source_id_to_messages)
                self.memory.store_raw_messages(unique_destinations, destination_id_to_messages)
            else:
                #! always using memory_update_at_end
                self.update_memory(unique_sources, source_id_to_messages)
                self.update_memory(unique_destinations, destination_id_to_messages)


        return source_node_embedding, destination_node_embedding, negative_node_embedding

    def contrast(self, src_idx, tgt_idx, bgd_idx, cut_time, e_idx,
                     subgraph_src, subgraph_tgt, subgraph_bgd,
                 explain_weights=None, edge_attr=None):

        if hasattr(self.embedding_module, 'atten_weights_list'): #! avoid cuda memory leakage
            self.embedding_module.atten_weights_list = []

        n_samples = len(src_idx)
        source_node_embedding, destination_node_embedding, negative_node_embedding = \
            self.get_node_emb(src_idx, tgt_idx, bgd_idx, cut_time, e_idx,
                     subgraph_src, subgraph_tgt, subgraph_bgd, explain_weights, edge_attr)

        score = self.affinity_score(torch.cat([source_node_embedding, source_node_embedding], dim=0),torch.cat([destination_node_embedding,
                                                                                     negative_node_embedding])).squeeze(dim=0)
        pos_score = score[:n_samples]
        neg_score = score[n_samples:]

        return pos_score, neg_score

    def retrieve_edge_features(self, subgraph_src,subgraph_tgt,subgraph_bgd):
        edge_1 = np.concatenate([subgraph_src[1][0], subgraph_tgt[1][0], subgraph_bgd[1][0]], axis=0)  #[3 * bsz, n]
        edge_2 = np.concatenate([subgraph_src[1][1], subgraph_tgt[1][1], subgraph_bgd[1][1]], axis=0)  #[3* bsz, n**2]
        edge_list = [edge_1, edge_2]
        edge_features = []
        for i in range(len(edge_list)):
            batch_edge_idx = torch.from_numpy(edge_list[i]).long().to(self.device)
            edge_features.append(self.edge_raw_features(batch_edge_idx))
        return edge_features

