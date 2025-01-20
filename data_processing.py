import torch
from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform

class MultigraphData(Data):
    """ 
    Some additional properties:
    * simplified_edge_index: for simple graph, only one edge between tuple of two nodes.
    * simplified_edge_batch: index to map from full edge_index to simplified_edge_index.
    * edge_direction: self-to-self, original, reverse (0, 1, 2).
    """
    
    def __inc__(self, key, value, *args, **kwargs):
        if key == 'simplified_edge_index':
            return self.num_nodes
        if key == 'simplified_edge_batch':
            return self.simplified_edge_index.size(-1)
        return super().__inc__(key, value, *args, **kwargs)

class ToMultigraph(BaseTransform):
    def __call__(self, data): 
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        
        # Create full edge_index; assume original input edge_index is single direction directed; multi-edge is allowed
        self_loop_indices = edge_index[0] == edge_index[1]
        self_loops = edge_index[:, self_loop_indices]
        other_edges = edge_index[:, ~self_loop_indices]
        reversed_other_edges = torch.stack([other_edges[1], other_edges[0]])

        edge_index = torch.cat([self_loops, other_edges, reversed_other_edges], dim=-1)
        
        if edge_attr is not None:
            edge_attr = torch.cat([edge_attr[self_loop_indices], 
                                   edge_attr[~self_loop_indices], 
                                   edge_attr[~self_loop_indices]], dim=0)

        edge_direction = torch.cat([
            torch.full((self_loops.size(-1),), 0),
            torch.full((other_edges.size(-1),), 1),
            torch.full((reversed_other_edges.size(-1),), 2)
        ], dim=0)

        # Map to simplified edge; currently ignore the edge direction (this is fine)
        simplified_edge_mapping = {}
        simplified_edge_index = []
        simplified_edge_batch = []
        i = 0
        
        for edge in edge_index.T:
            # Transform edge to tuple
            tuple_edge = tuple(edge.tolist())
            if tuple_edge not in simplified_edge_mapping:
                simplified_edge_index.append(edge)
                simplified_edge_mapping[tuple_edge] = i
                i += 1
            simplified_edge_batch.append(simplified_edge_mapping[tuple_edge])
        
        simplified_edge_index = torch.stack(simplified_edge_index).T
        simplified_edge_batch = torch.LongTensor(simplified_edge_batch)

        data = MultigraphData.from_dict(data.to_dict())
        data.edge_index = edge_index
        data.edge_attr = edge_attr
        data.edge_direction = edge_direction
        data.simplified_edge_index = simplified_edge_index
        data.simplified_edge_batch = simplified_edge_batch
        
        return data

class ToDynamicMultigraph(BaseTransform):
    def __call__(self, data):
        x, edge_index, edge_attr, timestamps = data.x, data.edge_index, data.edge_attr, data.timestamps
        
        # Create full edge index with temporal edges based on timestamps
        self_loop_indices = edge_index[0] == edge_index[1]
        self_loops = edge_index[:, self_loop_indices]

        # Define your threshold logic here (e.g., last N days)
        time_threshold = 0  # Replace with actual logic to calculate the threshold
        
        # Example condition for filtering edges
        recent_edges_mask = timestamps > time_threshold  
        
        recent_edges = edge_index[:, recent_edges_mask]
        
        other_edges = recent_edges[:, ~self_loop_indices]
        
        reversed_other_edges = torch.stack([other_edges[1], other_edges[0]])
        
        # Combine all edges including self-loops and reversed edges
        full_edge_index = torch.cat([self_loops, other_edges, reversed_other_edges], dim=-1)

        if edge_attr is not None:
            # Handle attributes accordingly (you may want to filter or transform them as well)
            edge_attr = torch.cat([edge_attr[self_loop_indices], 
                                   edge_attr[recent_edges_mask], 
                                   edge_attr[recent_edges_mask]], dim=0)

        # Edge direction handling remains the same as before
        edge_direction = torch.cat([
            torch.full((self_loops.size(-1),), 0),
            torch.full((other_edges.size(-1),), 1),
            torch.full((reversed_other_edges.size(-1),), 2)
        ], dim=0)

        # Map to simplified edges as before
        simplified_edge_mapping = {}
        simplified_edge_index = []
        simplified_edge_batch = []
        
        i = 0
        for edge in full_edge_index.T:
            tuple_edge = tuple(edge.tolist())
            if tuple_edge not in simplified_edge_mapping:
                simplified_edge_index.append(edge)
                simplified_edge_mapping[tuple_edge] = i
                i += 1
            simplified_edge_batch.append(simplified_edge_mapping[tuple_edge])
        
        simplified_edge_index = torch.stack(simplified_edge_index).T
        simplified_edge_batch = torch.LongTensor(simplified_edge_batch)

        # Create a new MultigraphData object with the updated information
        data = MultigraphData.from_dict(data.to_dict())
        
        # Update with new values
        data.edge_index = full_edge_index
        data.edge_attr = edge_attr
        data.edge_direction = edge_direction
        data.simplified_edge_index = simplified_edge_index
        data.simplified_edge_batch = simplified_edge_batch
        
        return data