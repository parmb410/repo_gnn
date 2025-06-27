import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv
from gnn.graph_builder import GraphBuilder
import numpy as np

class TemporalGCN(nn.Module):
    """
    Temporal Graph Convolutional Network for sensor-based activity recognition
    Combines 1D convolutions for temporal features with GCN for spatial features
    """
    def __init__(self, input_dim, hidden_dim, output_dim, graph_builder=None):
        super().__init__()
        self.graph_builder = graph_builder or GraphBuilder()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.in_features = input_dim  # For compatibility with algorithms
        self.out_features = output_dim  # For compatibility with algorithms
        
        # Temporal feature extractor
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(input_dim, 16, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        # Spatial graph convolutions
        self.gcn1 = GCNConv(32, hidden_dim)
        self.gcn2 = GCNConv(hidden_dim, hidden_dim)
        # Classifier
        self.fc = nn.Linear(hidden_dim, output_dim)
        
        # Reconstruction layer for pretraining
        self.recon = nn.Linear(output_dim, input_dim)  # For mean feature reconstruction
        
        # Cache for graphs to avoid rebuilding
        self.graph_cache = {}

    def _create_chain_graph(self, num_nodes: int) -> torch.LongTensor:
        """Create temporal chain graph as fallback"""
        edges = []
        for i in range(num_nodes - 1):
            # Connect consecutive time steps
            edges.append([i, i+1])
            edges.append([i+1, i])
        return torch.tensor(edges, dtype=torch.long).t().contiguous()

    def forward(self, x):
        # ==== BEGIN: Support PyG Data/Batch or plain Tensor ====
        if hasattr(x, 'x'):
            # x is a torch_geometric.data.Data or Batch
            x_feat = x.x
            batch = x.batch if hasattr(x, 'batch') else None
            if x_feat.dim() == 2 and batch is not None:
                batch_size = batch.max().item() + 1
                time_steps = torch.bincount(batch)[0].item()
                x_feat = x_feat.view(batch_size, time_steps, -1)
            x = x_feat
        # ==== END: Support PyG Data/Batch or plain Tensor ====

        # --- Ensure input is [batch, channels, time] ---
        if x.dim() == 3:
            # x.shape: [batch, time, channels] or [batch, channels, time]
            if x.shape[2] == self.input_dim:
                # [batch, time, channels] -> [batch, channels, time]
                x = x.permute(0, 2, 1)
            elif x.shape[1] == self.input_dim:
                # [batch, channels, time] -> already correct
                pass
            else:
                raise RuntimeError(f"Unexpected input shape {x.shape}, input_dim={self.input_dim}")
        else:
            raise RuntimeError(f"Expected 3D input, got {x.shape}")

        batch_size, channels, timesteps = x.shape

        # Temporal convolution: [batch, features, timesteps]
        x = self.temporal_conv(x)  # Output: [batch, 32, timesteps//4]
        _, features, reduced_timesteps = x.shape
        
        # Prepare for GCN: [batch, features, time] -> [batch, time, features]
        x = x.permute(0, 2, 1)  # [batch, reduced_timesteps, 32]
        x_flat = x.reshape(batch_size * reduced_timesteps, -1)
        
        # Build or retrieve graph
        cache_key = f"{reduced_timesteps}"
        if cache_key in self.graph_cache:
            edge_index = self.graph_cache[cache_key]
        else:
            try:
                # Use MEAN features across batch for representativeness
                mean_features = x.mean(dim=0).detach().cpu().numpy()
                edge_index = self.graph_builder.build_graph(mean_features)
                
                # Validate and clamp
                max_index = reduced_timesteps - 1
                if torch.any(edge_index > max_index):
                    print(f"Edge index contains out-of-bounds indices! Clamping to [0, {max_index}]")
                    edge_index = torch.clamp(edge_index, 0, max_index)
                
                self.graph_cache[cache_key] = edge_index
                print(f"Built new graph with {edge_index.shape[1]} edges for {reduced_timesteps} time steps")
                
            except Exception as e:
                print(f"Graph building failed: {e}, using chain fallback")
                edge_index = self._create_chain_graph(reduced_timesteps)
                self.graph_cache[cache_key] = edge_index
        
        # Move to device and clone to avoid warnings
        edge_index = self.graph_cache[cache_key].to(x.device).clone().detach()
        
        # Batch the graph with node offsetting
        edge_indices = []
        for i in range(batch_size):
            offset = i * reduced_timesteps
            edge_index_offset = edge_index + offset
            edge_indices.append(edge_index_offset)
        edge_index = torch.cat(edge_indices, dim=1)
        
        # Validate final edge indices
        max_index = batch_size * reduced_timesteps - 1
        if torch.any(edge_index > max_index):
            print(f"Final edge index contains out-of-bounds indices! Clamping to [0, {max_index}]")
            edge_index = torch.clamp(edge_index, 0, max_index)
        
        # Graph convolution
        x = F.relu(self.gcn1(x_flat, edge_index))
        x = F.relu(self.gcn2(x, edge_index))
        
        # Reshape back: [batch, reduced_timesteps, features]
        x = x.reshape(batch_size, reduced_timesteps, -1)
        
        # Global pooling over time
        x = torch.mean(x, dim=1)  # [batch, features]
        return self.fc(x)

    def reconstruct(self, features):
        """Reconstruct mean input features for pretraining"""
        return self.recon(features)
