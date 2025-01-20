import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter
import pytorch_lightning as pl  # For training wrapper
from sklearn.metrics import roc_auc_score, average_precision_score
import pyg_gnn_wrapper as gnn_wrapper

class AttentionLayer(nn.Module):
    def __init__(self, in_features):
        super(AttentionLayer, self).__init__()
        self.W = nn.Linear(in_features, in_features)

    def forward(self, h):
        Wh = self.W(h)
        e = torch.matmul(Wh, Wh.t())
        attention_weights = F.softmax(e / (Wh.size(-1) ** (1 / 2)), dim=-1)  # Scaled attention weights
        return torch.matmul(attention_weights, h)

class DeepSentinelGNN(pl.LightningModule):
    def __init__(self, model, estimation_layer, learning_rate=0.001, weight_decay=5e-4):
        super().__init__()
        self.model = model
        self.estimation_layer = estimation_layer
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.validation_step_outputs = []
        self.test_step_outputs = []

    def forward(self, data):
        raise NotImplementedError

    def validation_step(self, batch, batch_idx):
        if self.current_epoch > 0:
            output = (self(batch)[0], batch.y)
            self.validation_step_outputs.append(output)
            return output

    def on_validation_epoch_end(self):
        if self.current_epoch > 0:
            anomaly_scores = torch.cat([out[0] for out in self.validation_step_outputs]).cpu().detach()
            ys = torch.cat([out[1] for out in self.validation_step_outputs]).cpu().detach()
            metrics = {'val_roc_auc': roc_auc_score(ys, anomaly_scores), 'val_pr_auc': average_precision_score(ys, anomaly_scores)}
            self.log_dict(metrics)  # Use self.log_dict to log metrics properly
            self.validation_step_outputs.clear()  # Use self to access the list

    def test_step(self, batch, batch_idx):
        output = (self(batch)[0], batch.y)
        self.test_step_outputs.append(output)
        return output

    def on_test_epoch_end(self):
        anomaly_scores = torch.cat([out[0] for out in self.test_step_outputs]).cpu().detach()
        ys = torch.cat([out[1] for out in self.test_step_outputs]).cpu().detach()
        metrics = {'roc_auc': roc_auc_score(ys, anomaly_scores), 'pr_auc': average_precision_score(ys, anomaly_scores)}
        self.log_dict(metrics)  # Use self to log metrics properly
        self.test_step_outputs.clear()  # Use self to access the list

    def configure_optimizers(self):
        return torch.optim.Adam(list(self.model.parameters()) + list(self.estimation_layer.parameters()), 
                                 lr=self.learning_rate,
                                 weight_decay=self.weight_decay)  # Corrected parameter name here

class SentinelGNN(DeepSentinelGNN):
    def __init__(self, nfeat_node, nfeat_edge,
                 nhid=32,
                 nlayer=3,
                 dropout=0,
                 learning_rate=0.001,
                 weight_decay=0,
                 lambda_energy=0.1,
                 lambda_cov_diag=0.0005,
                 lambda_recon=0,
                 lambda_diversity=0,
                 lambda_entropy=0,
                 k_cls=2,
                 dim_metadata=None,
                 **kwargs):

        model = MultigraphGNNWrapper(nfeat_node, nfeat_edge, nhid, nhid, nlayer, dim_metadata=dim_metadata, dropout=dropout)
        estimation_layer = MLN(nhid, nhid, k_cls)

        super().__init__(model, estimation_layer, learning_rate, weight_decay)  # Call parent constructor correctly

        self.save_hyperparameters()  # Save hyperparameters for logging

        # Additional parameters for GMM and energy calculations
        self.radius = 0
        self.nu = 1
        self.eps = 1e-12

        # Buffers to store GMM parameters
        self.register_buffer("phi", torch.zeros(k_cls,))
        self.register_buffer("mu", torch.zeros(k_cls, nhid))

        # Loss weights
        self.lambda_energy = lambda_energy
        self.lambda_recon = lambda_recon
        self.lambda_diversity = lambda_diversity
        self.lambda_entropy = lambda_entropy

    def get_hiddens(self, data):
        return self.model(data)

    def get_means(self):
        return self.mu

    def get_validation_score(self, data):
        s_e, _, _, _ = self.forward(data)
        return torch.mean(s_e).item()

    def forward(self, data):
        if self.metadata_dim is None:
            embs = self.model(data)
        else:
            embs, logits = self.model(data)

        # Step1: From embs we get gamma
        gamma = self.estimation_layer(embs)

        # Step 2: We calculate mu,sigma
        if self.training:
            phi, mu = self.compute_gmm_params(embs,gamma)
        else:
            phi = self.phi
            mu = self.mu

        # Step 3: We estimate sample energy
        sample_energy = self.compute_energy(embs, phi, mu, gamma)

        if self.metadata_dim is None:
            return sample_energy, 0, gamma, mu
        else:
            return sample_energy, logits, gamma, mu

    def compute_gmm_params(self, z, gamma):
        N = gamma.size(0)
        
        phi = torch.sum(gamma , dim=0) + self.eps
        
        mu = torch.sum(gamma.unsqueeze(-1) * z.unsqueeze(1), dim=0) / phi.unsqueeze(-1)
        
        phi /= N
        
        # Register to buffer for forward pass with torch.no_grad():
        with torch.no_grad():
            self.mu = mu
            self.phi = phi
        
        return phi , mu

    def compute_energy(self , z , phi , mu , gamma , size_average=False):
       E = torch.cdist(z , mu).pow(2) * gamma
        
       E = torch.sum(E , dim=1 , keepdim=True)
        
       return E

    def training_step(self, batch, batch_idx):
       sample_energy , logits , gamma , mu = self(batch)

       if self.metadata_dim is None:
           clip_loss = 0.
       else:
           with torch.no_grad():
               N = logits.shape[0]
               labels =(2 *torch.eye(N) -torch.ones((N,N))).to(logits.device)
               clip_loss=-torch.sum(self.log_sigmoid(labels * logits)) / N

       if (self.k_cls > 1) and (self.lambda_diversity > 0):
           diversity_loss=-torch.logdet(torch.cov(mu))
           entropy_loss=-torch.mean((gamma *torch.log2(gamma)).sum(dim=1))
       elif (self.k_cls > 1) and (self.lambda_diversity == 0):
           entropy_loss=-torch.mean((gamma *torch.log2(gamma)).sum(dim=1))
           diversity_loss=0.
       else:
           diversity_loss=0.
           entropy_loss=0.

       loss=self.lambda_energy *torch.mean(sample_energy) +self.lambda_recon *clip_loss
       loss +=self.lambda_diversity *diversity_loss +self.lambda_entropy *entropy_loss

       with torch.no_grad():
           print("Loss: ", loss.item())
           print("Sample energy: ",self.lambda_energy*torch.mean(sample_energy).item())

           if (self.k_cls > 1):
               print("Diversity loss: ",self.lambda_diversity*diversity_loss)
               print("Entropy loss:",self.lambda_entropy*entropy_loss)

       return loss

    def log_sigmoid(self,x):
       return torch.clamp(x,max=0)-torch.log(torch.exp(-torch.abs(x))+1)

class BaseGNN(nn.Module):
    def __init__(self, nin, nout, nlayer, dropout=0, gcn_type='GINEConv'):
        super().__init__()

        # Initialize convolutional layers based on specified GNN type.
        self.convs = nn.ModuleList([getattr(gnn_wrapper, gcn_type)(nin, nin, bias=True) for _ in range(nlayer)])

        # Initialize normalization layers (BatchNorm).
        self.norms = nn.ModuleList([nn.BatchNorm1d(nin) for _ in range(nlayer)])

        # Set dropout rate.
        self.dropout = dropout

    def reset_parameters(self):
        # Reset parameters for convolutional and normalization layers.
        for conv, norm in zip(self.convs, self.norms):
            conv.reset_parameters()
            norm.reset_parameters()

    def forward(self, x=None, edge_index=None, batch=None):  # Corrected parameter definition
        previous_x = x

        for layer, norm in zip(self.convs, self.norms):
            x = layer(x=x, edge_index=edge_index, edge_attr=batch)  # Corrected parameter passing
            x = norm(x)
            x = F.dropout(x, self.dropout, self.training)
            x = x + previous_x
            previous_x = x

        x = F.relu(x)

        return x

class MultigraphGNNWrapper(nn.Module):
    def __init__(self, nfeat_node, nfeat_edge, nhid, nout, nlayer, dim_metadata=None, dropout=0) -> None:
        super().__init__()

        # Initialize node encoder based on input features or use DiscreteEncoder.
        if nfeat_node is None:
            max_num_values = 500
            node_encoder = self.node_encoder(nhid, max_num_values=max_num_values)
        else:
            node_encoder = self.node_encoder(nfeat_node, nhid, nlayer=1, with_final_activation=False)

        # Initialize edge encoders.
        edge_direction_encoder = self.edge_direction_encoder(nhid, max_num_values=4)

        edge_attr_encoder = self.edge_attr_encoder(nhid, max_num_values=10) if nfeat_edge is None else MLP(nfeat_edge, nhid, nlayer=3, with_final_activation=False)

        edge_transform = self.edge_transform(nhid, nhid, nlayer=1, with_final_activation=False)

        conv_model = self.conv_model(nhid, nhid, nlayer, dropout)

        output_encoder = self.output_encoder(nhid, nout, nlayer=2, with_final_activation=False, with_norm=True)

        graph_metadata_encoder = self.graph_metadata_encoder(nout, dim_metadata, nout, nout, nlayer=2) if dim_metadata else None

        self.reset_parameters()

    def reset_parameters(self):
        node_encoder.reset_parameters()
        
        edge_attr_encoder.reset_parameters()
        
        edge_direction_encoder.reset_parameters()
        
        edge_transform.reset_parameters()
        
        conv_model.reset_parameters()
        
        output_encoder.reset_parameters()

        if graph_metadata_encoder is not None:
            graph_metadata_encoder.reset_parameters()

    def forward(self,data):
        x=self.node_encoder(data.x)

        edge_attr=data.edge_index.new_zeros(data.edge_index.size(-1)) if data.edge_attr is None else data.edge_attr

        edge_attr=edge_attr if edge_attr.dim() > 1 else edge_attr.unsqueeze(-1)

        edge_attr=edge_attr +self.edge_attr_encoder(edge_attr) +self.edge_direction_encoder(data.edge_direction)

        simplified_edge_attr=scatter(edge_attr,index=data.simplified_edge_batch , dim=0 , reduce='add')

        simplified_edge_attr=edge_transform(simplified_edge_attr)

        x=self.conv_model(x,data.simplified_edge_index,simplified_edge_attr,data.batch)

        x=scatter(x,data.batch,dimsion_0,reduced='add')

        x=self.output_encoder(x)

        if graph_metadata_encoder is not None:
            x_logits=self.graph_metadata_encoder(x,data.metadata)
            return x,x_logits

        return x

if __name__ == '__main__':
    print("code for SentinelGNN")