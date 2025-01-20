import torch
import torch.nn as nn
import torch_geometric.nn as gnn
import torch.nn.functional as F
from torch_scatter import scatter
from torch_geometric.utils import degree

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, embed_size):
        super(MultiHeadAttention, self).__init__()

        # Initialize parameters for multi-head attention.
        self.heads = num_heads  
        self.embed_size = embed_size  
        self.values = nn.Linear(embed_size, embed_size, bias=False)
        self.keys = nn.Linear(embed_size, embed_size, bias=False)
        self.queries = nn.Linear(embed_size, embed_size, bias=False)
        self.fc_out = nn.Linear(embed_size, embed_size)

    def forward(self, x):  
        N = x.shape[0]  
        value_len, key_len, query_len = x.shape[1], x.shape[1], x.shape[1]  

        values = self.values(x)  
        keys = self.keys(x)  
        queries = self.queries(x)  

        values = values.view(N, value_len, self.heads, self.embed_size // self.heads)  
        keys = keys.view(N, key_len, self.heads, self.embed_size // self.heads)  
        queries = queries.view(N, query_len, self.heads, self.embed_size // self.heads)  

        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])  
        attention = F.softmax(energy / (self.embed_size ** (1 / 2)), dim=3)  

        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(N, query_len, -1)  

        return self.fc_out(out)

class GINEConv(nn.Module):
    def __init__(self, nin, nout, bias=True):
        super().__init__()
        self.nn = MLP(nin, nout, 2, False, bias=bias)
        self.layer = gnn.GINEConv(self.nn, train_eps=True)
        self.edge_linear = nn.Linear(nin, nin)

    def reset_parameters(self):
        self.nn.reset_parameters()
        self.layer.reset_parameters()

    def forward(self, x, edge_index, edge_attr):
        edge_attr = self.edge_linear(edge_attr)  # Update edge attributes
        return self.layer(x, edge_index, edge_attr)

class SetConv(nn.Module):
    def __init__(self, nin, nout, bias=True):
        super().__init__()
        self.nn = MLP(nin, nout, 2, False)
        self.linear = nn.Linear(nin, nin, bias=False)
        self.bn = nn.BatchNorm1d(nin)

    def reset_parameters(self):
        self.nn.reset_parameters()
        self.bn.reset_parameters()

    def forward(self, x, edge_index, edge_attr, batch):
        summation = scatter(x, batch, dim=0)  # Aggregate node features
        summation = self.linear(summation)
        summation = self.bn(summation)
        summation = F.relu(summation)
        
        return self.nn(x + summation[batch])  # Combine with original features

class GATConv(nn.Module):
    def __init__(self, nin, nout, bias=True, nhead=1):
        super().__init__()
        self.layer = gnn.GATConv(nin, nout // nhead , nhead , bias=bias)

    def reset_parameters(self):
        self.layer.reset_parameters()

    def forward(self,x , edge_index , edge_attr):
        return self.layer(x , edge_index)

class GCNConv(nn.Module):
    def __init__(self,nin,nout,bias=True):
        super().__init__()
        # Initialize GCN layer
        self.layer = gnn.GCNConv(nin,nout,bias=bias)

    def reset_parameters(self):
         # Reset parameters for GCN layer
         self.layer.reset_parameters()

    def forward(self,x , edge_index , edge_attr=None):
         return self.layer(x , edge_index)

# Define general message passing layers like DGN's paper or use the meta-layer
class SimplifiedPNAConv(gnn.MessagePassing):
    def __init__(self,nin,nout,bias=True,
                 aggregators=['mean', 'min', 'max', 'std'], **kwargs): 
                 
                 kwargs.setdefault('aggr', None)
                 super().__init__(node_dim=0 , **kwargs)
        
                 # Initialize parameters
                 self.aggregators = aggregators
                 self.pre_nn = MLP(3 * nin, nin, 2, False)
                 self.post_nn = MLP((len(aggregators) + 1 + 1) * nin , nout , 2 , False , bias=bias)
                 # Edge degree embedding
                 self.deg_embedder = nn.Embedding(200 , nin) 

    def reset_parameters(self):
        # Reset parameters for pre and post neural networks and degree embedder
        self.pre_nn.reset_parameters()
        self.post_nn.reset_parameters()
        self.deg_embedder.reset_parameters()

    def forward(self,x , edge_index , edge_attr , batch=None):
         out=self.propagate(edge_index,x=x , edge_attr=edge_attr)
         out=torch.cat([x,out],dim=-1)
         out=self.post_nn(out) 
         return out

    def message(self,x_i,x_j=edge_attr=None):
         if edge_attr is not None:
            h=torch.cat([x_i,x_j=edge_attr],dim=-1)
         else:
            h=torch.cat([x_i,x_j],dim=-1)
         return self.pre_nn(h)

    def aggregate(self , inputs,index , dim_size=None):
         outs=[]
         for aggregator in self.aggregators:
             if aggregator == 'sum':
                 out=scatter(inputs,index ,0,None ,dim_size ,reduce='sum')
             elif aggregator == 'mean':
                 out=scatter(inputs,index ,0,None ,dim_size ,reduce='mean')
             elif aggregator == 'min':
                 out=scatter(inputs,index ,0,None ,dim_size ,reduce='min')
             elif aggregator == 'max':
                 out=scatter(inputs,index ,0,None ,dim_size ,reduce='max')
             elif aggregator == 'var' or aggregator == 'std':
                 mean=scatter(inputs,index ,0,None ,dim_size,reduced='mean')
                 mean_squares=scatter(inputs * inputs,index ,0,None ,dim_size,reduced='mean')
                 out=mean_squares - mean * mean
                 if aggregator == 'std':
                     out=torch.sqrt(F.relu_(out)+1e-5)
             else:
                 raise ValueError(f'Unknown aggregator "{aggregator}".')  
             outs.append(out)

         outs.append(self.deg_embedder(degree(index,dim_size,dtype=index.dtype)))
         out=torch.cat(outs,dim=-1)

         return out

# Next test PNA without any normalization layer
class GINEDegConv(gnn.MessagePassing):
    def __init__(self,nin,nout,bias=True , **kwargs):
       kwargs.setdefault('aggr', None)
       super().__init__(node_dim=0 , **kwargs)
       # Initialize neural networks and parameters
       self.nn=MLP(3*nin,nout ,2 ,False,bias=bias)
       # Degree embedding for nodes
       self.deg_embedder=nn.Embedding(200,nin) 

       # Reset parameters method to initialize weights
       def reset_parameters(self):
           # Reset parameters for neural network and degree embedder
           self.nn.reset_parameters()
           # Initialize epsilon parameter for residual connections
           if not hasattr(self,'eps'):
               raise ValueError("Epsilon parameter not initialized.")
           else:
               eps.data.fill_(0)
           # Reset degree embedder parameters
           if hasattr(self,'deg_embedder'):
               deg_embedder.reset_parameters()

   def forward(self,x=edge_index=None,batch=None): 
       out,degs=self.propagate(edge_index,x=x )
       out=(1+self.eps)*x + deg +self.linear1(out)*self.linear2(deg)

       return out
    
   def message(self,x_j=edge_attr=None): 
       return (x_j +edge_attr).relu()

   def aggregate(self,input,index=None): 
       out=scatter(inputs,index=0,None,dimsize=dim_size,reduced='mean')
       deg=self.deg_embedder(degree(index,dimsizes=dtype=index.dtype))
       
       return out,degs

if __name__ == '__main__':
   print("code for SentinelGNN")