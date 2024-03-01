import torch.nn as nn
import torch.nn.functional as F

class Projection_Layer(nn.Module):
    def __init__(self, num_patches,patch_size,in_channels, embed_size):
        super(Projection_Layer, self).__init__()


        self.num_patches = num_patches
        self.layer_norm_1 = nn.LayerNorm(patch_size*patch_size*in_channels)
        self.embed_layer = nn.Linear(patch_size*patch_size*in_channels,embed_size)
        self.layer_norm_2 = nn.LayerNorm(embed_size)

    def forward(self, x):
        b,_,_,_ = x.shape
        x = x.view(b,self.num_patches,-1)
        x = self.layer_norm_1(x)
        x = self.embed_layer(x)
        x = self.layer_norm_2(x)

        return x

class Attention(nn.Module):
    def __init__(self, embed_size):
        super(Attention, self).__init__()

        self.Q_matrix = nn.Linear(embed_size,embed_size)
        self.K_matrix = nn.Linear(embed_size,embed_size)
        self.V_matrix = nn.Linear(embed_size,embed_size)
        self.temperature = embed_size**0.5

    def forward(self, x):
        q = self.Q_matrix(x)
        k = self.K_matrix(x)
        v = self.V_matrix(x)

        attention = nn.Softmax(dim=-1)(torch.matmul(q,k.transpose(-1,-2)))/self.temperature
        feature = torch.matmul(attention,v)
        return {
            "attention":attention,
            "feature_vector": feature,
        }
    
class Transformer_Block(nn.Module):
    def __init__(self, embed_size,num_heads):
        super(Transformer_Block, self).__init__()
        
        self.num_heads = num_heads
        self.norm = nn.LayerNorm(embed_size)
        self.attention = Attention(embed_size//num_heads)

    def forward(self, x):
        bs,n_1,embed_dim = x.shape
        x = self.norm(x)
        
        x = x.view(bs,self.num_heads,n_1,embed_dim//self.num_heads)  ## B, head, 256, 192//head
        x = self.attention(x)["feature_vector"]
        x = x.view(bs,n_1,embed_dim)

        return x
    
class Vision_Transformer(nn.Module):
    def __init__(self, image_size=256,in_channels=3, patch_size = 16, embed_size=192,hidden_dim=512, num_heads = 8,num_layers=4,dropout=0.01):
        super(Vision_Transformer, self).__init__()

        self.num_patches = (image_size//patch_size)**2
        self.embed_size = embed_size
        self.num_heads = num_heads
        self.projection_layer = Projection_Layer(self.num_patches,patch_size,in_channels, embed_size)

        self.cls_token = nn.Parameter(torch.randn(1, 1, self.embed_size))
        self.pos_emb = nn.Parameter(torch.randn(1, self.num_patches + 1, self.embed_size))

        self.layers = nn.ModuleList([])
        self.feed_forward = nn.Sequential(
            nn.LayerNorm(embed_size),
            nn.Linear(embed_size, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_size),
            nn.Dropout(dropout)
        )
        for _ in range(num_layers):
            self.layers.append(nn.Sequential(
                Transformer_Block(embed_size,num_heads),
                self.feed_forward    
            ))

        self.clf_head = nn.Linear(embed_size,10)

    def forward(self, x):
        bs,_,_,_ = x.shape
        x = self.projection_layer(x) ## B, 256, 192

        cls_token = torch.cat(bs*[self.cls_token], dim=0) ## broadcasting
        x = torch.concat([cls_token,x],dim=1)
        
        x = x + self.pos_emb

        for layer in self.layers:
            x = layer(x)

        x = self.clf_head(x[:,0,:])
        return x
