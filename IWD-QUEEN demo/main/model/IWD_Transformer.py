import torch
import torch.nn as nn

class PatchEmbedding(nn.Module):
    def __init__(self, img_height=17, img_width=11, patch_size=2, in_channels=1, embed_dim=64):
        super().__init__()
        self.num_patches_h = img_height // patch_size  
        self.num_patches_w = img_width // patch_size  
        self.num_patches = self.num_patches_h * self.num_patches_w

        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=(patch_size, patch_size), stride=(patch_size, patch_size))

    def forward(self, x):
        x = x.reshape(x.shape[0], 1, x.shape[1], x.shape[2])
        x = self.proj(x)  
        x = x.flatten(2)  
        x = x.transpose(1, 2)  
        return x
    
class CLSOnlyAttention(nn.Module):
    def __init__(self, embed_dim=64, num_heads=8, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.kv_proj = nn.Linear(embed_dim, embed_dim * 2, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, N, C = x.shape

        cls_token = x[:, 0:1, :]  
        q = self.q_proj(cls_token)  

        kv_input = x[:, 1:]  
        kv = self.kv_proj(kv_input) 
        k, v = kv.chunk(2, dim=-1) 

        def reshape(t): return t.reshape(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        q = reshape(q)  
        k = reshape(k)  
        v = reshape(v)

        attn = (q @ k.transpose(-2, -1)) * self.scale 
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, 1, C) 
        out = self.out_proj(out)
        return out  
    
class CoreNet(nn.Module):
    def __init__(self,embed_dim=64,num_heads=8,dropout=0.1):
        super().__init__()
        self.device= torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.patch_emdedding = PatchEmbedding(17, 11, 2, 1, embed_dim)
        num_patches = self.patch_emdedding.num_patches

        self.attention = CLSOnlyAttention(embed_dim, num_heads)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches + 1, embed_dim))

        self.center_par = nn.Parameter(torch.tensor(0.5))

        mlp_dim = 4 * embed_dim

        self.center_proj = nn.Sequential(
            nn.Flatten(),  
            nn.Linear(15, embed_dim),  
            nn.GELU()
        )

        self.mlp_transformer = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, embed_dim),
            nn.Dropout(dropout),
        )

        self.mlp_mapping = nn.Sequential(
            nn.Linear(16, 16),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
        
        self.CONV2D =  nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, groups=16),
            nn.GELU(),
            nn.Conv2d(16, 16, kernel_size=4, stride=1, padding=1, groups=16),
            nn.GELU(),
        )  

    def forward(self, x):
        x_center = x[:, 6:11, 4:7] 
        x_center = x_center.unsqueeze(1) 
        center_emb = self.center_proj(x_center) 

        x = self.patch_emdedding(x)
        batch_size = x.shape[0]

        center_par = torch.sigmoid(self.center_par)
        x_cls = (1 - center_par) * self.cls_token.expand(batch_size, -1, -1) + center_par * center_emb.unsqueeze(1)
        
        x = torch.cat((x_cls, x), dim=1)
        x = x + self.pos_embed

        x_cls = self.attention(self.norm1(x))  
        x = x.clone()
        x[:,0:1] = x[:,0:1] + x_cls

        x = x + self.mlp_transformer(self.norm2(x))

        x_cls = x[:, 0]  
        assert x_cls.shape[1] == 64, "Embed dim must be 64 to reshape to (16, 4)"
        
        x_cls_forCNN = x_cls.reshape(x_cls.size(0), 16, 2, 2)
        x = self.CONV2D(x_cls_forCNN) 
        x = x.view(x.size(0), -1) 

        x = self.mlp_mapping(x)

        return x
    
    
  
