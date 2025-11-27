import torch
import torch.nn as nn
import torch.nn.functional as F

class MainModel(nn.Module):
    def __init__(self, categories, num_continuous, dim, dim_out, depth, heads, attn_dropout, ff_dropout, U=1, cases=None):
        super(MainModel, self).__init__()
        
        self.num_continuous = num_continuous
        self.dim = dim
        
        # 1. Embeddings para Categóricas
        self.cat_embeddings = nn.ModuleList([
            nn.Embedding(num_classes, dim) for num_classes in categories
        ])
        
        # 2. Projeção Linear para Numéricas
        if num_continuous > 0:
            self.num_projection = nn.Linear(1, dim)
            self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        
        # 3. Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=heads,
            dim_feedforward=dim * 4,
            dropout=attn_dropout,
            activation="relu",
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        
        # 4. Cabeçalho Final (MLP Head)
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim // 2),
            nn.ReLU(),
            nn.Dropout(ff_dropout),
            nn.Linear(dim // 2, dim_out)
        )

    def forward(self, x_num, x_cat):
        batch_size = x_num.shape[0]
        embeddings = []
        
        # --- Processa Numéricas ---
        if self.num_continuous > 0:
            if x_num.dtype != torch.float32:
                x_num = x_num.float()
            
            x_num_expanded = x_num.unsqueeze(-1)
            num_emb = self.num_projection(x_num_expanded)
            embeddings.append(num_emb)
        
        # --- Processa Categóricas ---
        if len(self.cat_embeddings) > 0:
            cat_embs = []
            for i, emb_layer in enumerate(self.cat_embeddings):
                # Conversão para Long garantida aqui também
                c = x_cat[:, i].long()
                cat_embs.append(emb_layer(c).unsqueeze(1))
            
            embeddings.append(torch.cat(cat_embs, dim=1))
            
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        
        if embeddings:
            x = torch.cat([cls_tokens] + embeddings, dim=1)
        else:
            x = cls_tokens
            
        x = self.transformer(x)
        cls_output = x[:, 0, :] 
        logits = self.mlp_head(cls_output)
        return logits


class Num_Cat(nn.Module):
    def __init__(self, model, num_number, classes, Sample_size):
        super(Num_Cat, self).__init__()
        self.model = model
        
    def forward(self, *args):
        # --- SOLUÇÃO UNIVERSAL ---
        # O *args captura tudo o que for enviado.
        
        if len(args) == 1:
            # Caso 1: Recebeu uma lista empacotada [[num, cat]]
            inputs = args[0]
            x_num = inputs[0]
            x_cat = inputs[1]
        else:
            # Caso 2: Recebeu argumentos separados (num, cat)
            x_num = args[0]
            x_cat = args[1]
            
        # Garante o tipo Long para categóricas antes de passar pro modelo
        return self.model(x_num, x_cat.long())