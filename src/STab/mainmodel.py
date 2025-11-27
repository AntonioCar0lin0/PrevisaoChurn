import torch
import torch.nn as nn
import torch.nn.functional as F

class MainModel(nn.Module):
    def __init__(self, categories, num_continuous, dim, dim_out, depth, heads, attn_dropout, ff_dropout, U=1, cases=None):
        """
        Arquitetura Baseada em Transformer para Dados Tabulares (STab/FT-Transformer).
        
        Args:
            categories (tuple): Lista com a cardinalidade (qtd valores unicos) de cada coluna categórica.
            num_continuous (int): Quantidade de colunas numéricas.
            dim (int): Tamanho do vetor de embedding (d_model).
            dim_out (int): Tamanho da saída (2 para classificação binária).
            depth (int): Quantidade de camadas Transformer (Blocks).
            heads (int): Quantidade de cabeças de atenção (Multi-Head Attention).
            attn_dropout (float): Dropout na atenção.
            ff_dropout (float): Dropout na camada Feed Forward.
        """
        super(MainModel, self).__init__()
        
        self.num_continuous = num_continuous
        self.dim = dim
        
        # 1. Embeddings para Categóricas
        # Cria uma lista de embeddings, um para cada coluna categórica
        self.cat_embeddings = nn.ModuleList([
            nn.Embedding(num_classes, dim) for num_classes in categories
        ])
        
        # 2. Projeção Linear para Numéricas (Numerical Embedding)
        # Transforma cada valor numérico escalar em um vetor de tamanho 'dim'
        if num_continuous > 0:
            self.num_projection = nn.Linear(1, dim)
            # Token CLS (opcional em algumas arquiteturas, aqui usamos concatenação/pooling simples ou CLS aprendido)
            self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        
        # 3. Transformer Encoder (O "Cérebro")
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=heads,
            dim_feedforward=dim * 4, # Padrão geralmente é 4x o dim
            dropout=attn_dropout,
            activation="relu",
            batch_first=True # Importante: (Batch, Seq_Len, Dim)
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        
        # 4. Cabeçalho Final (MLP Head)
        # Pega a saída do transformer e decide a classe
        # O tamanho da entrada do MLP depende da estratégia (CLS token ou Flatten).
        # Aqui usaremos o CLS token strategy (tamanho dim)
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim // 2),
            nn.ReLU(),
            nn.Dropout(ff_dropout),
            nn.Linear(dim // 2, dim_out)
        )

    def forward(self, x_num, x_cat):
        """
        Args:
            x_num: Tensor (Batch, Num_Continuous)
            x_cat: Tensor (Batch, Num_Categorical) - Inteiros
        """
        batch_size = x_num.shape[0]
        
        # --- Passo 1: Tokenização (Embedding) ---
        embeddings = []
        
        # Processa Numéricas
        if self.num_continuous > 0:
            # x_num entra como (Batch, N_cols). Transformamos em (Batch, N_cols, 1)
            x_num_expanded = x_num.unsqueeze(-1)
            # Projeta para (Batch, N_cols, Dim)
            num_emb = self.num_projection(x_num_expanded)
            embeddings.append(num_emb)
        
        # Processa Categóricas
        if len(self.cat_embeddings) > 0:
            cat_embs = []
            for i, emb_layer in enumerate(self.cat_embeddings):
                # Pega a coluna i de todas as linhas: x_cat[:, i]
                c = x_cat[:, i] 
                # Embed: (Batch, Dim) -> Unsqueeze para (Batch, 1, Dim)
                cat_embs.append(emb_layer(c).unsqueeze(1))
            
            # Concatena todas as categóricas: (Batch, N_cat, Dim)
            embeddings.append(torch.cat(cat_embs, dim=1))
            
        # Adiciona o CLS Token no início (Token especial que resume a linha inteira)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        
        # Junta tudo na dimensão da sequência (Seq_Len)
        # Formato final: (Batch, N_num + N_cat + 1, Dim)
        if embeddings:
            x = torch.cat([cls_tokens] + embeddings, dim=1)
        else:
            x = cls_tokens
            
        # --- Passo 2: Transformer ---
        # A mágica acontece aqui: as colunas "conversam" entre si via Atenção
        x = self.transformer(x)
        
        # --- Passo 3: Predição ---
        # Pegamos apenas o vetor correspondente ao CLS Token (índice 0) para classificar
        cls_output = x[:, 0, :] 
        
        logits = self.mlp_head(cls_output)
        return logits


class Num_Cat(nn.Module):
    def __init__(self, model, num_number, classes, Sample_size):
        """
        Wrapper (Adaptador) para integrar com Keras4Torch.
        Ele recebe a lista mista [X_num, X_cat] e passa corretamente para o MainModel.
        """
        super(Num_Cat, self).__init__()
        self.model = model
        
    def forward(self, inputs):
        # inputs vem do Keras4Torch como uma lista: [Tensor_Num, Tensor_Cat]
        x_num = inputs[0]
        x_cat = inputs[1]
        
        return self.model(x_num, x_cat)
    

# Importância do arquivo:     