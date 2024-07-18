import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from torch_geometric.nn import GCNConv


class Debias_net(nn.Module):
    def __init__(self, input_dim, output_dim, freq_dim):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.freq_dim = freq_dim
        self.gamma = 0.5

        # self.bi_attention = BiAttention(output_dim)
        self.freq_linear = nn.Linear(1, 2, bias=False)
        self.main_mlp = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ELU(),
            # nn.Linear(output_dim, output_dim),
        )
        self.other_mlp = nn.Sequential(
            nn.Linear(input_dim*2, output_dim),
            nn.ELU(),
            # nn.Linear(output_dim, output_dim),
        )

    def forward(self, traj, main_data, cat_data, time_data, user_data, main_freq):
        freq_matrix = main_freq.unsqueeze(0).expand(traj.size(0), -1)
        freq_gate = torch.gather(freq_matrix, dim=1, index=traj).unsqueeze(-1)

        # main_data_processed = self.main_mlp(main_data)
        # other_data = torch.stack((cat_data, time_data, user_data), dim=2)
        # other_data = self.other_mlp(other_data)
        # weight = F.softmax((self.freq_linear(1 - freq_gate)), dim=-1)
        # weighted_other_data = other_data * weight.unsqueeze(-1)
        # aggregated_data = weighted_other_data.sum(dim=2)
        # other_data = self.other_mlp(other_data.sum(2))

        # out_data = self.gamma * aggregated_data + main_data_processed

        main_data = self.main_mlp(main_data)

        weight = torch.pow(torch.sigmoid(self.freq_linear(1 - freq_gate)), self.gamma)  # 0.5 is best
        # weight = torch.softmax(self.freq_linear(1 - freq_gate),dim=-1) # 0.5 is best
        weight1 = weight[:, :, 0].unsqueeze(-1)
        weight2 = weight[:, :, 1].unsqueeze(-1)
        other_data = self.other_mlp(torch.cat([weight1 * cat_data, weight2 * time_data], dim=-1))

        out_data = main_data + other_data

        return out_data


class OneTower(nn.Module):
    def __init__(self, hidden_dim, drop_p):
        super().__init__()
        dot_head = 4
        add_head = 1
        # self.Gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.self_attention = MultiHeadedAttention(h=dot_head, in_model=hidden_dim, d_model=hidden_dim, dropout=drop_p)
        self.short_norm = nn.LayerNorm(hidden_dim)
        # ---------------------历史轨迹处理-------------------------
        # 从历史轨迹中提取隐藏状态
        self.add_attention = MultiHeadedAddAttention(
            h=add_head, q_model=hidden_dim, k_model=hidden_dim, v_model=hidden_dim, d_model=hidden_dim, dropout=drop_p)
        # additive attention ablation
        # self.poi_attnlayer = MultiHeadedAttention(h=add_head, in_model=hidden_dim, d_model=hidden_dim,
        #                                           dropout=drop_p, bias=False)
        self.long_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(drop_p)

    def forward(self, seq_emb, seq_mask, his_seq_emb, his_seq_mask):
        poi_attn = self.dropout(
            self.short_norm(self.self_attention(seq_emb, seq_emb, seq_emb, seq_mask))) * seq_mask.unsqueeze(-1)
        poi_hidden = poi_attn.mean(1).unsqueeze(1)

        his_poi_up = his_seq_emb
        his_poi_hidden = self.dropout(
            poi_hidden + self.long_norm(self.add_attention(poi_hidden, his_poi_up, his_poi_up, his_seq_mask)))
        attn_weight = self.self_attention.attn[:, 0, :, :]

        seq_hidden = torch.cat([poi_hidden, his_poi_hidden], dim=-1).squeeze(1)
        # seq_hidden = poi_hidden+his_poi_hidden
        return seq_hidden, attn_weight


class ManModel(nn.Module):
    def __init__(self, emb_dim, poi_num, hidden_dim, user_num, poi_emb, cat_emb,
                 drop_p=0.3, poi_pin=None, model_kwargs=None):
        super(ManModel, self).__init__()
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.user_num = user_num
        self.poi_pin = poi_pin
        self.poi_num = poi_num
        self.cat_num = cat_emb
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model_kwargs = model_kwargs
        #
        # dot_head = 4
        # add_head = 1

        # self.mem_emb = nn.Embedding(poi_num, emb_dim, padding_idx=0)  # Contrastive_emb ablation
        self.mem_emb = nn.Embedding.from_pretrained(poi_emb, padding_idx=0)
        self.time_one_hot = nn.Linear(poi_num, emb_dim)
        self.cat_one_hot = nn.Linear(poi_num, emb_dim)
        self.user_emb = nn.Embedding(user_num, emb_dim, padding_idx=0)

        self.debias_net = Debias_net(emb_dim, hidden_dim, self.poi_num)

        self.his_pos_emb = PositionalEncoding(d_model=hidden_dim, dropout=0, max_len=2000)

        self.poi_tower = OneTower(hidden_dim, drop_p)
        # self.cat_tower = OneTower(hidden_dim, drop_p)

        # self.Gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.cat_out = nn.Linear(hidden_dim * 2, self.cat_num)
        self.poi_out = nn.Linear(hidden_dim * 2, poi_num)

        # self.dropout = nn.Dropout(drop_p)
        # self._init_parameters()

    def _init_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.GRUCell):
                nn.init.xavier_normal_(m.weight_hh)
                nn.init.xavier_normal_(m.weight_ih)
                nn.init.constant_(m.bias_hh, 0)
                nn.init.constant_(m.bias_ih, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, traj, time, cat, length, user, his_traj, his_time, his_cat, his_length):
        # assert model_kwargs is not None
        user_embs, poi_embs, gen_embs = self.get_embedding(traj, cat, time, user, self.model_kwargs, dist='short')
        mask = traj.gt(0)
        his_mask = his_traj.gt(0)

        user_embs, his_poi_embs, his_gen_embs, = self.get_embedding(his_traj, his_cat, his_time, user, self.model_kwargs,
                                                                   dist='long')

        p_hidden, attn_weight = self.poi_tower(poi_embs, mask, his_poi_embs, his_mask)
        # c_hidden, _ = self.cat_tower(gen_embs, mask, his_gen_embs, his_mask)
        logit_poi = self.poi_out(p_hidden)
        logit_cat = self.cat_out(p_hidden)

        # logit_poi = self.out(self.user_out(p_hidden))

        out_cat = F.softmax(logit_cat, dim=-1)
        _, pred_cat10 = torch.topk(out_cat, dim=-1, k=10)

        return logit_poi, logit_cat, attn_weight, pred_cat10

    def get_embedding(self, traj, cat, time, user, model_kwargs, dist='short'):
        tp_one_hot = model_kwargs.get('tp_one_hot')
        ct_one_hot = model_kwargs.get('ct_one_hot')
        # time_embedding = torch.cat([torch.zeros(1, self.emb_dim).cuda(), self.time_one_hot(tp_one_hot)], dim=0)
        # cat_embedding = torch.cat([torch.zeros(1, self.emb_dim).cuda(), self.cat_one_hot(ct_one_hot)], dim=0)
        time_embedding = torch.tanh(self.time_one_hot(tp_one_hot))
        cat_embedding = torch.tanh(self.cat_one_hot(ct_one_hot))

        time_emb = nn.Embedding.from_pretrained(time_embedding, padding_idx=0)
        cat_emb = nn.Embedding.from_pretrained(cat_embedding, padding_idx=0)
        time_embs = time_emb(time)
        cat_embs = cat_emb(cat)

        # scores = torch.matmul(cat_emb.weight, self.mem_emb.weight.T)

        # 添加代码
        gen_embs = torch.cat([cat_embs, time_embs], dim=-1)

        if dist == 'short':
            mem_poi_embs = self.mem_emb(traj)
            # gen_poi_embs = self.gen_emb(traj)
        else:
            mem_poi_embs = self.mem_emb(traj)
            # gen_poi_embs = self.gen_emb(traj)

        user_embs = self.user_emb(user)

        # user_poi_scores = torch.matmul(self.user_emb.weight, self.mem_emb.weight.T)
        # user_cat_scores = torch.matmul(self.user_emb.weight, cat_emb.weight.T)
        user_embs = torch.broadcast_to(user_embs.unsqueeze(1), cat_embs.shape)
        poi_embs = self.debias_net(traj, mem_poi_embs, cat_embs, time_embs, user_embs, self.poi_pin)

        poi_embs = self.his_pos_emb(poi_embs)

        # 新增
        # gen_embs = self.his_pos_emb(gen_embs)

        return user_embs, poi_embs, gen_embs,


class MultiHeadedAddAttention(nn.Module):
    def __init__(self, h, q_model, k_model, v_model, d_model, dropout=0.1, bias=False):
        super(MultiHeadedAddAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.W_q = nn.Linear(q_model, d_model, bias=bias)
        self.W_k = nn.Linear(k_model, d_model, bias=bias)
        self.W_v = nn.Linear(v_model, d_model, bias=bias)
        self.W_o = nn.Linear(d_model, d_model, bias=bias)

        self.W_a = nn.Linear(d_model // h, 1, bias=bias)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        B = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [l(x).view(B, -1, self.h, self.d_k).transpose(1, 2) for l, x in
                             zip((self.W_q, self.W_k, self.W_v), (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = self.add_attention(query, key, value, self.W_a, mask=mask, dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(B, -1, self.h * self.d_k)  # contiguous()使得view不会改变数据顺序
        return self.W_o(x)

    def add_attention(self, query, key, value, w_a, mask=None, dropout=None):
        features = torch.tanh(query.unsqueeze(3) + key.unsqueeze(2))
        scores = w_a(features).squeeze(-1)
        if mask is not None:
            mask = torch.broadcast_to(mask.unsqueeze(2), scores.shape)
            scores = scores.masked_fill(mask == 0, -1e9)
        weights = F.softmax(scores, dim=-1)
        if dropout is not None:
            weights = dropout(weights)
        return torch.matmul(weights, value), weights


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, in_model, d_model, dropout=0.1, bias=True):
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(in_model, d_model, bias=bias), 3)
        self.out = nn.Linear(d_model, d_model)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        B = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [l(x).view(B, -1, self.h, self.d_k).transpose(1, 2) for l, x in
                             zip(self.linears, (query, key, value))]
        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = self.attention(query, key, value, mask=mask, dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(B, -1, self.h * self.d_k)
        return self.out(x)

    def attention(self, query, key, value, mask=None, dropout=None):
        d_k = torch.tensor(query.size(-1))
        scores = torch.matmul(query, key.transpose(-2, -1)) / torch.sqrt(d_k)  # 注意力计算公式
        if mask is not None:
            mask = torch.broadcast_to(mask.unsqueeze(2), scores.shape)
            scores = scores.masked_fill(mask == 0, -1e9)
        p_attn = F.softmax(scores, dim=-1)
        if dropout is not None:
            p_attn = dropout(p_attn)
        return torch.matmul(p_attn, value), p_attn


def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        # 建个arrange表示词的位置以便公式计算，size=(max_len,1)
        i_mat = torch.pow(10000, torch.arange(0, d_model, 2).reshape(1, -1) / d_model)
        # div_term = torch.exp(torch.arange(0, d_model, 2) *  # 计算公式中10000**（2i/d_model)
        #                      -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position / i_mat)
        pe[:, 1::2] = torch.cos(position / i_mat)
        pe = pe.unsqueeze(0)  # size=(1, L, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)].detach()  # size = [batch, L, d_model]
        return self.dropout(x)  # size = [batch, L, d_model]


class GCN(nn.Module):
    def __init__(self, in_feats, hidden_feats, out_feats, num_heads, drop_p):
        super(GCN, self).__init__()
        # self.num_heads = num_heads
        # self.conv1 = GATConv(in_feats, hidden_feats, num_heads)
        self.conv1 = GCNConv(in_feats, hidden_feats)
        # self.conv2 = GATConv(hidden_feats * num_heads, out_feats // num_heads, num_heads)
        self.conv2 = GCNConv(hidden_feats, out_feats)
        self.dropout = nn.Dropout(drop_p)

    def forward(self, node, edge):
        # node, edge = graph.data, graph.edge_index
        h = self.conv1(node, edge)
        h = self.dropout(F.relu(h))
        h = self.conv2(h, edge)
        return h