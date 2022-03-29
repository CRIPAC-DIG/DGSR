#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2021/11/17 3:29
# @Author : ZM7
# @File : DGSR
# @Software: PyCharm
import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class DGSR(nn.Module):
    def __init__(self, user_num, item_num, input_dim, item_max_length, user_max_length, feat_drop=0.2, attn_drop=0.2,
                 user_long='orgat', user_short='att', item_long='ogat', item_short='att', user_update='rnn',
                 item_update='rnn', last_item=True, layer_num=3, time=True):
        super(DGSR, self).__init__()
        self.user_num = user_num
        self.item_num = item_num
        self.hidden_size = input_dim
        self.item_max_length = item_max_length
        self.user_max_length = user_max_length
        self.layer_num = layer_num
        self.time = time
        self.last_item = last_item
        # long- and short-term encoder
        self.user_long = user_long
        self.item_long = item_long
        self.user_short = user_short
        self.item_short = item_short
        # update function
        self.user_update = user_update
        self.item_update = item_update

        self.user_embedding = nn.Embedding(self.user_num, self.hidden_size)
        self.item_embedding = nn.Embedding(self.item_num, self.hidden_size)
        if self.last_item:
            self.unified_map = nn.Linear((self.layer_num + 1) * self.hidden_size, self.hidden_size, bias=False)
        else:
            self.unified_map = nn.Linear(self.layer_num * self.hidden_size, self.hidden_size, bias=False)
        self.layers = nn.ModuleList([DGSRLayers(self.hidden_size, self.hidden_size, self.user_max_length, self.item_max_length, feat_drop, attn_drop,
                                                self.user_long, self.user_short, self.item_long, self.item_short,
                                                self.user_update, self.item_update) for _ in range(self.layer_num)])
        self.reset_parameters()

    def forward(self, g, user_index=None, last_item_index=None, neg_tar=None, is_training=False):
        feat_dict = None
        user_layer = []
        g.nodes['user'].data['user_h'] = self.user_embedding(g.nodes['user'].data['user_id'].cuda())
        g.nodes['item'].data['item_h'] = self.item_embedding(g.nodes['item'].data['item_id'].cuda())
        if self.layer_num > 0:
            for conv in self.layers:
                feat_dict = conv(g, feat_dict)
                user_layer.append(graph_user(g, user_index, feat_dict['user']))
            if self.last_item:
                item_embed = graph_item(g, last_item_index, feat_dict['item'])
                user_layer.append(item_embed)
        unified_embedding = self.unified_map(torch.cat(user_layer, -1))
        score = torch.matmul(unified_embedding, self.item_embedding.weight.transpose(1, 0))
        if is_training:
            return score
        else:
            neg_embedding = self.item_embedding(neg_tar)
            score_neg = torch.matmul(unified_embedding.unsqueeze(1), neg_embedding.transpose(2, 1)).squeeze(1)
            return score, score_neg

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        for weight in self.parameters():
            if len(weight.shape) > 1:
                nn.init.xavier_normal_(weight, gain=gain)



class DGSRLayers(nn.Module):
    def __init__(self, in_feats, out_feats, user_max_length, item_max_length, feat_drop=0.2, attn_drop=0.2, user_long='orgat', user_short='att',
                 item_long='orgat', item_short='att', user_update='residual', item_update='residual', K=4):
        super(DGSRLayers, self).__init__()
        self.hidden_size = in_feats
        self.user_long = user_long
        self.item_long = item_long
        self.user_short = user_short
        self.item_short = item_short
        self.user_update_m = user_update
        self.item_update_m = item_update
        self.user_max_length = user_max_length
        self.item_max_length = item_max_length
        self.K = torch.tensor(K).cuda()
        if self.user_long in ['orgat', 'gcn', 'gru'] and self.user_short in ['last','att', 'att1']:
            self.agg_gate_u = nn.Linear(self.hidden_size * 2, self.hidden_size, bias=False)
        if self.item_long in ['orgat', 'gcn', 'gru'] and self.item_short in ['last', 'att', 'att1']:
            self.agg_gate_i = nn.Linear(self.hidden_size * 2, self.hidden_size, bias=False)
        if self.user_long in ['gru']:
            self.gru_u = nn.GRU(input_size=in_feats, hidden_size=in_feats, batch_first=True)
        if self.item_long in ['gru']:
            self.gru_i = nn.GRU(input_size=in_feats, hidden_size=in_feats, batch_first=True)
        if self.user_update_m == 'norm':
            self.norm_user = nn.LayerNorm(self.hidden_size)
        if self.item_update_m == 'norm':
            self.norm_item = nn.LayerNorm(self.hidden_size)
        self.feat_drop = nn.Dropout(feat_drop)
        self.atten_drop = nn.Dropout(attn_drop)
        self.user_weight = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.item_weight = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

        if self.user_update_m in ['concat', 'rnn']:
            self.user_update = nn.Linear(2 * self.hidden_size, self.hidden_size, bias=False)
        if self.item_update_m in ['concat', 'rnn']:
            self.item_update = nn.Linear(2 * self.hidden_size, self.hidden_size, bias=False)
        # attention+ attention mechanism
        if self.user_short in ['last', 'att']:
            self.last_weight_u = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        if self.item_short in ['last', 'att']:
            self.last_weight_i = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

        if self.item_long in ['orgat']:
            self.i_time_encoding = nn.Embedding(self.user_max_length, self.hidden_size)
            self.i_time_encoding_k = nn.Embedding(self.user_max_length, self.hidden_size)
        if self.user_long in ['orgat']:
            self.u_time_encoding = nn.Embedding(self.item_max_length, self.hidden_size)
            self.u_time_encoding_k = nn.Embedding(self.item_max_length, self.hidden_size)


    def user_update_function(self, user_now, user_old):
        if self.user_update_m == 'residual':
            return F.elu(user_now + user_old)
        elif self.user_update_m == 'gate_update':
            pass
        elif self.user_update_m == 'concat':
            return F.elu(self.user_update(torch.cat([user_now, user_old], -1)))
        elif self.user_update_m == 'light':
            pass
        elif self.user_update_m == 'norm':
            return self.feat_drop(self.norm_user(user_now)) + user_old
        elif self.user_update_m == 'rnn':
            return F.tanh(self.user_update(torch.cat([user_now, user_old], -1)))
        else:
            print('error: no user_update')
            exit()

    def item_update_function(self, item_now, item_old):
        if self.item_update_m == 'residual':
            return F.elu(item_now + item_old)
        elif self.item_update_m == 'concat':
            return F.elu(self.item_update(torch.cat([item_now, item_old], -1)))
        elif self.item_update_m == 'light':
            pass
        elif self.item_update_m == 'norm':
            return self.feat_drop(self.norm_item(item_now)) + item_old
        elif self.item_update_m == 'rnn':
            return F.tanh(self.item_update(torch.cat([item_now, item_old], -1)))
        else:
            print('error: no item_update')
            exit()

    def forward(self, g, feat_dict=None):
        if feat_dict == None:
            if self.user_long in ['gcn']:
                g.nodes['user'].data['norm'] = g['by'].in_degrees().unsqueeze(1).cuda()
            if self.item_long in ['gcn']:
                g.nodes['item'].data['norm'] = g['by'].out_degrees().unsqueeze(1).cuda()
            user_ = g.nodes['user'].data['user_h']
            item_ = g.nodes['item'].data['item_h']
        else:
            user_ = feat_dict['user'].cuda()
            item_ = feat_dict['item'].cuda()
            if self.user_long in ['gcn']:
                g.nodes['user'].data['norm'] = g['by'].in_degrees().unsqueeze(1).cuda()
            if self.item_long in ['gcn']:
                g.nodes['item'].data['norm'] = g['by'].out_degrees().unsqueeze(1).cuda()
        g.nodes['user'].data['user_h'] = self.user_weight(self.feat_drop(user_))
        g.nodes['item'].data['item_h'] = self.item_weight(self.feat_drop(item_))
        g = self.graph_update(g)
        g.nodes['user'].data['user_h'] = self.user_update_function(g.nodes['user'].data['user_h'], user_)
        g.nodes['item'].data['item_h'] = self.item_update_function(g.nodes['item'].data['item_h'], item_)
        f_dict = {'user': g.nodes['user'].data['user_h'], 'item': g.nodes['item'].data['item_h']}
        return f_dict

    def graph_update(self, g):
        # user_encoder 对user进行编码
        # update all nodes
        g.multi_update_all({'by': (self.user_message_func, self.user_reduce_func),
                            'pby': (self.item_message_func, self.item_reduce_func)}, 'sum')
        return g

    def item_message_func(self, edges):
        dic = {}
        dic['time'] = edges.data['time']
        dic['user_h'] = edges.src['user_h']
        dic['item_h'] = edges.dst['item_h']
        return dic

    def item_reduce_func(self, nodes):
        h = []
        #先根据time排序
        #order = torch.sort(nodes.mailbox['time'], 1)[1]
        order = torch.argsort(torch.argsort(nodes.mailbox['time'], 1), 1)
        re_order = nodes.mailbox['time'].shape[1] -order -1
        length = nodes.mailbox['item_h'].shape[0]
        #长期兴趣编码
        if self.item_long == 'orgat':
            e_ij = torch.sum((self.i_time_encoding(re_order) + nodes.mailbox['user_h']) * nodes.mailbox['item_h'], dim=2)\
                   /torch.sqrt(torch.tensor(self.hidden_size).float())
            alpha = self.atten_drop(F.softmax(e_ij, dim=1))
            if len(alpha.shape) == 2:
                alpha = alpha.unsqueeze(2)
            h_long = torch.sum(alpha * (nodes.mailbox['user_h'] + self.i_time_encoding_k(re_order)), dim=1)
            h.append(h_long)
        elif self.item_long == 'gru':
            rnn_order = torch.sort(nodes.mailbox['time'], 1)[1]
            _, hidden_u = self.gru_i(nodes.mailbox['user_h'][torch.arange(length).unsqueeze(1), rnn_order])
            h.append(hidden_u.squeeze(0))
        ## 短期兴趣编码
        last = torch.argmax(nodes.mailbox['time'], 1)
        last_em = nodes.mailbox['user_h'][torch.arange(length), last, :].unsqueeze(1)
        if self.item_short == 'att':
            e_ij1 = torch.sum(last_em * nodes.mailbox['user_h'], dim=2) / torch.sqrt(
                torch.tensor(self.hidden_size).float())
            alpha1 = self.atten_drop(F.softmax(e_ij1, dim=1))
            if len(alpha1.shape) == 2:
                alpha1 = alpha1.unsqueeze(2)
            h_short = torch.sum(alpha1 * nodes.mailbox['user_h'], dim=1)
            h.append(h_short)
        elif self.item_short == 'last':
            h.append(last_em.squeeze())
        if len(h) == 1:
            return {'item_h': h[0]}
        else:
            return {'item_h': self.agg_gate_i(torch.cat(h,-1))}

    def user_message_func(self, edges):
        dic = {}
        dic['time'] = edges.data['time']
        dic['item_h'] = edges.src['item_h']
        dic['user_h'] = edges.dst['user_h']
        return dic

    def user_reduce_func(self, nodes):
        h = []
        # 先根据time排序
        order = torch.argsort(torch.argsort(nodes.mailbox['time'], 1),1)
        re_order = nodes.mailbox['time'].shape[1] - order -1
        length = nodes.mailbox['user_h'].shape[0]
        # 长期兴趣编码
        if self.user_long == 'orgat':
            e_ij = torch.sum((self.u_time_encoding(re_order) + nodes.mailbox['item_h']) *nodes.mailbox['user_h'],
                             dim=2) / torch.sqrt(torch.tensor(self.hidden_size).float())
            alpha = self.atten_drop(F.softmax(e_ij, dim=1))
            if len(alpha.shape) == 2:
                alpha = alpha.unsqueeze(2)
            h_long = torch.sum(alpha * (nodes.mailbox['item_h'] + self.u_time_encoding_k(re_order)), dim=1)
            h.append(h_long)
        elif self.user_long == 'gru':
            rnn_order = torch.sort(nodes.mailbox['time'], 1)[1]
            _, hidden_i = self.gru_u(nodes.mailbox['item_h'][torch.arange(length).unsqueeze(1), rnn_order])
            h.append(hidden_i.squeeze(0))
        ## 短期兴趣编码
        last = torch.argmax(nodes.mailbox['time'], 1)
        last_em = nodes.mailbox['item_h'][torch.arange(length), last, :].unsqueeze(1)
        if self.user_short == 'att':
            e_ij1 = torch.sum(last_em * nodes.mailbox['item_h'], dim=2)/torch.sqrt(torch.tensor(self.hidden_size).float())
            alpha1 = self.atten_drop(F.softmax(e_ij1, dim=1))
            if len(alpha1.shape) == 2:
                alpha1 = alpha1.unsqueeze(2)
            h_short = torch.sum(alpha1 * nodes.mailbox['item_h'], dim=1)
            h.append(h_short)
        elif self.user_short == 'last':
            h.append(last_em.squeeze())

        if len(h) == 1:
            return {'user_h': h[0]}
        else:
            return {'user_h': self.agg_gate_u(torch.cat(h,-1))}

def graph_user(bg, user_index, user_embedding):
    b_user_size = bg.batch_num_nodes('user')
    # tmp = np.roll(np.cumsum(b_user_size).cpu(), 1)
    # ----numpy写法----
    # tmp = np.roll(np.cumsum(b_user_size.cpu().numpy()), 1)
    # tmp[0] = 0
    # new_user_index = torch.Tensor(tmp).long().cuda() + user_index
    # ----pytorch写法
    tmp = torch.roll(torch.cumsum(b_user_size, 0), 1)
    tmp[0] = 0
    new_user_index = tmp + user_index
    return user_embedding[new_user_index]

def graph_item(bg, last_index, item_embedding):
    b_item_size = bg.batch_num_nodes('item')
    # ----numpy写法----
    # tmp = np.roll(np.cumsum(b_item_size.cpu().numpy()), 1)
    # tmp[0] = 0
    # new_item_index = torch.Tensor(tmp).long().cuda() + last_index
    # ----pytorch写法
    tmp = torch.roll(torch.cumsum(b_item_size, 0), 1)
    tmp[0] = 0
    new_item_index = tmp + last_index
    return item_embedding[new_item_index]

def order_update(edges):
    dic = {}
    dic['order'] = torch.sort(edges.data['time'])[1]
    dic['re_order'] = len(edges.data['time']) - dic['order']
    return dic


def collate(data):
    user = []
    user_l = []
    graph = []
    label = []
    last_item = []
    for da in data:
        user.append(da[1]['user'])
        user_l.append(da[1]['u_alis'])
        graph.append(da[0][0])
        label.append(da[1]['target'])
        last_item.append(da[1]['last_alis'])
    return torch.tensor(user_l).long(), dgl.batch(graph), torch.tensor(label).long(), torch.tensor(last_item).long()


def neg_generate(user, data_neg, neg_num=100):
    neg = np.zeros((len(user), neg_num), np.int32)
    for i, u in enumerate(user):
        neg[i] = np.random.choice(data_neg[u], neg_num, replace=False)
    return neg


def collate_test(data, user_neg):
    # 生成负样本和每个序列的长度
    user = []
    graph = []
    label = []
    last_item = []
    for da in data:
        user.append(da[1]['u_alis'])
        graph.append(da[0][0])
        label.append(da[1]['target'])
        last_item.append(da[1]['last_alis'])
    return torch.tensor(user).long(), dgl.batch(graph), torch.tensor(label).long(), torch.tensor(last_item).long(), torch.Tensor(neg_generate(user, user_neg)).long()




