# Copyright 2019-present NAVER Corp.
# Apache License v2.0

# Wonseok Hwang

import os, json
from copy import deepcopy
from matplotlib.pylab import *

import torch
import torch.nn as nn
import torch.nn.functional as F


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from sqlova.utils.utils import topk_multi_dim
from sqlova.utils.utils_wikisql import *

class Seq2SQL_v1(nn.Module):
    def __init__(self, iS, hS, lS, dr, n_cond_ops, n_agg_ops, old=False):
        super(Seq2SQL_v1, self).__init__()
        self.iS = iS
        self.hS = hS
        self.ls = lS
        self.dr = dr

        self.max_wn = 4
        self.n_cond_ops = n_cond_ops
        self.n_agg_ops = n_agg_ops

        self.wcp = WCP(iS, hS, lS, dr)
        self.scp = SCP(iS, hS, lS, dr)
        self.sap = SAP(iS, hS, lS, dr, n_agg_ops, old=old)
        self.wnp = WNP(iS, hS, lS, dr)
        self.wcp = WCP(iS, hS, lS, dr)
        self.wop = WOP(iS, hS, lS, dr, n_cond_ops)
        self.wvp = WVP_se(iS, hS, lS, dr, n_cond_ops, old=old) # start-end-search-discriminative model


    def forward(self, wemb_n, l_n, wemb_hpu, l_hpu, l_hs,
                g_sc=None, g_sa=None, g_wn=None, g_wc=None, g_wo=None, g_wvi=None,
                show_p_sc=False, show_p_sa=False,
                show_p_wn=False, show_p_wc=False, show_p_wo=False, show_p_wv=False):

        # sc
        s_sc = self.scp(wemb_n, l_n, wemb_hpu, l_hpu, l_hs, show_p_sc=show_p_sc)

        if g_sc:
            pr_sc = g_sc
        else:
            pr_sc = pred_sc(s_sc)

        # sa
        s_sa = self.sap(wemb_n, l_n, wemb_hpu, l_hpu, l_hs, pr_sc, show_p_sa=show_p_sa)
        if g_sa:
            # it's not necessary though.
            pr_sa = g_sa
        else:
            pr_sa = pred_sa(s_sa)


        # wn
        s_wn = self.wnp(wemb_n, l_n, wemb_hpu, l_hpu, l_hs, show_p_wn=show_p_wn)

        if g_wn:
            pr_wn = g_wn
        else:
            pr_wn = pred_wn(s_wn)

        # wc
        s_wc = self.wcp(wemb_n, l_n, wemb_hpu, l_hpu, l_hs, show_p_wc=show_p_wc, penalty=True)

        if g_wc:
            pr_wc = g_wc
        else:
            pr_wc = pred_wc(pr_wn, s_wc)

        # wo
        s_wo = self.wop(wemb_n, l_n, wemb_hpu, l_hpu, l_hs, wn=pr_wn, wc=pr_wc, show_p_wo=show_p_wo)

        if g_wo:
            pr_wo = g_wo
        else:
            pr_wo = pred_wo(pr_wn, s_wo)

        # wv
        s_wv = self.wvp(wemb_n, l_n, wemb_hpu, l_hpu, l_hs, wn=pr_wn, wc=pr_wc, wo=pr_wo, show_p_wv=show_p_wv)

        return s_sc, s_sa, s_wn, s_wc, s_wo, s_wv

    def beam_forward(self, wemb_n, l_n, wemb_hpu, l_hpu, l_hs, engine, tb,
                     nlu_t, nlu_wp_t, wp_to_wh_index, nlu,
                     beam_size=4,
                     show_p_sc=False, show_p_sa=False,
                     show_p_wn=False, show_p_wc=False, show_p_wo=False, show_p_wv=False):
        """
        Execution-guided beam decoding.
        """
        # sc
        s_sc = self.scp(wemb_n, l_n, wemb_hpu, l_hpu, l_hs, show_p_sc=show_p_sc)
        prob_sc = F.softmax(s_sc, dim=-1)
        bS, mcL = s_sc.shape

        # minimum_hs_length = min(l_hs)
        # beam_size = minimum_hs_length if beam_size > minimum_hs_length else beam_size

        # sa
        # Construct all possible sc_sa_score
        prob_sc_sa = torch.zeros([bS, beam_size, self.n_agg_ops]).to(device)
        prob_sca = torch.zeros_like(prob_sc_sa).to(device)

        # get the top-k indices.  pr_sc_beam = [B, beam_size]
        pr_sc_beam = pred_sc_beam(s_sc, beam_size)

        # calculate and predict s_sa.
        for i_beam in range(beam_size):
            pr_sc = list( array(pr_sc_beam)[:,i_beam] )
            s_sa = self.sap(wemb_n, l_n, wemb_hpu, l_hpu, l_hs, pr_sc, show_p_sa=show_p_sa)
            prob_sa = F.softmax(s_sa, dim=-1)
            prob_sc_sa[:, i_beam, :] = prob_sa

            prob_sc_selected = prob_sc[range(bS), pr_sc] # [B]
            prob_sca[:,i_beam,:] =  (prob_sa.t() * prob_sc_selected).t()
            # [mcL, B] * [B] -> [mcL, B] (element-wise multiplication)
            # [mcL, B] -> [B, mcL]

        # Calculate the dimension of tensor
        # tot_dim = len(prob_sca.shape)

        # First flatten to 1-d
        idxs = topk_multi_dim(torch.tensor(prob_sca), n_topk=beam_size, batch_exist=True)
        # Now as sc_idx is already sorted, re-map them properly.

        idxs = remap_sc_idx(idxs, pr_sc_beam) # [sc_beam_idx, sa_idx] -> [sc_idx, sa_idx]
        idxs_arr = array(idxs)
        # [B, beam_size, remainig dim]
        # idxs[b][0] gives first probable [sc_idx, sa_idx] pairs.
        # idxs[b][1] gives of second.

        # Calculate prob_sca, a joint probability
        beam_idx_sca = [0] * bS
        beam_meet_the_final = [False] * bS
        while True:
            pr_sc = idxs_arr[range(bS),beam_idx_sca,0]
            pr_sa = idxs_arr[range(bS),beam_idx_sca,1]

            # map index properly

            check = check_sc_sa_pairs(tb, pr_sc, pr_sa)

            if sum(check) == bS:
                break
            else:
                for b, check1 in enumerate(check):
                    if not check1: # wrong pair
                        beam_idx_sca[b] += 1
                        if beam_idx_sca[b] >= beam_size:
                            beam_meet_the_final[b] = True
                            beam_idx_sca[b] -= 1
                    else:
                        beam_meet_the_final[b] = True

            if sum(beam_meet_the_final) == bS:
                break


        # Now pr_sc, pr_sa are properly predicted.
        pr_sc_best = list(pr_sc)
        pr_sa_best = list(pr_sa)

        # Now, Where-clause beam search.
        s_wn = self.wnp(wemb_n, l_n, wemb_hpu, l_hpu, l_hs, show_p_wn=show_p_wn)
        prob_wn = F.softmax(s_wn, dim=-1).detach().to('cpu').numpy()

        # Found "executable" most likely 4(=max_num_of_conditions) where-clauses.
        # wc
        s_wc = self.wcp(wemb_n, l_n, wemb_hpu, l_hpu, l_hs, show_p_wc=show_p_wc, penalty=True)
        prob_wc = F.sigmoid(s_wc).detach().to('cpu').numpy()
        # pr_wc_sorted_by_prob = pred_wc_sorted_by_prob(s_wc)

        # get max_wn # of most probable columns & their prob.
        pr_wn_max = [self.max_wn]*bS
        pr_wc_max = pred_wc(pr_wn_max, s_wc) # if some column do not have executable where-claouse, omit that column
        prob_wc_max = zeros([bS, self.max_wn])
        for b, pr_wc_max1 in enumerate(pr_wc_max):
            prob_wc_max[b,:] = prob_wc[b,pr_wc_max1]

        # get most probable max_wn where-clouses
        # wo
        s_wo_max = self.wop(wemb_n, l_n, wemb_hpu, l_hpu, l_hs, wn=pr_wn_max, wc=pr_wc_max, show_p_wo=show_p_wo)
        prob_wo_max = F.softmax(s_wo_max, dim=-1).detach().to('cpu').numpy()
        # [B, max_wn, n_cond_op]

        pr_wvi_beam_op_list = []
        prob_wvi_beam_op_list = []
        for i_op  in range(self.n_cond_ops-1):
            pr_wo_temp = [ [i_op]*self.max_wn ]*bS
            # wv
            s_wv = self.wvp(wemb_n, l_n, wemb_hpu, l_hpu, l_hs, wn=pr_wn_max, wc=pr_wc_max, wo=pr_wo_temp, show_p_wv=show_p_wv)
            prob_wv = F.softmax(s_wv, dim=-2).detach().to('cpu').numpy()

            # prob_wv
            pr_wvi_beam, prob_wvi_beam = pred_wvi_se_beam(self.max_wn, s_wv, beam_size)
            pr_wvi_beam_op_list.append(pr_wvi_beam)
            prob_wvi_beam_op_list.append(prob_wvi_beam)
            # pr_wvi_beam = [B, max_wn, k_logit**2 [st, ed] paris]

            # pred_wv_beam

        # Calculate joint probability of where-clause
        # prob_w = [batch, wc, wo, wv] = [B, max_wn, n_cond_op, n_pairs]
        n_wv_beam_pairs = prob_wvi_beam.shape[2]
        prob_w = zeros([bS, self.max_wn, self.n_cond_ops-1, n_wv_beam_pairs])
        for b in range(bS):
            for i_wn in range(self.max_wn):
                for i_op in range(self.n_cond_ops-1): # do not use final one
                    for i_wv_beam in range(n_wv_beam_pairs):
                        # i_wc = pr_wc_max[b][i_wn] # already done
                        p_wc = prob_wc_max[b, i_wn]
                        p_wo = prob_wo_max[b, i_wn, i_op]
                        p_wv = prob_wvi_beam_op_list[i_op][b, i_wn, i_wv_beam]

                        prob_w[b, i_wn, i_op, i_wv_beam] = p_wc * p_wo * p_wv

        # Perform execution guided decoding
        conds_max = []
        prob_conds_max = []
        # while len(conds_max) < self.max_wn:
        idxs = topk_multi_dim(torch.tensor(prob_w), n_topk=beam_size, batch_exist=True)
        # idxs = [B, i_wc_beam, i_op, i_wv_pairs]

        # Construct conds1
        for b, idxs1 in enumerate(idxs):
            conds_max1 = []
            prob_conds_max1 = []
            for i_wn, idxs11 in enumerate(idxs1):
                i_wc = pr_wc_max[b][idxs11[0]]
                i_op = idxs11[1]
                wvi = pr_wvi_beam_op_list[i_op][b][idxs11[0]][idxs11[2]]

                # get wv_str
                temp_pr_wv_str, _ = convert_pr_wvi_to_string([[wvi]], [nlu_t[b]], [nlu_wp_t[b]], [wp_to_wh_index[b]], [nlu[b]])
                merged_wv11 = merge_wv_t1_eng(temp_pr_wv_str[0][0], nlu[b])
                conds11 = [i_wc, i_op, merged_wv11]

                prob_conds11 = prob_w[b, idxs11[0], idxs11[1], idxs11[2] ]

                # test execution
                # print(nlu[b])
                # print(tb[b]['id'], tb[b]['types'], pr_sc[b], pr_sa[b], [conds11])
                pr_ans = engine.execute(tb[b]['id'], pr_sc[b], pr_sa[b], [conds11])
                if bool(pr_ans):
                    # pr_ans is not empty!
                    conds_max1.append(conds11)
                    prob_conds_max1.append(prob_conds11)
            conds_max.append(conds_max1)
            prob_conds_max.append(prob_conds_max1)

            # May need to do more exhuastive search?
            # i.e. up to.. getting all executable cases.

        # Calculate total probability to decide the number of where-clauses
        pr_sql_i = []
        prob_wn_w = []
        pr_wn_based_on_prob = []

        for b, prob_wn1 in enumerate(prob_wn):
            max_executable_wn1 = len( conds_max[b] )
            prob_wn_w1 = []
            prob_wn_w1.append(prob_wn1[0])  # wn=0 case.
            for i_wn in range(max_executable_wn1):
                prob_wn_w11 = prob_wn1[i_wn+1] * prob_conds_max[b][i_wn]
                prob_wn_w1.append(prob_wn_w11)
            pr_wn_based_on_prob.append(argmax(prob_wn_w1))
            prob_wn_w.append(prob_wn_w1)

            pr_sql_i1 = {'agg': pr_sa_best[b], 'sel': pr_sc_best[b], 'conds': conds_max[b][:pr_wn_based_on_prob[b]]}
            pr_sql_i.append(pr_sql_i1)
        # s_wv = [B, max_wn, max_nlu_tokens, 2]
        return prob_sca, prob_w, prob_wn_w, pr_sc_best, pr_sa_best, pr_wn_based_on_prob, pr_sql_i

class SCP(nn.Module):
    def __init__(self, iS=300, hS=100, lS=2, dr=0.3):
        super(SCP, self).__init__()
        self.iS = iS
        self.hS = hS
        self.lS = lS
        self.dr = dr


        self.enc_h = nn.LSTM(input_size=iS, hidden_size=int(hS / 2),
                             num_layers=lS, batch_first=True,
                             dropout=dr, bidirectional=True)

        self.enc_n = nn.LSTM(input_size=iS, hidden_size=int(hS / 2),
                             num_layers=lS, batch_first=True,
                             dropout=dr, bidirectional=True)

        self.W_att = nn.Linear(hS, hS)
        self.W_c = nn.Linear(hS, hS)
        self.W_hs = nn.Linear(hS, hS)
        self.sc_out = nn.Sequential(nn.Tanh(), nn.Linear(2 * hS, 1))

        self.softmax_dim1 = nn.Softmax(dim=1)
        self.softmax_dim2 = nn.Softmax(dim=2)

    def forward(self, wemb_n, l_n, wemb_hpu, l_hpu, l_hs, show_p_sc=False):
        # Encode
        wenc_n = encode(self.enc_n, wemb_n, l_n,
                        return_hidden=False,
                        hc0=None,
                        last_only=False)  # [b, n, dim]

        wenc_hs = encode_hpu(self.enc_h, wemb_hpu, l_hpu, l_hs)  # [b, hs, dim]

        bS = len(l_hs)
        mL_n = max(l_n)

        #   [bS, mL_hs, 100] * [bS, 100, mL_n] -> [bS, mL_hs, mL_n]
        att_h = torch.bmm(wenc_hs, self.W_att(wenc_n).transpose(1, 2))

        #   Penalty on blank parts
        for b, l_n1 in enumerate(l_n):
            if l_n1 < mL_n:
                att_h[b, :, l_n1:] = -10000000000

        p_n = self.softmax_dim2(att_h)
        if show_p_sc:
            # p = [b, hs, n]
            if p_n.shape[0] != 1:
                raise Exception("Batch size should be 1.")
            fig=figure(2001, figsize=(12,3.5))
            # subplot(6,2,7)
            subplot2grid((7,2), (3, 0), rowspan=2)
            cla()
            _color='rgbkcm'
            _symbol='.......'
            for i_h in range(l_hs[0]):
                color_idx = i_h % len(_color)
                plot(p_n[0][i_h][:].data.numpy() - i_h, '--'+_symbol[color_idx]+_color[color_idx], ms=7)

            title('sc: p_n for each h')
            grid(True)
            fig.tight_layout()
            fig.canvas.draw()
            show()



        #   p_n [ bS, mL_hs, mL_n]  -> [ bS, mL_hs, mL_n, 1]
        #   wenc_n [ bS, mL_n, 100] -> [ bS, 1, mL_n, 100]
        #   -> [bS, mL_hs, mL_n, 100] -> [bS, mL_hs, 100]
        c_n = torch.mul(p_n.unsqueeze(3), wenc_n.unsqueeze(1)).sum(dim=2)

        vec = torch.cat([self.W_c(c_n), self.W_hs(wenc_hs)], dim=2)
        s_sc = self.sc_out(vec).squeeze(2) # [bS, mL_hs, 1] -> [bS, mL_hs]

        # Penalty
        mL_hs = max(l_hs)
        for b, l_hs1 in enumerate(l_hs):
            if l_hs1 < mL_hs:
                s_sc[b, l_hs1:] = -10000000000

        return s_sc


class SAP(nn.Module):
    def __init__(self, iS=300, hS=100, lS=2, dr=0.3, n_agg_ops=-1, old=False):
        super(SAP, self).__init__()
        self.iS = iS
        self.hS = hS
        self.lS = lS
        self.dr = dr


        self.enc_h = nn.LSTM(input_size=iS, hidden_size=int(hS / 2),
                             num_layers=lS, batch_first=True,
                             dropout=dr, bidirectional=True)

        self.enc_n = nn.LSTM(input_size=iS, hidden_size=int(hS / 2),
                             num_layers=lS, batch_first=True,
                             dropout=dr, bidirectional=True)

        self.W_att = nn.Linear(hS, hS)
        self.sa_out = nn.Sequential(nn.Linear(hS, hS),
                                    nn.Tanh(),
                                    nn.Linear(hS, n_agg_ops))  # Fixed number of aggregation operator.

        self.softmax_dim1 = nn.Softmax(dim=1)
        self.softmax_dim2 = nn.Softmax(dim=2)

        if old:
            # for backwoard compatibility
            self.W_c = nn.Linear(hS, hS)
            self.W_hs = nn.Linear(hS, hS)

    def forward(self, wemb_n, l_n, wemb_hpu, l_hpu, l_hs, pr_sc, show_p_sa=False):
        # Encode
        wenc_n = encode(self.enc_n, wemb_n, l_n,
                        return_hidden=False,
                        hc0=None,
                        last_only=False)  # [b, n, dim]

        wenc_hs = encode_hpu(self.enc_h, wemb_hpu, l_hpu, l_hs)  # [b, hs, dim]

        bS = len(l_hs)
        mL_n = max(l_n)

        wenc_hs_ob = wenc_hs[list(range(bS)), pr_sc]  # list, so one sample for each batch.

        # [bS, mL_n, 100] * [bS, 100, 1] -> [bS, mL_n]
        att = torch.bmm(self.W_att(wenc_n), wenc_hs_ob.unsqueeze(2)).squeeze(2)

        #   Penalty on blank parts
        for b, l_n1 in enumerate(l_n):
            if l_n1 < mL_n:
                att[b, l_n1:] = -10000000000
        # [bS, mL_n]
        p = self.softmax_dim1(att)

        if show_p_sa:
            if p.shape[0] != 1:
                raise Exception("Batch size should be 1.")
            fig=figure(2001);
            subplot(7,2,3)
            cla()
            plot(p[0].data.numpy(), '--rs', ms=7)
            title('sa: nlu_weight')
            grid(True)
            fig.tight_layout()
            fig.canvas.draw()
            show()
            
        #    [bS, mL_n, 100] * ( [bS, mL_n, 1] -> [bS, mL_n, 100])
        #       -> [bS, mL_n, 100] -> [bS, 100]
        c_n = torch.mul(wenc_n, p.unsqueeze(2).expand_as(wenc_n)).sum(dim=1)
        s_sa = self.sa_out(c_n)

        return s_sa


class WNP(nn.Module):
    def __init__(self, iS=300, hS=100, lS=2, dr=0.3, ):
        super(WNP, self).__init__()
        self.iS = iS
        self.hS = hS
        self.lS = lS
        self.dr = dr

        self.mL_w = 4  # max where condition number

        self.enc_h = nn.LSTM(input_size=iS, hidden_size=int(hS / 2),
                             num_layers=lS, batch_first=True,
                             dropout=dr, bidirectional=True)

        self.enc_n = nn.LSTM(input_size=iS, hidden_size=int(hS / 2),
                             num_layers=lS, batch_first=True,
                             dropout=dr, bidirectional=True)

        self.W_att_h = nn.Linear(hS, 1)
        self.W_hidden = nn.Linear(hS, lS * hS)
        self.W_cell = nn.Linear(hS, lS * hS)

        self.W_att_n = nn.Linear(hS, 1)
        self.wn_out = nn.Sequential(nn.Linear(hS, hS),
                                    nn.Tanh(),
                                    nn.Linear(hS, self.mL_w + 1))  # max number (4 + 1)

        self.softmax_dim1 = nn.Softmax(dim=1)
        self.softmax_dim2 = nn.Softmax(dim=2)

    def forward(self, wemb_n, l_n, wemb_hpu, l_hpu, l_hs, show_p_wn=False):
        # Encode

        wenc_hs = encode_hpu(self.enc_h, wemb_hpu, l_hpu, l_hs)  # [b, mL_hs, dim]

        bS = len(l_hs)
        mL_n = max(l_n)
        mL_hs = max(l_hs)
        # mL_h = max(l_hpu)

        #   (self-attention?) column Embedding?
        #   [B, mL_hs, 100] -> [B, mL_hs, 1] -> [B, mL_hs]
        att_h = self.W_att_h(wenc_hs).squeeze(2)

        #   Penalty
        for b, l_hs1 in enumerate(l_hs):
            if l_hs1 < mL_hs:
                att_h[b, l_hs1:] = -10000000000
        p_h = self.softmax_dim1(att_h)

        if show_p_wn:
            if p_h.shape[0] != 1:
                raise Exception("Batch size should be 1.")
            fig=figure(2001);
            subplot(7,2,5)
            cla()
            plot(p_h[0].data.numpy(), '--rs', ms=7)
            title('wn: header_weight')
            grid(True)
            fig.canvas.draw()
            show()
            # input('Type Eenter to continue.')

        #   [B, mL_hs, 100] * [ B, mL_hs, 1] -> [B, mL_hs, 100] -> [B, 100]
        c_hs = torch.mul(wenc_hs, p_h.unsqueeze(2)).sum(1)

        #   [B, 100] --> [B, 2*100] Enlarge because there are two layers.
        hidden = self.W_hidden(c_hs)  # [B, 4, 200/2]
        hidden = hidden.view(bS, self.lS * 2, int(
            self.hS / 2))  # [4, B, 100/2] # number_of_layer_layer * (bi-direction) # lstm input convention.
        hidden = hidden.transpose(0, 1).contiguous()

        cell = self.W_cell(c_hs)  # [B, 4, 100/2]
        cell = cell.view(bS, self.lS * 2, int(self.hS / 2))  # [4, B, 100/2]
        cell = cell.transpose(0, 1).contiguous()

        wenc_n = encode(self.enc_n, wemb_n, l_n,
                        return_hidden=False,
                        hc0=(hidden, cell),
                        last_only=False)  # [b, n, dim]

        att_n = self.W_att_n(wenc_n).squeeze(2)  # [B, max_len, 100] -> [B, max_len, 1] -> [B, max_len]

        #    Penalty
        for b, l_n1 in enumerate(l_n):
            if l_n1 < mL_n:
                att_n[b, l_n1:] = -10000000000
        p_n = self.softmax_dim1(att_n)

        if show_p_wn:
            if p_n.shape[0] != 1:
                raise Exception("Batch size should be 1.")
            fig=figure(2001);
            subplot(7,2,6)
            cla()
            plot(p_n[0].data.numpy(), '--rs', ms=7)
            title('wn: nlu_weight')
            grid(True)
            fig.canvas.draw()

            show()
            # input('Type Enter to continue.')

        #    [B, mL_n, 100] *([B, mL_n] -> [B, mL_n, 1] -> [B, mL_n, 100] ) -> [B, 100]
        c_n = torch.mul(wenc_n, p_n.unsqueeze(2).expand_as(wenc_n)).sum(dim=1)
        s_wn = self.wn_out(c_n)

        return s_wn

class WCP(nn.Module):
    def __init__(self, iS=300, hS=100, lS=2, dr=0.3):
        super(WCP, self).__init__()
        self.iS = iS
        self.hS = hS
        self.lS = lS
        self.dr = dr

        self.enc_h = nn.LSTM(input_size=iS, hidden_size=int(hS / 2),
                             num_layers=lS, batch_first=True,
                             dropout=dr, bidirectional=True)

        self.enc_n = nn.LSTM(input_size=iS, hidden_size=int(hS / 2),
                             num_layers=lS, batch_first=True,
                             dropout=dr, bidirectional=True)

        self.W_att = nn.Linear(hS, hS)
        self.W_c = nn.Linear(hS, hS)
        self.W_hs = nn.Linear(hS, hS)
        self.W_out = nn.Sequential(
            nn.Tanh(), nn.Linear(2 * hS, 1)
        )

        self.softmax_dim1 = nn.Softmax(dim=1)
        self.softmax_dim2 = nn.Softmax(dim=2)

    def forward(self, wemb_n, l_n, wemb_hpu, l_hpu, l_hs, show_p_wc, penalty=True):
        # Encode
        wenc_n = encode(self.enc_n, wemb_n, l_n,
                        return_hidden=False,
                        hc0=None,
                        last_only=False)  # [b, n, dim]

        wenc_hs = encode_hpu(self.enc_h, wemb_hpu, l_hpu, l_hs)  # [b, hs, dim]

        # attention
        # wenc = [bS, mL, hS]
        # att = [bS, mL_hs, mL_n]
        # att[b, i_h, j_n] = p(j_n| i_h)
        att = torch.bmm(wenc_hs, self.W_att(wenc_n).transpose(1, 2))

        # penalty to blank part.
        mL_n = max(l_n)
        for b_n, l_n1 in enumerate(l_n):
            if l_n1 < mL_n:
                att[b_n, :, l_n1:] = -10000000000

        # make p(j_n | i_h)
        p = self.softmax_dim2(att)

        if show_p_wc:
            # p = [b, hs, n]
            if p.shape[0] != 1:
                raise Exception("Batch size should be 1.")
            fig=figure(2001);
            # subplot(6,2,7)
            subplot2grid((7,2), (3, 1), rowspan=2)
            cla()
            _color='rgbkcm'
            _symbol='.......'
            for i_h in range(l_hs[0]):
                color_idx = i_h % len(_color)
                plot(p[0][i_h][:].data.numpy() - i_h, '--'+_symbol[color_idx]+_color[color_idx], ms=7)

            title('wc: p_n for each h')
            grid(True)
            fig.tight_layout()
            fig.canvas.draw()
            show()
        # max nlu context vectors
        # [bS, mL_hs, mL_n]*[bS, mL_hs, mL_n]
        wenc_n = wenc_n.unsqueeze(1)  # [ b, n, dim] -> [b, 1, n, dim]
        p = p.unsqueeze(3)  # [b, hs, n] -> [b, hs, n, 1]
        c_n = torch.mul(wenc_n, p).sum(2)  # -> [b, hs, dim], c_n for each header.

        y = torch.cat([self.W_c(c_n), self.W_hs(wenc_hs)], dim=2)  # [b, hs, 2*dim]
        score = self.W_out(y).squeeze(2)  # [b, hs]

        if penalty:
            for b, l_hs1 in enumerate(l_hs):
                score[b, l_hs1:] = -1e+10

        return score


class WOP(nn.Module):
    def __init__(self, iS=300, hS=100, lS=2, dr=0.3, n_cond_ops=3):
        super(WOP, self).__init__()
        self.iS = iS
        self.hS = hS
        self.lS = lS
        self.dr = dr

        self.mL_w = 4 # max where condition number

        self.enc_h = nn.LSTM(input_size=iS, hidden_size=int(hS / 2),
                             num_layers=lS, batch_first=True,
                             dropout=dr, bidirectional=True)

        self.enc_n = nn.LSTM(input_size=iS, hidden_size=int(hS / 2),
                             num_layers=lS, batch_first=True,
                             dropout=dr, bidirectional=True)

        self.W_att = nn.Linear(hS, hS)
        self.W_c = nn.Linear(hS, hS)
        self.W_hs = nn.Linear(hS, hS)
        self.wo_out = nn.Sequential(
            nn.Linear(2*hS, hS),
            nn.Tanh(),
            nn.Linear(hS, n_cond_ops)
        )

        self.softmax_dim1 = nn.Softmax(dim=1)
        self.softmax_dim2 = nn.Softmax(dim=2)

    def forward(self, wemb_n, l_n, wemb_hpu, l_hpu, l_hs, wn, wc, wenc_n=None, show_p_wo=False):
        # Encode
        if not wenc_n:
            wenc_n = encode(self.enc_n, wemb_n, l_n,
                            return_hidden=False,
                            hc0=None,
                            last_only=False)  # [b, n, dim]

        wenc_hs = encode_hpu(self.enc_h, wemb_hpu, l_hpu, l_hs)  # [b, hs, dim]

        bS = len(l_hs)
        # wn


        wenc_hs_ob = [] # observed hs
        for b in range(bS):
            # [[...], [...]]
            # Pad list to maximum number of selections
            real = [wenc_hs[b, col] for col in wc[b]]
            pad = (self.mL_w - wn[b]) * [wenc_hs[b, 0]] # this padding could be wrong. Test with zero padding later.
            wenc_hs_ob1 = torch.stack(real + pad) # It is not used in the loss function.
            wenc_hs_ob.append(wenc_hs_ob1)

        # list to [B, 4, dim] tensor.
        wenc_hs_ob = torch.stack(wenc_hs_ob) # list to tensor.
        wenc_hs_ob = wenc_hs_ob.to(device)

        # [B, 1, mL_n, dim] * [B, 4, dim, 1]
        #  -> [B, 4, mL_n, 1] -> [B, 4, mL_n]
        # multiplication bewteen NLq-tokens and  selected column
        att = torch.matmul(self.W_att(wenc_n).unsqueeze(1),
                              wenc_hs_ob.unsqueeze(3)
                              ).squeeze(3)

        # Penalty for blank part.
        mL_n = max(l_n)
        for b, l_n1 in enumerate(l_n):
            if l_n1 < mL_n:
                att[b, :, l_n1:] = -10000000000

        p = self.softmax_dim2(att)  # p( n| selected_col )
        if show_p_wo:
            # p = [b, hs, n]
            if p.shape[0] != 1:
                raise Exception("Batch size should be 1.")
            fig=figure(2001)
            # subplot(6,2,7)
            subplot2grid((7,2), (5, 0), rowspan=2)
            cla()
            _color='rgbkcm'
            _symbol='.......'
            for i_wn in range(self.mL_w):
                color_idx = i_wn % len(_color)
                plot(p[0][i_wn][:].data.numpy() - i_wn, '--'+_symbol[color_idx]+_color[color_idx], ms=7)

            title('wo: p_n for selected h')
            grid(True)
            fig.tight_layout()
            fig.canvas.draw()
            show()

        # [B, 1, mL_n, dim] * [B, 4, mL_n, 1]
        #  --> [B, 4, mL_n, dim]
        #  --> [B, 4, dim]
        c_n = torch.mul(wenc_n.unsqueeze(1), p.unsqueeze(3)).sum(dim=2)

        # [bS, 5-1, dim] -> [bS, 5-1, 3]

        vec = torch.cat([self.W_c(c_n), self.W_hs(wenc_hs_ob)], dim=2)
        s_wo = self.wo_out(vec)

        return s_wo

class WVP_se(nn.Module):
    """
    Discriminative model
    Get start and end.
    Here, classifier for [ [투수], [팀1], [팀2], [연도], ...]
    Input:      Encoded nlu & selected column.
    Algorithm: Encoded nlu & selected column. -> classifier -> mask scores -> ...
    """
    def __init__(self, iS=300, hS=100, lS=2, dr=0.3, n_cond_ops=4, old=False):
        super(WVP_se, self).__init__()
        self.iS = iS
        self.hS = hS
        self.lS = lS
        self.dr = dr
        self.n_cond_ops = n_cond_ops

        self.mL_w = 4  # max where condition number

        self.enc_h = nn.LSTM(input_size=iS, hidden_size=int(hS / 2),
                             num_layers=lS, batch_first=True,
                             dropout=dr, bidirectional=True)

        self.enc_n = nn.LSTM(input_size=iS, hidden_size=int(hS / 2),
                             num_layers=lS, batch_first=True,
                             dropout=dr, bidirectional=True)

        self.W_att = nn.Linear(hS, hS)
        self.W_c = nn.Linear(hS, hS)
        self.W_hs = nn.Linear(hS, hS)
        self.W_op = nn.Linear(n_cond_ops, hS)

        # self.W_n = nn.Linear(hS, hS)
        if old:
            self.wv_out =  nn.Sequential(
            nn.Linear(4 * hS, 2)
            )
        else:
            self.wv_out = nn.Sequential(
                nn.Linear(4 * hS, hS),
                nn.Tanh(),
                nn.Linear(hS, 2)
            )
        # self.wv_out = nn.Sequential(
        #     nn.Linear(3 * hS, hS),
        #     nn.Tanh(),
        #     nn.Linear(hS, self.gdkL)
        # )

        self.softmax_dim1 = nn.Softmax(dim=1)
        self.softmax_dim2 = nn.Softmax(dim=2)

    def forward(self, wemb_n, l_n, wemb_hpu, l_hpu, l_hs, wn, wc, wo, wenc_n=None, show_p_wv=False):

        # Encode
        if not wenc_n:
            wenc_n, hout, cout = encode(self.enc_n, wemb_n, l_n,
                            return_hidden=True,
                            hc0=None,
                            last_only=False)  # [b, n, dim]

        wenc_hs = encode_hpu(self.enc_h, wemb_hpu, l_hpu, l_hs)  # [b, hs, dim]

        bS = len(l_hs)

        wenc_hs_ob = []  # observed hs

        for b in range(bS):
            # [[...], [...]]
            # Pad list to maximum number of selections
            real = [wenc_hs[b, col] for col in wc[b]]
            pad = (self.mL_w - wn[b]) * [wenc_hs[b, 0]]  # this padding could be wrong. Test with zero padding later.
            wenc_hs_ob1 = torch.stack(real + pad)  # It is not used in the loss function.
            wenc_hs_ob.append(wenc_hs_ob1)

        # list to [B, 4, dim] tensor.
        wenc_hs_ob = torch.stack(wenc_hs_ob)  # list to tensor.
        wenc_hs_ob = wenc_hs_ob.to(device)


        # Column attention
        # [B, 1, mL_n, dim] * [B, 4, dim, 1]
        #  -> [B, 4, mL_n, 1] -> [B, 4, mL_n]
        # multiplication bewteen NLq-tokens and  selected column
        att = torch.matmul(self.W_att(wenc_n).unsqueeze(1),
                           wenc_hs_ob.unsqueeze(3)
                           ).squeeze(3)
        # Penalty for blank part.
        mL_n = max(l_n)
        for b, l_n1 in enumerate(l_n):
            if l_n1 < mL_n:
                att[b, :, l_n1:] = -10000000000

        p = self.softmax_dim2(att)  # p( n| selected_col )

        if show_p_wv:
            # p = [b, hs, n]
            if p.shape[0] != 1:
                raise Exception("Batch size should be 1.")
            fig=figure(2001)
            # subplot(6,2,7)
            subplot2grid((7,2), (5, 1), rowspan=2)
            cla()
            _color='rgbkcm'
            _symbol='.......'
            for i_wn in range(self.mL_w):
                color_idx = i_wn % len(_color)
                plot(p[0][i_wn][:].data.numpy() - i_wn, '--'+_symbol[color_idx]+_color[color_idx], ms=7)

            title('wv: p_n for selected h')
            grid(True)
            fig.tight_layout()
            fig.canvas.draw()
            show()


        # [B, 1, mL_n, dim] * [B, 4, mL_n, 1]
        #  --> [B, 4, mL_n, dim]
        #  --> [B, 4, dim]
        c_n = torch.mul(wenc_n.unsqueeze(1), p.unsqueeze(3)).sum(dim=2)

        # Select observed headers only.
        # Also generate one_hot vector encoding info of the operator
        # [B, 4, dim]
        wenc_op = []
        for b in range(bS):
            # [[...], [...]]
            # Pad list to maximum number of selections
            wenc_op1 = torch.zeros(self.mL_w, self.n_cond_ops)
            wo1 = wo[b]
            idx_scatter = []
            l_wo1 = len(wo1)
            for i_wo11 in range(self.mL_w):
                if i_wo11 < l_wo1:
                    wo11 = wo1[i_wo11]
                    idx_scatter.append([int(wo11)])
                else:
                    idx_scatter.append([0]) # not used anyway

            wenc_op1 = wenc_op1.scatter(1, torch.tensor(idx_scatter), 1)

            wenc_op.append(wenc_op1)

        # list to [B, 4, dim] tensor.
        wenc_op = torch.stack(wenc_op)  # list to tensor.
        wenc_op = wenc_op.to(device)

        # Now after concat, calculate logits for each token
        # [bS, 5-1, 3*hS] = [bS, 4, 300]
        vec = torch.cat([self.W_c(c_n), self.W_hs(wenc_hs_ob), self.W_op(wenc_op)], dim=2)

        # Make extended vector based on encoded nl token containing column and operator information.
        # wenc_n = [bS, mL, 100]
        # vec2 = [bS, 4, mL, 400]
        vec1e = vec.unsqueeze(2).expand(-1,-1, mL_n, -1) # [bS, 4, 1, 300]  -> [bS, 4, mL, 300]
        wenc_ne = wenc_n.unsqueeze(1).expand(-1, 4, -1, -1) # [bS, 1, mL, 100] -> [bS, 4, mL, 100]
        vec2 = torch.cat( [vec1e, wenc_ne], dim=3)

        # now make logits
        s_wv = self.wv_out(vec2) # [bS, 4, mL, 400] -> [bS, 4, mL, 2]

        # penalty for spurious tokens
        for b, l_n1 in enumerate(l_n):
            if l_n1 < mL_n:
                s_wv[b, :, l_n1:, :] = -10000000000
        return s_wv

def Loss_sw_se(s_sc, s_sa, s_wn, s_wc, s_wo, s_wv, g_sc, g_sa, g_wn, g_wc, g_wo, g_wvi):
    """

    :param s_wv: score  [ B, n_conds, T, score]
    :param g_wn: [ B ]
    :param g_wvi: [B, conds, pnt], e.g. [[[0, 6, 7, 8, 15], [0, 1, 2, 3, 4, 15]], [[0, 1, 2, 3, 16], [0, 7, 8, 9, 16]]]
    :return:
    """
    loss = 0
    loss += Loss_sc(s_sc, g_sc)
    loss += Loss_sa(s_sa, g_sa)
    loss += Loss_wn(s_wn, g_wn)
    loss += Loss_wc(s_wc, g_wc)
    loss += Loss_wo(s_wo, g_wn, g_wo)
    loss += Loss_wv_se(s_wv, g_wn, g_wvi)

    return loss

def Loss_sc(s_sc, g_sc):
    loss = F.cross_entropy(s_sc, torch.tensor(g_sc).to(device))
    return loss


def Loss_sa(s_sa, g_sa):
    loss = F.cross_entropy(s_sa, torch.tensor(g_sa).to(device))
    return loss

def Loss_wn(s_wn, g_wn):
    loss = F.cross_entropy(s_wn, torch.tensor(g_wn).to(device))

    return loss

def Loss_wc(s_wc, g_wc):

    # Construct index matrix
    bS, max_h_len = s_wc.shape
    im = torch.zeros([bS, max_h_len]).to(device)
    for b, g_wc1 in enumerate(g_wc):
        for g_wc11 in g_wc1:
            im[b, g_wc11] = 1.0
    # Construct prob.
    p = F.sigmoid(s_wc)
    loss = F.binary_cross_entropy(p, im)

    return loss


def Loss_wo(s_wo, g_wn, g_wo):

    # Construct index matrix
    loss = 0
    for b, g_wn1 in enumerate(g_wn):
        if g_wn1 == 0:
            continue
        g_wo1 = g_wo[b]
        s_wo1 = s_wo[b]
        loss += F.cross_entropy(s_wo1[:g_wn1], torch.tensor(g_wo1).to(device))

    return loss

def Loss_wv_se(s_wv, g_wn, g_wvi):
    """
    s_wv:   [bS, 4, mL, 2], 4 stands for maximum # of condition, 2 tands for start & end logits.
    g_wvi:  [ [1, 3, 2], [4,3] ] (when B=2, wn(b=1) = 3, wn(b=2) = 2).
    """
    loss = 0
    # g_wvi = torch.tensor(g_wvi).to(device)
    for b, g_wvi1 in enumerate(g_wvi):
        # for i_wn, g_wvi11 in enumerate(g_wvi1):

        g_wn1 = g_wn[b]
        if g_wn1 == 0:
            continue
        g_wvi1 = torch.tensor(g_wvi1).to(device)
        g_st1 = g_wvi1[:,0]
        g_ed1 = g_wvi1[:,1]
        # loss from the start position
        loss += F.cross_entropy(s_wv[b,:g_wn1,:,0], g_st1)

        # print("st_login: ", s_wv[b,:g_wn1,:,0], g_st1, loss)
        # loss from the end position
        loss += F.cross_entropy(s_wv[b,:g_wn1,:,1], g_ed1)
        # print("ed_login: ", s_wv[b,:g_wn1,:,1], g_ed1, loss)

    return loss