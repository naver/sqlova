# Wonseok Hwang
# Sep30, 2018
import os, sys, argparse, re, json
import random as python_random

from matplotlib.pylab import *
import torch.nn as nn
import torch
import torch.nn.functional as F
# import torchvision.datasets as dsets

# BERT
import bert.tokenization as tokenization
from bert.modeling import BertConfig, BertModel

from sqlova.utils.utils_wikisql import *
from sqlova.model.nl2sql.wikisql_models import *
from sqlnet.dbengine import DBEngine

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def construct_hyper_param(parser):
    parser.add_argument('--tepoch', default=200, type=int)
    parser.add_argument("--bS", default=16, type=int,
                        help="Batch size")
    parser.add_argument("--accumulate_gradients", default=2, type=int,
                        help="The number of accumulation of backpropagation to effectivly increase the batch size.")
    parser.add_argument('--fine_tune',
                        default=True,
                        action='store_true',
                        help="If present, BERT is trained.")

    parser.add_argument("--model_type", default='FT_Scalar_1', type=str,
                        help="Type of model.")

    parser.add_argument('--aug',
                        default=False,
                        action='store_true',
                        help="If present, aug.train.jsonl is used.")

    # 1.2 BERT Parameters
    parser.add_argument("--vocab_file",
                        default='vocab.txt', type=str,
                        help="The vocabulary file that the BERT model was trained on.")
    parser.add_argument("--max_seq_length",
                        default=222, type=int, # Set based on maximum length of input tokens.
                        help="The maximum total input sequence length after WordPiece tokenization. Sequences "
                             "longer than this will be truncated, and sequences shorter than this will be padded.")
    parser.add_argument("--num_target_layers",
                        default=1, type=int,
                        help="The Number of final layers of BERT to be used in downstream task.")
    parser.add_argument('--lr_bert', default=6e-6, type=float, help='BERT model learning rate.')
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--no_pretraining', action='store_true', help='Use BERT pretrained model')
    parser.add_argument("--bert_type_abb", default='uS', type=str,
                        help="Type of BERT model to load. e.g.) uS, uL, cS, cL, and mcS")
    parser.add_argument("--col_pool_type", default='start_tok', type=str,
                        help="Which col-token shall be used? start_tok, end_tok, or avg are possible choices.")

    # 1.3 Seq-to-SQL module parameters
    parser.add_argument('--lS', default=2, type=int, help="The number of LSTM layers.")
    parser.add_argument('--dr', default=0.3, type=float, help="Dropout rate.")
    parser.add_argument('--lr', default=1e-5, type=float, help="Learning rate.")
    parser.add_argument("--hS", default=100, type=int, help="The dimension of hidden vector in the seq-to-SQL module.")


    # 1.4 Execution-guided decoding beam-size. It is used only in test.py
    parser.add_argument('--EG',
                        default=False,
                        action='store_true',
                        help="If present, Execution guided decoding is used in test.")
    parser.add_argument('--beam_size',
                        type=int,
                        default=4,
                        help="The size of beam for smart decoding")


    # 1.5 Arguments only for test.py
    parser.add_argument('--sn', default=42, type=int, help="The targetting session number.")
    parser.add_argument("--target_epoch", default='best', type=str,
                        help="Targer epoch (the save name from nsml).")

    parser.add_argument("--tag", default='', type=str,
                        help="Tag of saved files. e.g.) '', 'FT1', 'FT1_aug', 'no_pretraining', 'no_tuning',..")

    args = parser.parse_args()

    map_bert_type_abb = {'uS': 'uncased_L-12_H-768_A-12',
                         'uL': 'uncased_L-24_H-1024_A-16',
                         'cS': 'cased_L-12_H-768_A-12',
                         'cL': 'cased_L-24_H-1024_A-16',
                         'mcS': 'multi_cased_L-12_H-768_A-12'}
    args.bert_type = map_bert_type_abb[args.bert_type_abb]
    print(f"BERT-type: {args.bert_type}")

    #
    # Decide whether to use lower_case.
    if args.bert_type_abb == 'cS' or args.bert_type_abb == 'cL' or args.bert_type_abb == 'mcS':
        args.do_lower_case = False
    else:
        args.do_lower_case = True

    # args.toy_model = not torch.cuda.is_available()
    args.toy_model = not True
    args.toy_size = 32
    if args.model_type == 'FT_Scalar_1':
        assert args.num_target_layers == 1
        assert args.fine_tune == True


    # Seeds for random number generation.
    python_random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available:
        torch.cuda.manual_seed_all(args.seed)

    return args


def get_bert(BERT_PT_PATH, bert_type, do_lower_case, no_pretraining):


    bert_config_file = os.path.join(BERT_PT_PATH, f'bert_config_{bert_type}.json')
    vocab_file = os.path.join(BERT_PT_PATH, f'vocab_{bert_type}.txt')
    init_checkpoint = os.path.join(BERT_PT_PATH, f'pytorch_model_{bert_type}.bin')



    bert_config = BertConfig.from_json_file(bert_config_file)
    tokenizer = tokenization.FullTokenizer(
        vocab_file=vocab_file, do_lower_case=do_lower_case)
    bert_config.print_status()

    model_bert = BertModel(bert_config)
    if no_pretraining:
        pass
    else:
        model_bert.load_state_dict(torch.load(init_checkpoint, map_location='cpu'))
        print("Load pre-trained parameters.")
    model_bert.to(device)

    return model_bert, tokenizer, bert_config

def get_opt(model, model_bert, model_type):
    if model_type == 'FT_Scalar_1':
        # Model itself does not have trainable parameters. Thus,
        opt_bert = torch.optim.Adam(list(filter(lambda p: p.requires_grad, model.parameters())) \
                               # + list(model_bert.parameters()),
                               + list(filter(lambda p: p.requires_grad, model_bert.parameters())),
                               lr=args.lr, weight_decay=0)
        opt = opt_bert # for consistency in interface
    else:
        raise NotImplementedError
        # opt = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
        #                        lr=args.lr, weight_decay=0)
        #
        # opt_bert = torch.optim.Adam(filter(lambda p: p.requires_grad, model_bert.parameters()),
        #                             lr=args.lr_bert, weight_decay=0)

    return opt, opt_bert

def get_models(args, BERT_PT_PATH, trained=False, path_model_bert=None, path_model=None):
    # some constants
    agg_ops = ['', 'MAX', 'MIN', 'COUNT', 'SUM', 'AVG']
    cond_ops = ['=', '>', '<', 'OP']  # do not know why 'OP' required. Hence,

    print(f"Batch_size = {args.bS * args.accumulate_gradients}")
    print(f"BERT parameters:")
    print(f"learning rate: {args.lr_bert}")
    print(f"Fine-tune BERT: {args.fine_tune}")

    # Get BERT
    model_bert, tokenizer, bert_config = get_bert(BERT_PT_PATH, args.bert_type, args.do_lower_case,
                                                  args.no_pretraining)
    args.iS = bert_config.hidden_size * args.num_target_layers  # Seq-to-SQL input vector dimenstion

    # Get Seq-to-SQL

    n_cond_ops = len(cond_ops)
    n_agg_ops = len(agg_ops)
    print(f"Seq-to-SQL: the number of final BERT layers to be used: {args.num_target_layers}")
    print(f"Seq-to-SQL: the size of hidden dimension = {args.hS}")
    print(f"Seq-to-SQL: LSTM encoding layer size = {args.lS}")
    print(f"Seq-to-SQL: dropout rate = {args.dr}")
    print(f"Seq-to-SQL: learning rate = {args.lr}")
    model = FT_Scalar_1(args.iS, args.hS, args.lS, args.dr, n_cond_ops, n_agg_ops)
    model = model.to(device)

    if trained:
        assert path_model_bert != None
        assert path_model != None

        if nsml.IS_ON_NSML:
            res = torch.load(path_model_bert)
        else:
            res = torch.load(path_model_bert, map_location='cpu')
        model_bert.load_state_dict(res['model_bert'])
        model_bert.to(device)

        if nsml.IS_ON_NSML:
            res = torch.load(path_model)
        else:
            res = torch.load(path_model, map_location='cpu')

        model.load_state_dict(res['model'])

    return model, model_bert, tokenizer, bert_config

def get_data(path_wikisql, args):
    train_data, train_table, dev_data, dev_table, _, _ = load_wikisql(path_wikisql, args.toy_model, args.toy_size,
                                                                      no_w2i=True, no_hs_tok=True,
                                                                      aug=args.aug)
    train_loader, dev_loader = get_loader_wikisql(train_data, dev_data, args.bS, shuffle_train=True)

    return train_data, train_table, dev_data, dev_table, train_loader, dev_loader


def gen_nsml_report(acc_train, aux_out_train, acc_dev, aux_out_dev):
    ave_loss, acc_sc, acc_sa, \
    acc_wn, acc_wc, acc_wo, acc_wvi, acc_wv, \
    acc_lx, acc_x = acc_train

    grad_abs_mean_mean, grad_abs_mean_sig, grad_abs_sig_mean = aux_out_train

    ave_loss_t, acc_sc_t, acc_sa_t, \
    acc_wn_t, acc_wc_t, acc_wo_t, acc_wvi_t, acc_wv_t, \
    acc_lx_t, acc_x_t = acc_dev

    nsml.report(
        step=epoch,
        epoch=epoch,
        epochs_total=args.tepoch,
        train__loss=ave_loss,
        train__acc_sc=acc_sc,
        train__acc_sa=acc_sa,
        train__acc_wn=acc_wn,
        train__acc_wc=acc_wc,
        train__acc_wo=acc_wo,
        train__acc_wvi=acc_wvi,
        train__acc_wv=acc_wv,
        train__acc_lx=acc_lx,
        train__acc_x=acc_x,
        train_grad_abs_mean_mean=float(grad_abs_mean_mean),  # error appeared when numpy.float32 is used
        train_grad_abs_mean_sig=float(grad_abs_mean_sig),
        train_grad_abs_sig_mean=float(grad_abs_sig_mean),
        dev__loss=ave_loss_t,
        dev__acc_sc_t=acc_sc_t,
        dev__acc_sa_t=acc_sa_t,
        dev__acc_wn_t=acc_wn_t,
        dev__acc_wc_t=acc_wc_t,
        dev__acc_wo_t=acc_wo_t,
        dev__acc_wvi_t=acc_wvi_t,
        dev__acc_wv_t=acc_wv_t,
        dev__acc_lx_t=acc_lx_t,
        dev__acc_x=acc_x_t,
        scope=locals()
    )


def train(train_loader, train_table, model, model_bert, opt, bert_config, tokenizer,
          max_seq_length, num_target_layers, accumulate_gradients=1, check_grad=False,
          st_pos=0, opt_bert=None, path_db=None, dset_name='train', col_pool_type='start_tok', aug=False):
    model.train()
    model_bert.train()

    ave_loss = 0
    cnt = 0 # count the # of examples
    cnt_sc = 0 # count the # of correct predictions of select column
    cnt_sa = 0 # of selectd aggregation
    cnt_wn = 0 # of where number
    cnt_wc = 0 # of where column
    cnt_wo = 0 # of where operator
    cnt_wv = 0 # of where-value
    cnt_wvi = 0 # of where-value index (on question tokens)
    cnt_lx = 0  # of logical form acc
    cnt_x = 0   # of execution acc

    # Engine for SQL querying.
    engine = DBEngine(os.path.join(path_db, f"{dset_name}.db"))

    for iB, t in enumerate(train_loader):
        cnt += len(t)

        if cnt < st_pos:
            continue
        # Get fields
        nlu, nlu_t, sql_i, sql_q, sql_t, tb, hs_t, hds = get_fields(t, train_table, no_hs_t=True, no_sql_t=True)
        g_sc, g_sa, g_wn, g_wc, g_wo, g_wv = get_g(sql_i)
        # get ground truth where-value index under CoreNLP tokenization scheme. It's done already on trainset.
        g_wvi_corenlp = get_g_wvi_corenlp(t)


        all_encoder_layer, pooled_output, tokens, i_nlu, i_hds, \
        l_n, l_hpu, l_hs, \
        nlu_tt, t_to_tt_idx, tt_to_t_idx \
            = get_bert_output(model_bert, tokenizer, nlu_t, hds, max_seq_length)

        try:
            #
            g_wvi = get_g_wvi_bert_from_g_wvi_corenlp(t_to_tt_idx, g_wvi_corenlp)
        except:
            # Exception happens when where-condition is not found in nlu_tt.
            # In this case, that train example is not used.
            # During test, that example considered as wrongly answered.
            # e.g. train: 32.
            continue

        wemb_n = get_wemb_n(i_nlu, l_n, bert_config.hidden_size,
                            bert_config.num_hidden_layers, all_encoder_layer, 1)
        wemb_h = get_wemb_h_FT_Scalar_1(i_hds, l_hs, bert_config.hidden_size, all_encoder_layer,
                                        col_pool_type=col_pool_type)
        # wemb_h = [B, max_header_number, hS]
        cls_vec = pooled_output

        # model specific part
        # get g_wvi (it is idex for word-piece tok)
        # score
        s_sc, s_sa, s_wn, s_wc, s_wo, s_wv = model(wemb_n, l_n, wemb_h, l_hs, cls_vec,
                                                   g_sc=g_sc, g_sa=g_sa, g_wn=g_wn, g_wc=g_wc, g_wo=g_wo, g_wvi=g_wvi)

        # Calculate loss & step
        loss = Loss_sw_se(s_sc, s_sa, s_wn, s_wc, s_wo, s_wv, g_sc, g_sa, g_wn, g_wc, g_wo, g_wvi)

        # Calculate gradient
        if iB % accumulate_gradients == 0: # mode
            # at start, perform zero_grad
            opt.zero_grad()
            if opt_bert:
                opt_bert.zero_grad()
            loss.backward()
            if accumulate_gradients == 1:
                opt.step()
                if opt_bert:
                    opt_bert.step()
        elif iB % accumulate_gradients == (accumulate_gradients-1):
            # at the final, take step with accumulated graident
            loss.backward()
            opt.step()
            if opt_bert:
                opt_bert.step()
        else:
            # at intermediate stage, just accumulates the gradients
            loss.backward()

        if check_grad:
            named_parameters = model.named_parameters()

            mu_list, sig_list = get_mean_grad(named_parameters)

            grad_abs_mean_mean = mean(mu_list)
            grad_abs_mean_sig = std(mu_list)
            grad_abs_sig_mean = mean(sig_list)
        else:
            grad_abs_mean_mean = 1
            grad_abs_mean_sig = 1
            grad_abs_sig_mean = 1

        # Prediction
        pr_sc, pr_sa, pr_wn, pr_wc, pr_wo, pr_wvi = pred_sw_se(s_sc, s_sa, s_wn, s_wc, s_wo, s_wv, )
        pr_wv_str, pr_wv_str_wp = convert_pr_wvi_to_string(pr_wvi, nlu_t, nlu_tt, tt_to_t_idx, nlu)
        pr_sql_i = generate_sql_i(pr_sc, pr_sa, pr_wn, pr_wc, pr_wo, pr_wv_str, nlu)


        # Cacluate accuracy
        cnt_sc1_list, cnt_sa1_list, cnt_wn1_list, \
        cnt_wc1_list, cnt_wo1_list, \
        cnt_wvi1_list, cnt_wv1_list = get_cnt_sw_list(g_sc, g_sa, g_wn, g_wc, g_wo, g_wvi,
                                                                   pr_sc, pr_sa, pr_wn, pr_wc, pr_wo, pr_wvi,
                                                                   sql_i, pr_sql_i,
                                                                   mode='train')

        cnt_lx1_list = get_cnt_lx_list(cnt_sc1_list, cnt_sa1_list, cnt_wn1_list, cnt_wc1_list,
                                       cnt_wo1_list, cnt_wv1_list)
        # lx stands for logical form accuracy

        # Execution accuracy test.
        if not aug:
            cnt_x1_list, g_ans, pr_ans = get_cnt_x_list(engine, tb, g_sc, g_sa, sql_i, pr_sc, pr_sa, pr_sql_i)
        else:
            cnt_x1_list = [0] * len(t)
            g_ans = ['N/A (data augmented'] * len(t)
            pr_ans = ['N/A (data augmented'] * len(t)
        # statistics
        ave_loss += loss.item()

        # count
        cnt_sc += sum(cnt_sc1_list)
        cnt_sa += sum(cnt_sa1_list)
        cnt_wn += sum(cnt_wn1_list)
        cnt_wc += sum(cnt_wc1_list)
        cnt_wo += sum(cnt_wo1_list)
        cnt_wvi += sum(cnt_wvi1_list)
        cnt_wv += sum(cnt_wv1_list)
        cnt_lx += sum(cnt_lx1_list)
        cnt_x += sum(cnt_x1_list)

    ave_loss /= cnt
    acc_sc = cnt_sc / cnt
    acc_sa = cnt_sa / cnt
    acc_wn = cnt_wn / cnt
    acc_wc = cnt_wc / cnt
    acc_wo = cnt_wo / cnt
    acc_wvi = cnt_wv / cnt
    acc_wv = cnt_wv / cnt
    acc_lx = cnt_lx / cnt
    acc_x = cnt_x / cnt

    acc = [ave_loss, acc_sc, acc_sa, acc_wn, acc_wc, acc_wo, acc_wvi, acc_wv, acc_lx, acc_x]
    aux_out = [grad_abs_mean_mean, grad_abs_mean_sig, grad_abs_sig_mean]

    return acc, aux_out

def report_detail(hds, nlu,
                  g_sc, g_sa, g_wn, g_wc, g_wo, g_wv, g_wv_str, g_sql_q, g_ans,
                  pr_sc, pr_sa, pr_wn, pr_wc, pr_wo, pr_wv_str, pr_sql_q, pr_ans,
                  cnt_list, current_cnt):
    cnt_tot, cnt, cnt_sc, cnt_sa, cnt_wn, cnt_wc, cnt_wo, cnt_wv, cnt_wvi, cnt_lx, cnt_x = current_cnt

    print(f'cnt = {cnt} / {cnt_tot} ===============================')

    print(f'headers: {hds}')
    print(f'nlu: {nlu}')

    # print(f's_sc: {s_sc[0]}')
    # print(f's_sa: {s_sa[0]}')
    # print(f's_wn: {s_wn[0]}')
    # print(f's_wc: {s_wc[0]}')
    # print(f's_wo: {s_wo[0]}')
    # print(f's_wv: {s_wv[0][0]}')
    print(f'===============================')
    print(f'g_sc : {g_sc}')
    print(f'pr_sc: {pr_sc}')
    print(f'g_sa : {g_sa}')
    print(f'pr_sa: {pr_sa}')
    print(f'g_wn : {g_wn}')
    print(f'pr_wn: {pr_wn}')
    print(f'g_wc : {g_wc}')
    print(f'pr_wc: {pr_wc}')
    print(f'g_wo : {g_wo}')
    print(f'pr_wo: {pr_wo}')
    print(f'g_wv : {g_wv}')
    # print(f'pr_wvi: {pr_wvi}')
    print('g_wv_str:', g_wv_str)
    print('p_wv_str:', pr_wv_str)
    print(f'g_sql_q:  {g_sql_q}')
    print(f'pr_sql_q: {pr_sql_q}')
    print(f'g_ans: {g_ans}')
    print(f'pr_ans: {pr_ans}')
    print(f'--------------------------------')

    print(cnt_list)

    print(f'acc_lx = {cnt_lx/cnt:.3f}, acc_x = {cnt_x/cnt:.3f}\n',
          f'acc_sc = {cnt_sc/cnt:.3f}, acc_sa = {cnt_sa/cnt:.3f}, acc_wn = {cnt_wn/cnt:.3f}\n',
          f'acc_wc = {cnt_wc/cnt:.3f}, acc_wo = {cnt_wo/cnt:.3f}, acc_wv = {cnt_wv/cnt:.3f}')
    print(f'===============================')

def test(data_loader, data_table, model, model_bert, bert_config, tokenizer,
         max_seq_length,
         num_target_layers, detail=False, st_pos=0, cnt_tot=1, EG=False, beam_size=4,
         path_db=None, dset_name='test', col_pool_type='start_tok', aug=False):
    model.eval()
    model_bert.eval()

    ave_loss = 0
    cnt = 0
    cnt_sc = 0
    cnt_sa = 0
    cnt_wn = 0
    cnt_wc = 0
    cnt_wo = 0
    cnt_wv = 0
    cnt_wvi = 0
    cnt_lx = 0
    cnt_x = 0

    cnt_list = []
    p_list = [] # List of prediction probabilities.
    data_list = [] # Miscellanerous data. Save it for later convenience of analysis.

    engine = DBEngine(os.path.join(path_db, f"{dset_name}.db"))
    results = []
    for iB, t in enumerate(data_loader):

        cnt += len(t)
        if cnt < st_pos:
            continue
        # Get fields
        nlu, nlu_t, sql_i, sql_q, sql_t, tb, hs_t, hds = get_fields(t, data_table, no_hs_t=True, no_sql_t=True)

        g_sc, g_sa, g_wn, g_wc, g_wo, g_wv = get_g(sql_i)
        g_wvi_corenlp = get_g_wvi_corenlp(t)

        all_encoder_layer, pooled_output, tokens, i_nlu, i_hds, \
        l_n, l_hpu, l_hs, \
        nlu_tt, t_to_tt_idx, tt_to_t_idx \
            = get_bert_output(model_bert, tokenizer, nlu_t, hds, max_seq_length)

        try:
            g_wvi = get_g_wvi_bert_from_g_wvi_corenlp(t_to_tt_idx, g_wvi_corenlp)
            g_wv_str, g_wv_str_wp = convert_pr_wvi_to_string(g_wvi, nlu_t, nlu_tt, tt_to_t_idx, nlu)

        except:
            # Exception happens when where-condition is not found in nlu_tt.
            # In this case, that train example is not used.
            # During test, that example considered as wrongly answered.
            for b in range(len(nlu)):
                results1 = {}
                results1["error"] = "Skip happened"
                results1["nlu"] = nlu[b]
                results1["table_id"] = tb[b]["id"]
                results.append(results1)
            continue

        # model specific part
        # score
        wemb_n = get_wemb_n(i_nlu, l_n, bert_config.hidden_size,
                            bert_config.num_hidden_layers, all_encoder_layer, 1)
        wemb_h = get_wemb_h_FT_Scalar_1(i_hds, l_hs, bert_config.hidden_size, all_encoder_layer,
                                        col_pool_type=col_pool_type)
        # wemb_h = [B, max_header_number, hS]
        cls_vec = pooled_output
        # No Execution guided decoding
        if not EG:

            s_sc, s_sa, s_wn, s_wc, s_wo, s_wv = model(wemb_n, l_n, wemb_h, l_hs, cls_vec)

            # get loss & step
            loss = Loss_sw_se(s_sc, s_sa, s_wn, s_wc, s_wo, s_wv, g_sc, g_sa, g_wn, g_wc, g_wo, g_wvi)

            # prediction
            pr_sc, pr_sa, pr_wn, pr_wc, pr_wo, pr_wvi = pred_sw_se(s_sc, s_sa, s_wn, s_wc, s_wo, s_wv,)
            pr_wv_str, pr_wv_str_wp = convert_pr_wvi_to_string(pr_wvi, nlu_t, nlu_tt, tt_to_t_idx, nlu)
            # g_sql_i = generate_sql_i(g_sc, g_sa, g_wn, g_wc, g_wo, g_wv_str, nlu)
            pr_sql_i = generate_sql_i(pr_sc, pr_sa, pr_wn, pr_wc, pr_wo, pr_wv_str, nlu)

            # calculate probability
            p_tot, p_select, p_where, p_sc, p_sa, p_wn, p_wc, p_wo, p_wvi \
                = cal_prob(s_sc, s_sa, s_wn, s_wc, s_wo, s_wv,
                           pr_sc, pr_sa, pr_wn, pr_wc, pr_wo, pr_wvi)

        else:
            # Execution guided decoding
            pr_sc_best, pr_sa_best, pr_wn_based_on_prob, pr_wvi_best, \
            pr_sql_i, p_tot, p_select, p_where, p_sc_best, p_sa_best, \
            p_wn_best, p_wc_best, p_wo_best, p_wvi_best\
                = model.forward_EG(wemb_n, l_n, wemb_h, l_hs, cls_vec, engine, tb,
                   nlu_t, nlu_tt, tt_to_t_idx, nlu,
                   beam_size=beam_size)


            pr_sc = pr_sc_best
            pr_sa = pr_sa_best
            pr_wn = pr_wn_based_on_prob

            p_sc = p_sc_best
            p_sa = p_sa_best
            p_wn = p_wn_best

            # sort and generate: prob-based-sort (descending) -> wc-idx-based-sort (ascending)
            pr_wc, pr_wo, pr_wv_str, pr_wvi, pr_sql_i, \
            p_wc, p_wo, p_wvi = sort_and_generate_pr_w(pr_sql_i, pr_wvi_best, p_wc_best, p_wo_best, p_wvi_best)

            # Follosing variables are just for the consistency with no-EG case.
            pr_wv_str_wp=None
            loss = torch.tensor([0])

        p_list_batch = [p_tot, p_select, p_where, p_sc, p_sa, p_wn, p_wc, p_wo, p_wvi ]
        p_list.append(p_list_batch)

        g_sql_q = generate_sql_q(sql_i, tb)
        pr_sql_q = generate_sql_q(pr_sql_i, tb)

        # Saving for the official evaluation later.
        for b, pr_sql_i1 in enumerate(pr_sql_i):
            results1 = {}
            results1["query"] = pr_sql_i1
            results1["table_id"] = tb[b]["id"]
            results1["nlu"] = nlu[b]
            results.append(results1)

        cnt_sc1_list, cnt_sa1_list, cnt_wn1_list, \
        cnt_wc1_list, cnt_wo1_list, \
        cnt_wvi1_list, cnt_wv1_list = get_cnt_sw_list(g_sc, g_sa,g_wn, g_wc,g_wo, g_wvi,
                                                                   pr_sc, pr_sa, pr_wn, pr_wc, pr_wo, pr_wvi,
                                                                   sql_i, pr_sql_i,
                                                                   mode='test')

        cnt_lx1_list = get_cnt_lx_list(cnt_sc1_list, cnt_sa1_list, cnt_wn1_list, cnt_wc1_list,
                                       cnt_wo1_list, cnt_wv1_list)

        # Execution accura y test
        cnt_x1_list = []
        # lx stands for logical form accuracy

        # Execution accuracy test.
        if not aug:
            cnt_x1_list, g_ans, pr_ans = get_cnt_x_list(engine, tb, g_sc, g_sa, sql_i, pr_sc, pr_sa, pr_sql_i)
        else:
            cnt_x1_list = [0] * len(t)
            g_ans = ['N/A (data augmented'] * len(t)
            pr_ans = ['N/A (data augmented'] * len(t)
        # stat
        ave_loss += loss.item()

        # count
        cnt_sc += sum(cnt_sc1_list)
        cnt_sa += sum(cnt_sa1_list)
        cnt_wn += sum(cnt_wn1_list)
        cnt_wc += sum(cnt_wc1_list)
        cnt_wo += sum(cnt_wo1_list)
        cnt_wv += sum(cnt_wv1_list)
        cnt_wvi += sum(cnt_wvi1_list)
        cnt_lx += sum(cnt_lx1_list)
        cnt_x += sum(cnt_x1_list)

        current_cnt = [cnt_tot, cnt, cnt_sc, cnt_sa, cnt_wn, cnt_wc, cnt_wo, cnt_wv, cnt_wvi, cnt_lx, cnt_x]
        cnt_list_batch = [cnt_sc1_list, cnt_sa1_list, cnt_wn1_list, cnt_wc1_list, cnt_wo1_list, cnt_wv1_list, cnt_lx1_list,
                     cnt_x1_list]
        cnt_list.append(cnt_list_batch)
        # report
        if detail:
            report_detail(hds, nlu,
                          g_sc, g_sa, g_wn, g_wc, g_wo, g_wv, g_wv_str, g_sql_q, g_ans,
                          pr_sc, pr_sa, pr_wn, pr_wc, pr_wo, pr_wv_str, pr_sql_q, pr_ans,
                          cnt_list_batch, current_cnt)
        data_batch = []
        for b, nlu1 in enumerate(nlu):
            data1 = [nlu[b], nlu_t[b], sql_i[b], g_sql_q[b], g_ans[b],
                     pr_sql_i[b], pr_sql_q[b], pr_ans[b], tb[b]]
            data_batch.append(data1)

        data_list.append(data_batch)

    ave_loss /= cnt
    acc_sc = cnt_sc / cnt
    acc_sa = cnt_sa / cnt
    acc_wn = cnt_wn / cnt
    acc_wc = cnt_wc / cnt
    acc_wo = cnt_wo / cnt
    acc_wvi = cnt_wvi / cnt
    acc_wv = cnt_wv / cnt
    acc_lx = cnt_lx / cnt
    acc_x = cnt_x / cnt

    acc = [ave_loss, acc_sc, acc_sa, acc_wn, acc_wc, acc_wo, acc_wvi, acc_wv, acc_lx, acc_x]
    return acc, results, cnt_list, p_list, data_list


def print_result(epoch, acc, dname):
    ave_loss, acc_sc, acc_sa, acc_wn, acc_wc, acc_wo, acc_wvi, acc_wv, acc_lx, acc_x = acc

    print(f'{dname} results ------------')
    print(
        f" Epoch: {epoch}, ave loss: {ave_loss}, acc_sc: {acc_sc:.3f}, acc_sa: {acc_sa:.3f}, acc_wn: {acc_wn:.3f}, \
        acc_wc: {acc_wc:.3f}, acc_wo: {acc_wo:.3f}, acc_wvi: {acc_wvi:.3f}, acc_wv: {acc_wv:.3f}, acc_lx: {acc_lx:.3f}, acc_x: {acc_x:.3f}"
    )

if __name__ == '__main__':

    ## 1. Hyper parameters
    parser = argparse.ArgumentParser()
    args = construct_hyper_param(parser)

    ## 2. Paths
    path_h = '/home/wonseok'
    path_wikisql = os.path.join(path_h, 'data', 'wikisql_tok')
    BERT_PT_PATH = path_wikisql

    path_save_for_evaluation = './'

    ## 3. Load data
    train_data, train_table, dev_data, dev_table, train_loader, dev_loader = get_data(path_wikisql, args)

    ## 4. Build & Load models
    model, model_bert, tokenizer, bert_config = get_models(args, BERT_PT_PATH)

    # nsml binding

    ## 5. Get optimizers
    opt, opt_bert = get_opt(model, model_bert, args.model_type)

    ## 6. Train
    acc_lx_t_best = -1
    epoch_best = -1
    for epoch in range(args.tepoch):
        # train
        acc_train, aux_out_train = train(train_loader,
                                         train_table,
                                         model,
                                         model_bert,
                                         opt,
                                         bert_config,
                                         tokenizer,
                                         args.max_seq_length,
                                         args.num_target_layers,
                                         args.accumulate_gradients,
                                         opt_bert=opt_bert,
                                         st_pos=0,
                                         path_db=path_wikisql,
                                         dset_name='train',
                                         col_pool_type=args.col_pool_type,
                                         aug=args.aug)

        # check DEV
        with torch.no_grad():
            acc_dev, results_dev, cnt_list_dev, p_list_dev, data_list_dev = test(dev_loader,
                                                dev_table,
                                                model,
                                                model_bert,
                                                bert_config,
                                                tokenizer,
                                                args.max_seq_length,
                                                args.num_target_layers,
                                                detail=False,
                                                path_db=path_wikisql,
                                                st_pos=0,
                                                dset_name='dev', EG=args.EG,
                                                col_pool_type=args.col_pool_type,
                                                beam_size=args.beam_size,
                                                aug=args.aug)


        print_result(epoch, acc_train, 'train')
        print_result(epoch, acc_dev, 'dev')

        # save results for the offical evaluation
        save_for_evaluation(path_save_for_evaluation, results_dev, 'dev')

        # save best model
        # Based on Dev Set logical accuracy lx
        acc_lx_t = acc_dev[-2]
        if acc_lx_t > acc_lx_t_best:
            acc_lx_t_best = acc_lx_t
            epoch_best = epoch
            # save best model
            state = {'model': model.state_dict()}
            torch.save(state, os.path.join('.', 'model_best.pt'))

            state = {'model_bert': model_bert.state_dict()}
            torch.save(state, os.path.join('.', 'model_bert_best.pt'))

        print(f" Best Dev lx acc: {acc_lx_t_best} at epoch: {epoch_best}")
