from layers import *
th.set_printoptions(profile="full")

class Net(nn.Module):
    def __init__(self, args):
        super(Net, self).__init__()
        self.layers = args.layers
        self._act = get_activation(args.model_activation)
        self.TGCN = nn.ModuleList()
        self.TGCN.append(GCMCLayer(args.rating_vals,  # [0, 1]
                                   args.src_in_units,   # e.g., 768
                                   args.dst_in_units,   # e.g., 768
                                   args.gcn_agg_units,  # e.g., 700
                                   args.gcn_out_units,  # e.g., 75
                                   args.dropout,        # e.g., 0.3
                                   args.gcn_agg_accum,  # e.g., "sum"
                                   agg_act=self._act,   # should be "leaky"
                                   share_user_item_param=args.share_param,
                                   device=args.device))
        self.gcn_agg_accum = args.gcn_agg_accum
        self.rating_vals = args.rating_vals
        self.device = args.device
        self.gcn_agg_units = args.gcn_agg_units
        self.src_in_units = args.src_in_units
        for i in range(1, args.layers):
            if args.gcn_agg_accum == 'stack':
                gcn_out_units = args.gcn_out_units * len(args.rating_vals)
            else:
                gcn_out_units = args.gcn_out_units
            self.TGCN.append(GCMCLayer(args.rating_vals,
                                       args.gcn_out_units,
                                       args.gcn_out_units,
                                       gcn_out_units,
                                       args.gcn_out_units,
                                       args.dropout,
                                       args.gcn_agg_accum,
                                       agg_act=self._act,
                                       share_user_item_param=args.share_param,
                                       ini=False,
                                       device=args.device))
        
        self.FGCN = FGCN(args.fdim_drug,
                         args.fdim_disease,
                         args.nhid1,
                         args.nhid2,
                         args.dropout)
        
        # Add a new argument to control whether to use gated attention fusion.
        # 在 Net 类的 __init__ 方法中修改
        if False:
            self.gatedfusion = GatedMultimodalLayer(args.gcn_out_units,
                                                    args.gcn_out_units,
                                                    args.gcn_out_units)
        else:
            # 修改 Attention 类的输入维度（拓扑特征+相似性图特征+特征相似性图特征）
            self.attention = Attention(args.gcn_out_units, dropout_rate=args.attention_dropout)
        
        self.decoder = MLPDecoder(in_units=args.gcn_out_units, dropout_rate=args.dropout)
        self.rating_vals = args.rating_vals

    def forward(self, enc_graph, dec_graph,
                drug_graph, drug_sim_feat, drug_feat,
                dis_graph, disease_sim_feat, dis_feat,
                drug_feature_graph=None, disease_feature_graph=None,
                Two_Stage=False):

        # Topology convolution operation (保持不变)
        for i in range(0, self.layers):
            drug_o, dis_o = self.TGCN[i](enc_graph, drug_feat, dis_feat, Two_Stage)
            if i == 0:
                drug_out = drug_o
                dis_out = dis_o
            else:
                drug_out = drug_out + drug_o / float(i + 1)
                dis_out = dis_out + dis_o / float(i + 1)
            drug_feat = drug_o
            dis_feat = dis_o

        # Feature convolution operation with multiple graphs
        drug_sim_out, dis_sim_out, drug_sim_only, drug_feat_only, dis_sim_only, dis_feat_only = self.FGCN(
            drug_graph, drug_sim_feat,
            dis_graph, disease_sim_feat,
            drug_feature_graph, disease_feature_graph
        )
        
        # 修改后的融合方式，处理多种图嵌入
        if False:
            # 如果启用门控注意力，融合拓扑嵌入和多图特征嵌入
            drug_feats = self.gatedfusion(drug_out, drug_sim_out)
            dis_feats = self.gatedfusion(dis_out, dis_sim_out)
        else:
            # 否则使用堆叠+注意力的方式融合
            # 可以选择包含所有单独的嵌入，或只包括最终融合的嵌入


            drug_feats = th.stack([drug_out, drug_sim_out], dim=1)
            
            drug_feats, att_drug = self.attention(drug_feats)

            dis_feats = th.stack([dis_out, dis_sim_out], dim=1)
            
            dis_feats, att_dis = self.attention(dis_feats)
        
        # Decode (保持不变)
        pred_ratings = self.decoder(dec_graph, drug_feats, dis_feats)
        
        # 返回所有中间表示，便于监督或分析
        return pred_ratings, drug_out, drug_sim_out, dis_out, dis_sim_out