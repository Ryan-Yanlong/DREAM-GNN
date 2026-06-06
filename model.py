from layers import *
th.set_printoptions(profile="full")

class Net(nn.Module):
    def __init__(self, args):
        super(Net, self).__init__()
        self.layers = args.layers
        self._act = get_activation(args.model_activation)
        self.TGCN = nn.ModuleList()
        self.TGCN.append(GCMCLayer(args.rating_vals,  # [0, 1]
                                   args.src_in_units,   
                                   args.dst_in_units,  
                                   args.gcn_agg_units,  
                                   args.gcn_out_units,  
                                   args.dropout,        
                                   args.gcn_agg_accum,  
                                   agg_act=self._act,  
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
        
        self.attention = Attention(args.gcn_out_units, dropout_rate=args.attention_dropout)

        # Fusion of topology (TGCN) and feature (FGCN) channels:
        #   'attn'   -> attention convex-combination (original, decoder input = gcn_out_units)
        #   'concat' -> keep both channels (decoder input = 2 * gcn_out_units)
        self.fusion = getattr(args, 'fusion', 'attn')
        dec_in = args.gcn_out_units if self.fusion == 'attn' else 2 * args.gcn_out_units

        # Decoder: plain MLP, or GBA decoder that also consumes leakage-free pair features.
        self.use_gba = getattr(args, 'use_gba', False)
        if self.use_gba:
            self.decoder = GBADecoder(dec_in, n_pair=getattr(args, 'n_pair', 2), dropout_rate=args.dropout)
        else:
            self.decoder = MLPDecoder(in_units=dec_in, dropout_rate=args.dropout)
        self.rating_vals = args.rating_vals

    def forward(self, enc_graph, dec_graph,
                drug_graph, drug_sim_feat, drug_feat,
                dis_graph, disease_sim_feat, dis_feat,
                drug_feature_graph=None, disease_feature_graph=None,
                Two_Stage=False, pair_feat=None):

        # Topology convolution operation
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
        
        # Fuse topology (TGCN) and feature (FGCN) channels
        if self.fusion == 'concat':
            # keep both channels; decoder input width is 2 * gcn_out_units
            drug_feats = th.cat([drug_out, drug_sim_out], dim=1)
            dis_feats = th.cat([dis_out, dis_sim_out], dim=1)
        else:
            # attention convex-combination (original)
            drug_feats, att_drug = self.attention(th.stack([drug_out, drug_sim_out], dim=1))
            dis_feats, att_dis = self.attention(th.stack([dis_out, dis_sim_out], dim=1))
        
        # Decode (GBA decoder also consumes leakage-free pair features)
        if self.use_gba:
            pred_ratings = self.decoder(dec_graph, drug_feats, dis_feats, pair_feat)
        else:
            pred_ratings = self.decoder(dec_graph, drug_feats, dis_feats)
        
        # Return all intermediate representations for supervision or analysis
        return pred_ratings, drug_out, drug_sim_out, dis_out, dis_sim_out