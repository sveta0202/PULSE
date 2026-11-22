# Все эксперименты регрессора (leave-one-day-out CV)

Сгенерировано из `nn_results.jsonl`. Дефолты: hidden=64, depth=5, k=5, drop=0.2, gru=1 (BiGRU), steps=1000, crop=400, gain_aug=0.5, feat=env. Самые ранние прогоны шли с gain_aug=0.2.
`gru`: 0 — только TCN, 1 — BiGRU, 2 — BiLSTM×2, 3 — Transformer, 4 — U-Net, 5 — U-Net+BiLSTM. `gbm_base` — LightGBM (`gbm_cv.py`), `rt_*` — retrieval cross-attention (`retr.py`). `pseudo_src=ens_oof_v2.npy` — студенты транздуктивной дистилляции.

| tag | CV wMAE (oracle action) | non-default args | Open | Little | Fist |
|---|---|---|---|---|---|
| epl_raw_h96_s1500 | 0.0897 | hidden=96, gru=2, steps=1500, feat=envraw, seeds=2, pseudo=1, pseudo_src=ens_oof_v2.npy, chdrop=0.15 | 0.106 | 0.122 | 0.094 |
| epl_raw_h96_s1500_con | 0.0898 | hidden=96, drop=0.3, gru=2, steps=1500, feat=envraw, seeds=2, pseudo=1, pseudo_src=ens_oof_v2.npy, contrast=0.2, chdrop=0.15 | 0.105 | 0.122 | 0.093 |
| epl_lstmraw_s1500 | 0.0901 | drop=0.3, gru=2, steps=1500, feat=envraw, seeds=2, pseudo=1, pseudo_src=ens_oof_v2.npy, chdrop=0.15 | 0.107 | 0.121 | 0.093 |
| epl_raw_h96_s2000 | 0.0902 | hidden=96, drop=0.3, gru=2, steps=2000, feat=envraw, seeds=2, pseudo=1, pseudo_src=ens_oof_v2.npy, chdrop=0.15 | 0.108 | 0.123 | 0.094 |
| epl_tan_h96 | 0.0903 | hidden=96, gru=2, feat=envtan, seeds=2, pseudo=1, pseudo_src=ens_oof_v2.npy | 0.105 | 0.123 | 0.095 |
| epl_corr_lstm | 0.0904 | gru=2, feat=envcorr, seeds=2, pseudo=1, pseudo_src=ens_oof_v2.npy | 0.105 | 0.124 | 0.094 |
| epl_tan | 0.0905 | gru=2, feat=envtan, seeds=2, pseudo=1, pseudo_src=ens_oof_v2.npy | 0.105 | 0.124 | 0.095 |
| epl_corr_lstm_s1500 | 0.0905 | drop=0.3, gru=2, steps=1500, feat=envcorr, seeds=2, pseudo=1, pseudo_src=ens_oof_v2.npy | 0.106 | 0.124 | 0.094 |
| epl_corr_h96 | 0.0905 | hidden=96, gru=2, feat=envcorr, seeds=2, pseudo=1, pseudo_src=ens_oof_v2.npy | 0.106 | 0.124 | 0.095 |
| epl_tan_s1500 | 0.0906 | drop=0.3, gru=2, steps=1500, feat=envtan, seeds=2, pseudo=1, pseudo_src=ens_oof_v2.npy | 0.105 | 0.124 | 0.095 |
| epl_raw_h96 | 0.0906 | hidden=96, gru=2, feat=envraw, seeds=2, pseudo=1, pseudo_src=ens_oof_v2.npy, chdrop=0.15 | 0.107 | 0.122 | 0.094 |
| epl_raw_w2 | 0.0908 | gru=2, feat=envraw, seeds=2, pseudo=1, pseudo_src=ens_oof_v2.npy, pseudo_w=2.0, chdrop=0.15 | 0.109 | 0.121 | 0.094 |
| epl_raw_w1 | 0.0909 | gru=2, feat=envraw, seeds=2, pseudo=1, pseudo_src=ens_oof_v2.npy, chdrop=0.15 | 0.108 | 0.121 | 0.094 |
| r2_corr | 0.0913 | feat=envcorr, seeds=2, pseudo=1, pseudo_src=teacher_r1.npy | 0.108 | 0.124 | 0.095 |
| epl_corr | 0.0914 | feat=envcorr, seeds=2, pseudo=1, pseudo_src=ens_oof_v2.npy | 0.105 | 0.126 | 0.096 |
| epl_w2 | 0.0919 | gru=2, seeds=2, pseudo=1, pseudo_src=ens_oof_v2.npy, pseudo_w=2.0 | 0.111 | 0.124 | 0.094 |
| epl_envd | 0.0919 | gru=2, feat=envd, seeds=2, pseudo=1, pseudo_src=ens_oof_v2.npy | 0.113 | 0.124 | 0.094 |
| epl_w1 | 0.0920 | gru=2, seeds=2, pseudo=1, pseudo_src=ens_oof_v2.npy | 0.112 | 0.125 | 0.094 |
| epl_con02 | 0.0921 | gru=2, seeds=2, pseudo=1, pseudo_src=ens_oof_v2.npy, contrast=0.2 | 0.112 | 0.124 | 0.094 |
| r2_lstm_env | 0.0921 | gru=2, seeds=2, pseudo=1, pseudo_src=teacher_r1.npy | 0.113 | 0.124 | 0.093 |
| epl_ssl | 0.0923 | gru=2, seeds=2, pseudo=1, pseudo_src=ens_oof_v2.npy, ssl_steps=1500 | 0.11 | 0.125 | 0.095 |
| epl_raw_gru | 0.0927 | feat=envraw, seeds=2, pseudo=1, pseudo_src=ens_oof_v2.npy, chdrop=0.15 | 0.109 | 0.123 | 0.097 |
| epl_w05 | 0.0930 | gru=2, seeds=2, pseudo=1, pseudo_src=ens_oof_v2.npy, pseudo_w=0.5 | 0.114 | 0.125 | 0.094 |
| epl_gru | 0.0937 | seeds=2, pseudo=1, pseudo_src=ens_oof_v2.npy | 0.11 | 0.126 | 0.096 |
| epl_raw_tf | 0.0943 | gru=3, feat=envraw, seeds=2, pseudo=1, pseudo_src=ens_oof_v2.npy, chdrop=0.15 | 0.108 | 0.123 | 0.102 |
| r18_lstm_ssl3000 | 0.0950 | gru=2, seeds=2, ssl_steps=3000 | 0.113 | 0.126 | 0.095 |
| r17_lstm_envd_s3 | 0.0952 | gru=2, feat=envd, seeds=3 | 0.118 | 0.129 | 0.096 |
| r15_lstm_ssl1500 | 0.0952 | gru=2, seeds=2, ssl_steps=1500 | 0.112 | 0.129 | 0.094 |
| r17_lstm_envd_ssl500 | 0.0953 | gru=2, feat=envd, seeds=3, ssl_steps=500 | 0.115 | 0.129 | 0.098 |
| c_lstm_ctx_c400 | 0.0953 | gru=2, seeds=2, ctx=1 | 0.112 | 0.13 | 0.099 |
| c_lstm_ctx_envd | 0.0953 | gru=2, feat=envd, seeds=2, ctx=1 | 0.117 | 0.128 | 0.1 |
| r11_lstm_s1500 | 0.0954 | drop=0.3, gru=2, steps=1500, seeds=2 | 0.115 | 0.131 | 0.099 |
| r14_lstm_envd | 0.0954 | gru=2, feat=envd, seeds=2 | 0.114 | 0.129 | 0.099 |
| r15_lstm_ssl500 | 0.0954 | gru=2, seeds=2, ssl_steps=500 | 0.114 | 0.129 | 0.094 |
| k_con02 | 0.0955 | gru=2, seeds=2, contrast=0.2 | 0.115 | 0.127 | 0.094 |
| c_lstm_ctx_s2 | 0.0956 | gru=2, seeds=2, ctx=1 | 0.115 | 0.131 | 0.098 |
| r11_lstm_pseudo | 0.0957 | gru=2, seeds=2, pseudo=1, pseudo_w=0.5 | 0.116 | 0.129 | 0.097 |
| r17_lstm_ssl500_s3 | 0.0957 | gru=2, seeds=3, ssl_steps=500 | 0.115 | 0.13 | 0.096 |
| r13_lstm_raw_s1500 | 0.0958 | drop=0.3, gru=2, steps=1500, feat=envraw, seeds=2, chdrop=0.15 | 0.107 | 0.129 | 0.099 |
| n1_tta8 | 0.0959 | gru=2, seeds=2, tta=8 | 0.114 | 0.13 | 0.098 |
| e_moe | 0.0959 | gru=2, seeds=2, moe=1 | 0.119 | 0.133 | 0.096 |
| k_con01 | 0.0959 | gru=2, seeds=2, contrast=0.1 | 0.115 | 0.129 | 0.096 |
| u_lstm_d7 | 0.0959 | depth=7, gru=2, seeds=2 | 0.118 | 0.126 | 0.1 |
| r11_lstm_chd | 0.0961 | gru=2, seeds=2, chdrop=0.15 | 0.12 | 0.126 | 0.094 |
| aw_065 | 0.0961 | gru=2, seeds=2, active_w=0.65 | 0.116 | 0.13 | 0.096 |
| d_dann003 | 0.0961 | gru=2, seeds=2, dann=0.03 | 0.115 | 0.13 | 0.098 |
| r14_lstm_ref | 0.0962 | gru=2, seeds=2 | 0.113 | 0.131 | 0.097 |
| n1_mixch01 | 0.0962 | gru=2, seeds=2, mixch=0.1 | 0.118 | 0.128 | 0.095 |
| e_moe_wd05 | 0.0962 | gru=2, wd=0.05, seeds=2, moe=1 | 0.119 | 0.134 | 0.097 |
| e_moe | 0.0962 | gru=2, seeds=2 | 0.114 | 0.131 | 0.097 |
| r8_lstm | 0.0963 | gru=2, seeds=2 | 0.114 | 0.131 | 0.097 |
| r17_lstm_ref_s3 | 0.0963 | gru=2, seeds=3 | 0.115 | 0.13 | 0.099 |
| r11_lstm_raw_chd | 0.0964 | gru=2, feat=envraw, seeds=2, chdrop=0.15 | 0.106 | 0.128 | 0.099 |
| r11_lstm_h96 | 0.0964 | hidden=96, gru=2, seeds=2 | 0.118 | 0.131 | 0.097 |
| r13_lstm_crop800 | 0.0964 | drop=0.3, gru=2, steps=1500, bs=16, crop=800, seeds=2 | 0.122 | 0.129 | 0.097 |
| r14_lstm_vel05 | 0.0966 | gru=2, seeds=2, vel_w=0.5 | 0.116 | 0.128 | 0.102 |
| aw_035 | 0.0966 | gru=2, seeds=2, active_w=0.35 | 0.114 | 0.132 | 0.099 |
| d_dann01 | 0.0966 | gru=2, seeds=2, dann=0.1 | 0.117 | 0.131 | 0.099 |
| aw_02 | 0.0967 | gru=2, seeds=2, active_w=0.2 | 0.115 | 0.13 | 0.095 |
| aw_08 | 0.0967 | gru=2, seeds=2, active_w=0.8 | 0.116 | 0.131 | 0.098 |
| k_con04 | 0.0967 | gru=2, seeds=2, contrast=0.4 | 0.117 | 0.129 | 0.097 |
| rt_envd | 0.0967 | heads=4, bs=16, mem_k=2, mem_len=600, feat=envd, seeds=2 | 0.117 | 0.128 | 0.096 |
| k_con02_raw | 0.0968 | gru=2, feat=envraw, seeds=2, contrast=0.2, chdrop=0.15 | 0.106 | 0.125 | 0.101 |
| c_lstm_ctx_raw | 0.0969 | gru=2, feat=envraw, seeds=2, ctx=1, chdrop=0.15 | 0.107 | 0.125 | 0.103 |
| g05_seeds3 | 0.0970 | seeds=3 | 0.116 | 0.13 | 0.098 |
| r5_pseudo1_w05 | 0.0970 | seeds=2, pseudo=1, pseudo_w=0.5 | 0.116 | 0.129 | 0.097 |
| e_moe_envd | 0.0971 | gru=2, feat=envd, seeds=2, moe=1 | 0.131 | 0.128 | 0.098 |
| rt_base | 0.0971 | heads=4, bs=16, mem_k=2, mem_len=600, seeds=2 | 0.121 | 0.13 | 0.097 |
| r5_pseudo1 | 0.0972 | seeds=2, pseudo=1 | 0.118 | 0.128 | 0.097 |
| p_lstm_pca8 | 0.0974 | gru=2, seeds=2, pca_k=8 | 0.119 | 0.137 | 0.097 |
| r4_ref | 0.0976 | seeds=2 | 0.116 | 0.131 | 0.098 |
| r4_chd | 0.0978 | seeds=2, chdrop=0.15 | 0.121 | 0.127 | 0.1 |
| d_dann03 | 0.0978 | gru=2, seeds=2, dann=0.3 | 0.128 | 0.13 | 0.097 |
| rt_k4 | 0.0980 | heads=4, bs=16, mem_k=4, mem_len=400, seeds=2 | 0.12 | 0.129 | 0.097 |
| r4_st01 | 0.0982 | seeds=2, stretch=0.1 | 0.12 | 0.13 | 0.098 |
| r4_st02 | 0.0982 | seeds=2, stretch=0.2 | 0.12 | 0.13 | 0.099 |
| r4_lr1e3 | 0.0982 | lr=0.001, steps=2000, seeds=2 | 0.121 | 0.13 | 0.101 |
| r10_raw_chd_pseudo | 0.0982 | feat=envraw, seeds=2, pseudo=1, pseudo_w=0.5, chdrop=0.15 | 0.116 | 0.127 | 0.102 |
| r15_lstm_raw_ssl1000 | 0.0983 | gru=2, feat=envraw, seeds=2, ssl_steps=1000, chdrop=0.15 | 0.115 | 0.124 | 0.102 |
| r13_lstm_s2500 | 0.0984 | drop=0.3, gru=2, steps=2500, seeds=2 | 0.125 | 0.137 | 0.101 |
| r4_k7d6 | 0.0986 | depth=6, k=7, seeds=2 | 0.12 | 0.129 | 0.102 |
| n1_cfg_d02_w0 | 0.0986 | gru=2, seeds=2, cfg_drop=0.2 | 0.117 | 0.13 | 0.096 |
| gain05 | 0.0987 |  | 0.117 | 0.132 | 0.099 |
| r10_tf_raw_chd | 0.0987 | gru=3, feat=envraw, seeds=2, chdrop=0.15 | 0.113 | 0.127 | 0.108 |
| g05_ema | 0.0989 | ema=0.995 | 0.118 | 0.131 | 0.099 |
| c_lstm_ctx_c1900 | 0.0989 | gru=2, steps=1500, bs=8, crop=1900, seeds=2, ctx=1 | 0.134 | 0.131 | 0.096 |
| r10_raw_chd3 | 0.0989 | feat=envraw, seeds=2, chdrop=0.3 | 0.112 | 0.127 | 0.105 |
| r9_raw_chd | 0.0991 | feat=envraw, seeds=2, chdrop=0.15 | 0.113 | 0.13 | 0.103 |
| c_lstm_ctx_c1000 | 0.0993 | gru=2, bs=16, crop=1000, seeds=2, ctx=1 | 0.132 | 0.132 | 0.098 |
| r8_lstm_raw | 0.0994 | gru=2, feat=envraw, seeds=2 | 0.106 | 0.138 | 0.102 |
| r18_gru_ssl1500 | 0.0995 | seeds=2, ssl_steps=1500 | 0.12 | 0.131 | 0.104 |
| u_unet_lstm_h32 | 0.0998 | hidden=32, gru=5, seeds=2 | 0.116 | 0.139 | 0.103 |
| g05_ln01 | 0.0999 | label_noise=0.1 | 0.118 | 0.132 | 0.1 |
| r14_lstm_mix04 | 0.1000 | gru=2, seeds=2, mixup=0.4 | 0.124 | 0.136 | 0.102 |
| r12_env_chd3 | 0.1000 | seeds=2, chdrop=0.3 | 0.129 | 0.128 | 0.102 |
| p_lstm_pca5 | 0.1001 | gru=2, seeds=2, pca_k=5 | 0.119 | 0.142 | 0.104 |
| r9_h32 | 0.1006 | hidden=32, seeds=2 | 0.118 | 0.133 | 0.104 |
| g05_s2000 | 0.1007 | steps=2000 | 0.131 | 0.134 | 0.103 |
| u_unet_h32 | 0.1011 | hidden=32, gru=4, seeds=2 | 0.118 | 0.14 | 0.112 |
| c_lstm_c1900 | 0.1012 | gru=2, steps=1500, bs=8, crop=1900, seeds=2 | 0.142 | 0.132 | 0.098 |
| n1_mixch03 | 0.1012 | gru=2, seeds=2, mixch=0.3 | 0.124 | 0.131 | 0.096 |
| m_lstm_mvc1 | 0.1012 | gru=2, seeds=2, mvc=1 | 0.131 | 0.14 | 0.1 |
| base1000 | 0.1015 | gain_aug=0.2 | 0.14 | 0.138 | 0.104 |
| crop800 | 0.1016 | bs=16, crop=800, gain_aug=0.2 | 0.136 | 0.136 | 0.104 |
| r9_tf_raw | 0.1017 | gru=3, feat=envraw, seeds=2 | 0.12 | 0.137 | 0.106 |
| g05_d6h96 | 0.1019 | hidden=96, depth=6 | 0.126 | 0.136 | 0.106 |
| p_lstm_pca3 | 0.1019 | gru=2, seeds=2, pca_k=3 | 0.121 | 0.142 | 0.11 |
| drop04 | 0.1020 | drop=0.4, wd=0.05, gain_aug=0.2 | 0.142 | 0.138 | 0.104 |
| h128_1000 | 0.1022 | hidden=128, gain_aug=0.2 | 0.139 | 0.14 | 0.107 |
| r12_raw_chd5 | 0.1022 | feat=envraw, seeds=2, chdrop=0.5 | 0.124 | 0.124 | 0.106 |
| base3000 | 0.1024 | steps=3000, gain_aug=0.2 | 0.139 | 0.139 | 0.105 |
| r16_tan_lstm | 0.1028 | gru=2, feat=envtan, seeds=2 | 0.11 | 0.145 | 0.109 |
| both1000 | 0.1030 | gain_aug=0.2, inmode=both | 0.126 | 0.146 | 0.11 |
| g08 | 0.1032 | gain_aug=0.8 | 0.147 | 0.133 | 0.1 |
| r8_tf | 0.1035 | gru=3, seeds=2 | 0.14 | 0.134 | 0.106 |
| p_lstm_pca2 | 0.1035 | gru=2, seeds=2, pca_k=2 | 0.121 | 0.142 | 0.11 |
| r6_envraw_h128 | 0.1036 | hidden=128, feat=envraw, seeds=2 | 0.121 | 0.143 | 0.105 |
| nogru1000 | 0.1038 | depth=7, gru=0, gain_aug=0.2 | 0.125 | 0.141 | 0.11 |
| r6_envraw | 0.1043 | feat=envraw, seeds=2 | 0.12 | 0.143 | 0.107 |
| r5_nogru | 0.1045 | depth=7, gru=0, seeds=2 | 0.119 | 0.135 | 0.11 |
| smoke | 0.1048 | steps=300, gain_aug=0.2 | 0.121 | 0.144 | 0.11 |
| r6_envraw_s2000 | 0.1050 | steps=2000, feat=envraw, seeds=2 | 0.125 | 0.148 | 0.105 |
| r7_both | 0.1052 | inmode=both, seeds=2 | 0.145 | 0.14 | 0.107 |
| al1 | 0.1054 | seeds=2, align=1 | 0.177 | 0.129 | 0.098 |
| gain0 | 0.1059 | gain_aug=0.0 | 0.126 | 0.152 | 0.108 |
| r16_tan_gru | 0.1063 | feat=envtan, seeds=2 | 0.115 | 0.146 | 0.115 |
| r7_center | 0.1072 | inmode=center, seeds=2 | 0.181 | 0.132 | 0.1 |
| al1_g02 | 0.1074 | gain_aug=0.2, seeds=2, align=1 | 0.176 | 0.135 | 0.1 |
| al1_both | 0.1076 | inmode=both, seeds=2, align=1 | 0.157 | 0.14 | 0.107 |
| center1000 | 0.1080 | gain_aug=0.2, inmode=center | 0.171 | 0.141 | 0.103 |
| r14_lstm_mix2 | 0.1080 | gru=2, seeds=2, mixup=2.0 | 0.131 | 0.15 | 0.112 |
| r5_envcorr_chd | 0.1087 | feat=envcorr, seeds=2, chdrop=0.15 | 0.141 | 0.146 | 0.111 |
| r7_envcorr_pseudo | 0.1093 | feat=envcorr, seeds=2, pseudo=1, pseudo_w=0.5 | 0.144 | 0.141 | 0.108 |
| al2 | 0.1098 | seeds=2, align=2 | 0.186 | 0.141 | 0.106 |
| r7_envcorr_k7 | 0.1099 | depth=6, k=7, feat=envcorr, seeds=2 | 0.146 | 0.142 | 0.111 |
| p_lstm_pca1 | 0.1100 | gru=2, seeds=2, pca_k=1 | 0.13 | 0.139 | 0.121 |
| r12_corr_chd3 | 0.1101 | feat=envcorr, seeds=2, chdrop=0.3 | 0.144 | 0.147 | 0.113 |
| r8_lstm_corr | 0.1102 | gru=2, feat=envcorr, seeds=2 | 0.143 | 0.146 | 0.11 |
| r5_envcorr | 0.1103 | feat=envcorr, seeds=2 | 0.144 | 0.144 | 0.111 |
| envframe1000 | 0.1109 | gain_aug=0.2, feat=envframe | 0.139 | 0.149 | 0.108 |
| frame1000 | 0.1113 | gain_aug=0.2, feat=frame | 0.131 | 0.15 | 0.113 |
| r8_tf_corr | 0.1123 | gru=3, feat=envcorr, seeds=2 | 0.146 | 0.146 | 0.116 |
| r18_corr_ssl1500 | 0.1138 | feat=envcorr, seeds=2, ssl_steps=1500 | 0.133 | 0.156 | 0.117 |
| r7_envframe | 0.1150 | feat=envframe, seeds=2 | 0.166 | 0.153 | 0.108 |
| n1_cfg_d02_w03 | 0.1160 | gru=2, seeds=2, cfg_drop=0.2, cfg_w=0.3 | 0.133 | 0.148 | 0.101 |
| r7_frame | 0.1184 | feat=frame, seeds=2 | 0.169 | 0.159 | 0.113 |
| n1_flip05 | 0.1208 | gru=2, seeds=2, flip=0.5 | 0.115 | 0.184 | 0.118 |
| gbm_base | 0.1228 | span=40, rounds=300, lr=0.05, leaves=31 | 0.116 | 0.19 | 0.136 |
| aw_095 | 0.1252 | gru=2, seeds=2, active_w=0.95 | 0.116 | 0.168 | 0.101 |
| m_lstm_mvc2 | 0.1269 | gru=2, seeds=2, mvc=2 | 0.173 | 0.14 | 0.114 |
| m_gru_mvc2 | 0.1359 | seeds=2, mvc=2 | 0.184 | 0.169 | 0.131 |
| nocond1000 | 0.1403 | cond=0, gain_aug=0.2 | 0.178 | 0.162 | 0.123 |
| nocond3000 | 0.1409 | cond=0, steps=3000, crop=1000, gain_aug=0.2 | 0.17 | 0.155 | 0.113 |
| r4_ema999 | 0.1676 | seeds=2, ema=0.999 | 0.12 | 0.199 | 0.199 |
