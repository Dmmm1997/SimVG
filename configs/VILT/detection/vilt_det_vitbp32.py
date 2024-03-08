model = dict(
    type="MIX",
    vis_enc=dict(
        type="ViLTransformerSS",
        vocab_size=30522,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=768 * 4,
        max_position_embeddings=40,
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        pretrain="/home/dmmm/demo_mirror/pretrain/ViLT/pretrain_weights/vilt_200k_mlm_itm.ckpt",
    ),
    lan_enc=None,
    fusion=None,
    head=None
)


optimizer_config = dict(type="Adam", lr=0.0005, betas=(0.9, 0.98), eps=1e-9, weight_decay=0, amsgrad=True)
# optimizer_config = dict(type="AdamW", lr=0.0002)
grad_norm_clip = 0.15
scheduler_config = dict(type="MultiStepLRWarmUp", warmup_epochs=5, decay_steps=[25], decay_ratio=0.1, max_epoch=30)
