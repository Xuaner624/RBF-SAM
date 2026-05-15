encoder_embed_dim = 768
encoder_depth = 12
encoder_num_heads = 12
encoder_global_attn_indexes = [2, 5, 8, 11]

prompt_embed_dim = 256
image_size = 1024
vit_patch_size = 16
image_embedding_size = image_size // vit_patch_size

model = dict(
    type='rbf_sam',
    image_encoder=dict(
        type='ImageEncoderViT_Robust',
        depth=encoder_depth,
        embed_dim=encoder_embed_dim,
        img_size=image_size,
        mlp_ratio=4,
        num_heads=encoder_num_heads,
        patch_size=vit_patch_size,
        qkv_bias=True,
        use_rel_pos=True,
        global_attn_indexes=encoder_global_attn_indexes,
        window_size=14,
        out_chans=prompt_embed_dim,
    ),
    wavelet_block=dict(
            type='Wavelet_Block',
            in_chans=encoder_embed_dim,
            embed_dim=prompt_embed_dim,
            num_heads=8,
            token_dim=prompt_embed_dim,
            dropout=0.1,
    ),
    resample_block=dict(
            type='Feature_Resample_Block',
            embedding_dim=prompt_embed_dim,
            num_groups=8,
    ),
    prompt_encoder=dict(
        type='PromptEncoder_Robust',
        embed_dim=prompt_embed_dim,
        image_embedding_size=(image_embedding_size, image_embedding_size),
        input_image_size=(image_size, image_size),
        mask_in_chans=16,
    ),
    mask_decoder=dict(
        type='MaskDecoder_Robust',
        transformer_dim=prompt_embed_dim,
        transformer=dict(
            type='Deformable_TwoWayTransformer',
            embedding_dim=prompt_embed_dim,
            depth=2,
            mlp_dim=2048,
            num_heads=8,
        ),
    )
)


train_cfg = dict(
    max_num=150,
)

test_cfg = dict(
)
