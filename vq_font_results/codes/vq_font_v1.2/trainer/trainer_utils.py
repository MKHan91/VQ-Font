

import torch
from collections import OrderedDict


def load_checkpoint(path, gen, disc, g_optim, d_optim, g_scheduler, d_scheduler):
    """
    load_checkpoint
    """
    ckpt = torch.load(path,map_location={'cuda:1': 'cuda:0'})
    gen.load_state_dict(ckpt['generator'])
    g_optim.load_state_dict(ckpt['optimizer_states'])
    g_scheduler.load_state_dict(ckpt['g_scheduler'])

    if disc is not None:
        disc.load_state_dict(ckpt['discriminator'])
        d_optim.load_state_dict(ckpt['d_optimizer'])
        d_scheduler.load_state_dict(ckpt['d_scheduler'])

    st_epoch = ckpt['epoch'] + 1
    loss = ckpt['loss']

    return st_epoch, loss


def load_checkpoint_torch(ckpt_path, gen, disc, load_codebook_only=False):
    """
    PyTorch ckpt 로드용
    - gen: vq-font generator
    - disc: discriminator (optional)
    - load_codebook_only: True면 generator의 codebook만 로드
    """
    ckpt = torch.load(ckpt_path, map_location='cpu')
    state_dict = ckpt.get("state_dict", ckpt)  # Lightning ckpt와 일반 ckpt 모두 지원

    if load_codebook_only:
        # codebook key만 골라서 로드
        codebook_keys = [k for k in state_dict.keys() if "quantize.embedding" in k]

        for k in codebook_keys:
            v = state_dict[k]
            # gen.codebook.embedding으로 매핑
            gen.codebook.embedding.data.copy_(v)

        print("✅ Codebook loaded into generator (encoder/decoder untouched).")

    else:
        # 전체 generator state_dict 로드
        gen.load_state_dict({k.replace("generator.", ""): v for k, v in state_dict.items() if k.startswith("generator.")}, strict=False)
        print("✅ Full generator weights loaded.")

    # discriminator 로드 (있으면)
    if disc is not None:
        disc_keys = [k for k in state_dict.keys() if k.startswith("discriminator.")]
        dt = OrderedDict()
        for k in disc_keys:
            dt[k.replace("discriminator.", "")] = state_dict[k]
        disc.load_state_dict(dt, strict=False)
        print("✅ Discriminator weights loaded.")

    # PyTorch에서는 optimizer / scheduler는 domain mismatch 가능성 있으므로 **로드하지 않음**
    st_epoch = ckpt.get('epoch', 0) + 1
    loss = ckpt.get('loss', None)

    return st_epoch, loss