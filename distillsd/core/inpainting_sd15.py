"""Inpainting with SD 1.5 model. Will not work for regular models.
"""

import gc
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from einops import rearrange, repeat
from omegaconf import OmegaConf
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms
from tqdm import tqdm
from transformers.utils import logging as tlog

from distillsd.core.config import CompressionConfig
from distillsd.ldm.ddim import DDIMSampler
from distillsd.vae.util import instantiate_from_config

try:
    import xformers.ops

    CompressionConfig.available_xformers = True
except ModuleNotFoundError as err:
    print(err)

# tlog.set_verbosity_info()
tlog.set_verbosity_error()

torch.set_grad_enabled(False)


input_path = r"PATH_TO_IMAGE_FOLDER"
cfg_path = r"PATH\custom-v1-inpainting-inference.yaml"
ckpt_path = r"PATH\sd-v1-5-inpainting.ckpt"

prompt = "moon planet in background, high quality, 4k"
# prompt = "an astronaut floating in space, high quality, 4k"
negative_prompt = "noisy, blurry, low quality, text, logo"
image = r"PATH/img.png"
# mask = r"PATH/img_mask.png"
mask = r"PATH/img_mask_inv.png"


def get_xformers_flash_attention_op(q, k, v):
    try:
        flash_attention_op = xformers.ops.MemoryEfficientAttentionFlashAttentionOp
        # flash_attention_op = xformers.ops.MemoryEfficientAttentionOp
        fw, bw = flash_attention_op
        if fw.supports(xformers.ops.fmha.Inputs(query=q, key=k, value=v, attn_bias=None)):
            return flash_attention_op
    except Exception as e:
        print(e, "enabling flash attention")

    return None


def patch_xformers_attn_forward(self, x):
    """Can replace LDM vqgan attention method forward with xformers attention. Should also work when replacing other
    attention code. Code copied from,
    https://github.com/AUTOMATIC1111/stable-diffusion-webui/blob/master/modules/sd_hijack_optimizations.py.
    """
    h_ = x
    h_ = self.norm(h_)
    q = self.q(h_)
    k = self.k(h_)
    v = self.v(h_)
    b, c, h, w = q.shape
    q, k, v = map(lambda t: rearrange(t, 'b c h w -> b (h w) c'), (q, k, v))
    # dtype = q.dtype
    # if True:
    #     q, k = q.float(), k.float()
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    # out = xformers.ops.memory_efficient_attention(q, k, v, op=get_xformers_flash_attention_op(q, k, v))
    out = xformers.ops.memory_efficient_attention(q, k, v)
    # out = out.to(dtype)
    out = rearrange(out, 'b (h w) c -> b c h w', h=h)
    out = self.proj_out(out)
    return x + out


def preprocess_vqgan(x):
    return 2. * x - 1.


class CustomImageDataset(Dataset):
    def __init__(
            self,
            input_path: str,
            image_size: int,
            keep_aspect_ratio: bool,
    ) -> None:
        super().__init__()

        self.input_path = input_path
        self.image_size = image_size
        self.files_list = [
            p for ext in CompressionConfig.dataset_image_exts for p in Path(f'{input_path}').glob(f'*.{ext}')
        ]

        resize_type = transforms.Resize(image_size) if keep_aspect_ratio else transforms.Resize(
            (image_size, image_size))

        self.transform = transforms.Compose([
            resize_type,
            transforms.ToTensor(),
            transforms.Lambda(preprocess_vqgan),
        ])

    def __len__(self) -> int:
        return len(self.files_list)

    def __getitem__(self, index) -> Tuple[torch.Tensor, str, str]:
        path = self.files_list[index]
        img = Image.open(path).convert('RGB')
        transformed_img = self.transform(img)
        return transformed_img, path.stem, path.name


def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)


def custom_to_pil(x):
    x = torch.clamp(x, -1., 1.)
    x = (x + 1.) / 2.
    x = x.permute(1, 2, 0)
    x = x.detach().cpu().numpy()
    x = (255 * x).astype(np.uint8)
    x = Image.fromarray(x)
    if not x.mode == "RGB":
        x = x.convert("RGB")
    return x


def make_image_mask_batch(image, mask, device):
    image = np.array(Image.open(image).convert("RGB"))
    image = image.astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)

    mask = np.array(Image.open(mask).convert("L"))
    mask = mask.astype(np.float32) / 255.0
    mask = mask[None, None]
    mask[mask < 0.5] = 0
    mask[mask >= 0.5] = 1
    mask = torch.from_numpy(mask)

    masked_image = (1 - mask) * image

    batch = {"image": image, "mask": mask, "masked_image": masked_image}
    for k in batch:
        batch[k] = batch[k].to(device=device)
        batch[k] = batch[k] * 2.0 - 1.0
    return batch


def make_batch_sd(
        image,
        mask,
        txt,
        device,
        num_samples=1,
        invert_mask=True,
):
    image = np.array(Image.open(image).convert("RGB"))
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image).to(dtype=torch.float32) / 127.5 - 1.0

    mask = np.array(Image.open(mask).convert("L"))
    mask = mask.astype(np.float32) / 255.0
    mask = mask[None, None]

    mask[mask < 0.5] = 0
    mask[mask >= 0.5] = 1

    mask = torch.from_numpy(mask)

    masked_image = image * (mask < 0.5)

    batch = {
        "image": repeat(image.to(device=device), "1 ... -> n ...", n=num_samples),
        "txt": num_samples * [txt],
        "mask": repeat(mask.to(device=device), "1 ... -> n ...", n=num_samples),
        "masked_image": repeat(masked_image.to(device=device), "1 ... -> n ...", n=num_samples),
    }
    return batch


class StableDiffusionInference:
    def __init__(
            self,
            input_path: str,
            # output_path: str,
            cfg_path: str,
            ckpt_path: str,
            image_size: int = 256,
            batch_size: int = 1,
            num_workers: int = 0,
            device: str = 'cuda',
            keep_aspect_ratio: bool = False,
            use_xformers: bool = False,
            process_text_encoder_in_cpu: bool = False,
            process_vae_in_cpu: bool = False,
    ):
        self.process_vae_in_cpu = process_vae_in_cpu
        self.batch_size = batch_size
        self.device = device
        img_dataset = CustomImageDataset(
            input_path=input_path,
            image_size=image_size,
            keep_aspect_ratio=keep_aspect_ratio,
        )
        self.img_data_loader = DataLoader(
            img_dataset,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=num_workers,
            drop_last=False,
            collate_fn=collate_fn,
        )

        config = OmegaConf.load(cfg_path)
        pl_sd = torch.load(ckpt_path, map_location="cpu")
        print(pl_sd.keys())
        sd = pl_sd["state_dict"]
        del pl_sd
        gc.collect()
        sd_keys = sd.keys()
        print(sd_keys)
        self.ldm_model = instantiate_from_config(config.model)
        self.ldm_model.load_state_dict(sd, strict=False)
        self.ldm_model.half()
        self.ldm_model = self.ldm_model.to(device)
        self.ldm_model.eval()
        for param in self.ldm_model.parameters():
            param.requires_grad = False

        self.sampler = DDIMSampler(self.ldm_model)

    def process(self) -> None:
        pass

    def run_conditional_inpainting(self) -> None:
        with tqdm(self.img_data_loader) as pbar:
            for batch_idx, (input_data, filename, full_filename) in enumerate(pbar):
                # input_data = input_data.to(self.device)

                ddim_steps = 16
                num_samples = 1
                h = 256
                w = 256
                scale = 9.0
                eta = 0.0
                vae_out_channels = 4
                scale_factor = 8    # kl-f8

                seed = 0
                # prng = np.random.RandomState(seed)
                # start_code = prng.randn(num_samples, vae_out_channels, h // scale_factor, w // scale_factor)
                # start_code = torch.from_numpy(start_code).to(device=self.device, dtype=torch.float32)

                with (
                    torch.no_grad(),
                    torch.autocast(device_type='cuda', dtype=torch.float16)
                ):
                    batch = make_batch_sd(image, mask, txt=prompt, device=self.device, num_samples=num_samples)

                    c = self.ldm_model.cond_stage_model.encode(batch["txt"])

                    c_cat = list()
                    for ck in self.ldm_model.concat_keys:
                        # cc = batch[ck].float()
                        cc = batch[ck]
                        if ck != self.ldm_model.masked_image_key:
                            bchw = [num_samples, vae_out_channels, h // scale_factor, w // scale_factor]
                            cc = torch.nn.functional.interpolate(cc, size=bchw[-2:])
                        else:
                            cc = self.ldm_model.get_first_stage_encoding(self.ldm_model.encode_first_stage(cc))
                        c_cat.append(cc)
                    c_cat = torch.cat(c_cat, dim=1)

                    # cond
                    cond = {"c_concat": [c_cat], "c_crossattn": [c]}

                    # uncond cond
                    # uc_cross = self.ldm_model.get_unconditional_conditioning(num_samples, "")
                    uc_cross = self.ldm_model.get_unconditional_conditioning(num_samples, negative_prompt)
                    uc_full = {"c_concat": [c_cat], "c_crossattn": [uc_cross]}

                    shape = [self.ldm_model.channels, h // scale_factor, w // scale_factor]
                    samples_cfg, intermediates = self.sampler.sample(
                        ddim_steps,
                        num_samples,
                        shape,
                        cond,
                        verbose=False,
                        eta=eta,
                        unconditional_guidance_scale=scale,
                        unconditional_conditioning=uc_full,
                        # x_T=start_code,
                    )
                    x_samples_ddim = self.ldm_model.decode_first_stage(samples_cfg)

                    result = custom_to_pil(x_samples_ddim[0])
                    plt.imshow(result)
                    plt.show()


if __name__ == '__main__':
    sd_inference = StableDiffusionInference(
        input_path=input_path,
        cfg_path=cfg_path,
        ckpt_path=ckpt_path,
    )
    sd_inference.run_conditional_inpainting()
