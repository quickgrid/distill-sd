import gc
from pathlib import Path
from typing import Tuple, List

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageDraw
from einops import rearrange
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

input_path = r"PATH_TO_IMAGES_FOLDER"
cfg_path = r"PATH_TO\sd-v1\ldm-unet-inference.yaml"
original_cfg_path = r"PATH_TO\sd-v1\custom-v1-inference.yaml"
# ckpt_path = r"PATH_TO\sd_1.4.ckpt"
ckpt_path = r"PATH_TO\v1-5-pruned-emaonly.ckpt"


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
        # print(pl_sd.keys())
        sd = pl_sd["state_dict"]
        del pl_sd
        gc.collect()
        sd_keys = sd.keys()

        # print(sd_keys)

        def delete_model_layers(layer_initial_list: List[str]):
            for layer_initial in layer_initial_list:
                key_delete_list = []
                for dkey in sd_keys:
                    if dkey.startswith(layer_initial):
                        key_delete_list.append(dkey)

                for k in key_delete_list:
                    del sd[f'{k}']

        delete_model_layers(["first_stage_model.encoder"])
        # print(sd.keys())

        self.ldm_model = instantiate_from_config(config.model)
        self.ldm_model.load_state_dict(sd, strict=False)
        self.ldm_model.half()
        self.ldm_model = self.ldm_model.to(device)
        self.ldm_model.eval()
        for param in self.ldm_model.parameters():
            param.requires_grad = False

        self.sampler = DDIMSampler(self.ldm_model)

        if use_xformers:
            import distillsd.vae.model
            distillsd.vae.model.AttnBlock.forward = patch_xformers_attn_forward

        original_config = OmegaConf.load(original_cfg_path)

        remove_prefix = 'cond_stage_model.'
        new_state_dict = {k[len(remove_prefix):] if k.startswith(remove_prefix) else k: v for k, v in sd.items()}

        text_embed_model = instantiate_from_config(original_config.model.params.cond_stage_config)
        text_embed_model.load_state_dict(new_state_dict, strict=False)

        if process_text_encoder_in_cpu:
            text_embed_model.device = "cpu"
        else:
            text_embed_model.half()
            text_embed_model = text_embed_model.to(self.device)

        text_embed_model = text_embed_model.eval()
        text_embed_model.train = False
        for param in text_embed_model.parameters():
            param.requires_grad = False

        # Negative prompt working or not?
        self.input_prompts = [
            "a car in shape of carrot flying through solar system, high quality, 4k"
        ]
        negative_prompts = [
            "low quality, noisy, blurry, cropped, logo, text"
        ]

        # Text encoding shape of (batch, max_token, embed_dim), for SD, max_token=77, embed_dim=768.
        self.text_z = text_embed_model.encode(self.input_prompts)
        self.text_z_uncond = text_embed_model.encode(negative_prompts)
        if process_text_encoder_in_cpu:
            self.text_z = self.text_z.to(self.device)
            self.text_z_uncond = self.text_z_uncond.to(self.device)

        del text_embed_model
        gc.collect()
        torch.cuda.empty_cache()

        remove_prefix = 'first_stage_model.'
        new_state_dict = {k[len(remove_prefix):] if k.startswith(remove_prefix) else k: v for k, v in sd.items()}

        self.vae_stage = instantiate_from_config(original_config.model.params.first_stage_config)
        self.vae_stage.load_state_dict(new_state_dict, strict=False)

        if process_vae_in_cpu:
            self.vae_stage = self.vae_stage.cpu()
        else:
            self.vae_stage.half()
            self.vae_stage = self.vae_stage.to(self.device)

        self.vae_stage.eval()
        self.vae_stage.train = False
        for param in self.vae_stage.parameters():
            param.requires_grad = False

        del sd
        gc.collect()
        torch.cuda.empty_cache()

        self.precision_type = torch.float16
        self.sampling_steps = 16
        self.unconditional_guidance_scale = 7.0

        # vae downsample factor based on configuration. kl-f8 is downsample by 8x.
        downsample_factor = 8
        img_h = 512
        img_w = 512
        # output channel of first stage can be vqgan or autoencoder model.
        vae_output_channels = 4
        self.sd_output_shape = [
            vae_output_channels, img_h // downsample_factor, img_w // downsample_factor
        ]

    def process(self) -> None:
        pass

    def run_text_to_image(self) -> None:
        with tqdm(self.img_data_loader) as pbar:
            for batch_idx, (input_data, filename, full_filename) in enumerate(pbar):
                # input_data = input_data.to(self.device)

                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    with torch.no_grad():
                        with self.ldm_model.ema_scope():
                            samples_ddim, _ = self.sampler.sample(
                                S=self.sampling_steps,
                                conditioning=self.text_z,
                                batch_size=self.batch_size,
                                shape=self.sd_output_shape,
                                verbose=False,
                                unconditional_guidance_scale=self.unconditional_guidance_scale,
                                unconditional_conditioning=self.text_z_uncond,
                                # unconditional_guidance_scale=1.0,
                                # unconditional_conditioning=None,
                                eta=0.0,
                                temperature=1.0,
                                noise_dropout=0.0,
                            )

                            # Decode first stage function normalizes by this formula, (z = 1. / 0.18215 * z).
                            samples_ddim *= 1. / 0.18215

                if self.process_vae_in_cpu:
                    with torch.no_grad():
                        samples_ddim = samples_ddim.cpu()
                        xrec = self.vae_stage.decode(samples_ddim)
                        xrec = xrec.to(self.device)
                else:
                    with torch.autocast(device_type='cuda', dtype=torch.float16):
                        with torch.no_grad():
                            with self.ldm_model.ema_scope():
                                xrec = self.vae_stage.decode(samples_ddim)

                print(xrec.shape)
                x0 = custom_to_pil(xrec[0])
                ImageDraw.Draw(x0).text((0, 0), f'{self.input_prompts}', (255, 255, 255))
                plt.imshow(x0)
                plt.show()

                break


if __name__ == '__main__':
    sd_inference = StableDiffusionInference(
        input_path=input_path,
        cfg_path=cfg_path,
        ckpt_path=ckpt_path,
    )
    sd_inference.run_text_to_image()
