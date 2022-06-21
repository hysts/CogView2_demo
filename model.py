#This code is adapted from https://github.com/THUDM/CogView2/blob/4e55cce981eb94b9c8c1f19ba9f632fd3ee42ba8/cogview2_text2image.py

from __future__ import annotations

import argparse
import functools
import logging
import pathlib
import sys
import time
from typing import Any

import gradio as gr
import numpy as np
import torch
from icetk import IceTokenizer
from SwissArmyTransformer import get_args
from SwissArmyTransformer.arguments import set_random_seed
from SwissArmyTransformer.generation.autoregressive_sampling import \
    filling_sequence
from SwissArmyTransformer.model import CachedAutoregressiveModel

app_dir = pathlib.Path(__file__).parent
submodule_dir = app_dir / 'CogView2'
sys.path.insert(0, submodule_dir.as_posix())

from coglm_strategy import CoglmStrategy
from sr_pipeline import SRGroup

formatter = logging.Formatter(
    '[%(asctime)s] %(name)s %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S')
stream_handler = logging.StreamHandler(stream=sys.stdout)
stream_handler.setLevel(logging.DEBUG)
stream_handler.setFormatter(formatter)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.propagate = False
logger.addHandler(stream_handler)

ICETK_MODEL_DIR = app_dir / 'icetk_models'


def get_masks_and_position_ids_coglm(
        seq: torch.Tensor, context_length: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    tokens = seq.unsqueeze(0)

    attention_mask = torch.ones((1, len(seq), len(seq)), device=tokens.device)
    attention_mask.tril_()
    attention_mask[..., :context_length] = 1
    attention_mask.unsqueeze_(1)

    position_ids = torch.zeros(len(seq),
                               device=tokens.device,
                               dtype=torch.long)
    torch.arange(0, context_length, out=position_ids[:context_length])
    torch.arange(512,
                 512 + len(seq) - context_length,
                 out=position_ids[context_length:])

    position_ids = position_ids.unsqueeze(0)
    return tokens, attention_mask, position_ids


class InferenceModel(CachedAutoregressiveModel):
    def final_forward(self, logits, **kwargs):
        logits_parallel = logits
        logits_parallel = torch.nn.functional.linear(
            logits_parallel.float(),
            self.transformer.word_embeddings.weight[:20000].float())
        return logits_parallel


def get_recipe(name: str) -> dict[str, Any]:
    r = {
        'attn_plus': 1.4,
        'temp_all_gen': 1.15,
        'topk_gen': 16,
        'temp_cluster_gen': 1.,
        'temp_all_dsr': 1.5,
        'topk_dsr': 100,
        'temp_cluster_dsr': 0.89,
        'temp_all_itersr': 1.3,
        'topk_itersr': 16,
        'query_template': '{}<start_of_image>',
    }
    if name == 'none':
        pass
    elif name == 'mainbody':
        r['query_template'] = '{} 高清摄影 隔绝<start_of_image>'

    elif name == 'photo':
        r['query_template'] = '{} 高清摄影<start_of_image>'

    elif name == 'flat':
        r['query_template'] = '{} 平面风格<start_of_image>'
        # r['attn_plus'] = 1.8
        # r['temp_cluster_gen'] = 0.75
        r['temp_all_gen'] = 1.1
        r['topk_dsr'] = 5
        r['temp_cluster_dsr'] = 0.4

        r['temp_all_itersr'] = 1
        r['topk_itersr'] = 5
    elif name == 'comics':
        r['query_template'] = '{} 漫画 隔绝<start_of_image>'
        r['topk_dsr'] = 5
        r['temp_cluster_dsr'] = 0.4
        r['temp_all_gen'] = 1.1
        r['temp_all_itersr'] = 1
        r['topk_itersr'] = 5
    elif name == 'oil':
        r['query_template'] = '{} 油画风格<start_of_image>'
        pass
    elif name == 'sketch':
        r['query_template'] = '{} 素描风格<start_of_image>'
        r['temp_all_gen'] = 1.1
    elif name == 'isometric':
        r['query_template'] = '{} 等距矢量图<start_of_image>'
        r['temp_all_gen'] = 1.1
    elif name == 'chinese':
        r['query_template'] = '{} 水墨国画<start_of_image>'
        r['temp_all_gen'] = 1.12
    elif name == 'watercolor':
        r['query_template'] = '{} 水彩画风格<start_of_image>'
    return r


def get_default_args() -> argparse.Namespace:
    arg_list = ['--mode', 'inference', '--fp16']
    args = get_args(arg_list)
    known = argparse.Namespace(img_size=160,
                               only_first_stage=False,
                               inverse_prompt=False,
                               style='mainbody')
    args = argparse.Namespace(**vars(args), **vars(known),
                              **get_recipe(known.style))
    return args


class Model:
    def __init__(self, only_first_stage: bool = False):
        self.args = get_default_args()
        self.args.only_first_stage = only_first_stage

        self.tokenizer = self.load_tokenizer()

        self.model, self.args = self.load_model()
        self.strategy = self.load_strategy()
        self.srg = self.load_srg()

        self.query_template = self.args.query_template
        self.style = self.args.style
        self.device = torch.device(self.args.device)
        self.fp16 = self.args.fp16
        self.max_batch_size = self.args.max_inference_batch_size
        self.only_first_stage = self.args.only_first_stage

    def load_tokenizer(self) -> IceTokenizer:
        logger.info('--- load_tokenizer ---')
        start = time.perf_counter()

        tokenizer = IceTokenizer(ICETK_MODEL_DIR.as_posix())
        tokenizer.add_special_tokens(
            ['<start_of_image>', '<start_of_english>', '<start_of_chinese>'])

        elapsed = time.perf_counter() - start
        logger.info(f'Elapsed: {elapsed}')
        logger.info('--- done ---')
        return tokenizer

    def load_model(self) -> tuple[InferenceModel, argparse.Namespace]:
        logger.info('--- load_model ---')
        start = time.perf_counter()

        model, args = InferenceModel.from_pretrained(self.args, 'coglm')

        elapsed = time.perf_counter() - start
        logger.info(f'Elapsed: {elapsed}')
        logger.info('--- done ---')
        return model, args

    def load_strategy(self) -> CoglmStrategy:
        logger.info('--- load_strategy ---')
        start = time.perf_counter()

        invalid_slices = [slice(self.tokenizer.num_image_tokens, None)]
        strategy = CoglmStrategy(invalid_slices,
                                 temperature=self.args.temp_all_gen,
                                 top_k=self.args.topk_gen,
                                 top_k_cluster=self.args.temp_cluster_gen)

        elapsed = time.perf_counter() - start
        logger.info(f'Elapsed: {elapsed}')
        logger.info('--- done ---')
        return strategy

    def load_srg(self) -> SRGroup:
        logger.info('--- load_srg ---')
        start = time.perf_counter()

        srg = None if self.args.only_first_stage else SRGroup(self.args)

        elapsed = time.perf_counter() - start
        logger.info(f'Elapsed: {elapsed}')
        logger.info('--- done ---')
        return srg

    def update_style(self, style: str) -> None:
        if style == self.style:
            return
        logger.info('--- update_style ---')
        start = time.perf_counter()

        self.args = argparse.Namespace(**(vars(self.args) | get_recipe(style)))
        self.query_template = self.args.query_template
        logger.info(f'{self.query_template=}')

        self.strategy.temperature = self.args.temp_all_gen

        if self.srg is not None:
            self.srg.dsr.strategy.temperature = self.args.temp_all_dsr
            self.srg.dsr.strategy.topk = self.args.topk_dsr
            self.srg.dsr.strategy.temperature2 = self.args.temp_cluster_dsr

            self.srg.itersr.strategy.temperature = self.args.temp_all_itersr
            self.srg.itersr.strategy.topk = self.args.topk_itersr

        elapsed = time.perf_counter() - start
        logger.info(f'Elapsed: {elapsed}')
        logger.info('--- done ---')

    def run(self, text: str, style: str, seed: int, only_first_stage: bool,
            num: int) -> list[np.ndarray] | None:
        set_random_seed(seed)
        seq, txt_len = self.preprocess_text(text)
        if seq is None:
            return None
        self.update_style(style)
        self.only_first_stage = only_first_stage
        tokens = self.generate_tokens(seq, txt_len, num)
        res = self.generate_images(seq, txt_len, tokens)
        return res

    @torch.inference_mode()
    def preprocess_text(
            self, text: str) -> tuple[torch.Tensor, int] | tuple[None, None]:
        logger.info('--- preprocess_text ---')
        start = time.perf_counter()

        text = self.query_template.format(text)
        logger.info(f'{text=}')
        seq = self.tokenizer.encode(text)
        logger.info(f'{len(seq)=}')
        if len(seq) > 110:
            logger.info('The input text is too long.')
            return None, None
        txt_len = len(seq) - 1
        seq = torch.tensor(seq + [-1] * 400, device=self.device)

        elapsed = time.perf_counter() - start
        logger.info(f'Elapsed: {elapsed}')
        logger.info('--- done ---')
        return seq, txt_len

    @torch.inference_mode()
    def generate_tokens(self,
                        seq: torch.Tensor,
                        txt_len: int,
                        num: int = 8) -> torch.Tensor:
        logger.info('--- generate_tokens ---')
        start = time.perf_counter()

        # calibrate text length
        log_attention_weights = torch.zeros(
            len(seq),
            len(seq),
            device=self.device,
            dtype=torch.half if self.fp16 else torch.float32)
        log_attention_weights[:, :txt_len] = self.args.attn_plus
        get_func = functools.partial(get_masks_and_position_ids_coglm,
                                     context_length=txt_len)

        output_list = []
        remaining = num
        for _ in range((num + self.max_batch_size - 1) // self.max_batch_size):
            self.strategy.start_pos = txt_len + 1
            coarse_samples = filling_sequence(
                self.model,
                seq.clone(),
                batch_size=min(remaining, self.max_batch_size),
                strategy=self.strategy,
                log_attention_weights=log_attention_weights,
                get_masks_and_position_ids=get_func)[0]
            output_list.append(coarse_samples)
            remaining -= self.max_batch_size
        output_tokens = torch.cat(output_list, dim=0)
        logger.info(f'{output_tokens.shape=}')

        elapsed = time.perf_counter() - start
        logger.info(f'Elapsed: {elapsed}')
        logger.info('--- done ---')
        return output_tokens

    @staticmethod
    def postprocess(tensor: torch.Tensor) -> np.ndarray:
        return tensor.cpu().mul(255).add_(0.5).clamp_(0, 255).permute(
            1, 2, 0).to(torch.uint8).numpy()

    @torch.inference_mode()
    def generate_images(self, seq: torch.Tensor, txt_len: int,
                        tokens: torch.Tensor) -> list[np.ndarray]:
        logger.info('--- generate_images ---')
        start = time.perf_counter()

        logger.info(f'{self.only_first_stage=}')
        res = []
        if self.only_first_stage:
            for i in range(len(tokens)):
                seq = tokens[i]
                decoded_img = self.tokenizer.decode(image_ids=seq[-400:])
                decoded_img = torch.nn.functional.interpolate(decoded_img,
                                                              size=(480, 480))
                decoded_img = self.postprocess(decoded_img[0])
                res.append(decoded_img)  # only the last image (target)
        else:  # sr
            iter_tokens = self.srg.sr_base(tokens[:, -400:], seq[:txt_len])
            for seq in iter_tokens:
                decoded_img = self.tokenizer.decode(image_ids=seq[-3600:])
                decoded_img = torch.nn.functional.interpolate(decoded_img,
                                                              size=(480, 480))
                decoded_img = self.postprocess(decoded_img[0])
                res.append(decoded_img)  # only the last image (target)

        elapsed = time.perf_counter() - start
        logger.info(f'Elapsed: {elapsed}')
        logger.info('--- done ---')
        return res


class AppModel(Model):
    def __init__(self, only_first_stage: bool):
        super().__init__(only_first_stage)
        self.translator = gr.Interface.load(
            'spaces/chinhon/translation_eng2ch')

    def make_grid(self, images: list[np.ndarray] | None) -> np.ndarray | None:
        if images is None or len(images) == 0:
            return None
        ncols = 1
        while True:
            if ncols**2 >= len(images):
                break
            ncols += 1
        nrows = (len(images) + ncols - 1) // ncols
        h, w = images[0].shape[:2]
        grid = np.zeros((h * nrows, w * ncols, 3), dtype=np.uint8)
        for i in range(nrows):
            for j in range(ncols):
                index = ncols * i + j
                if index >= len(images):
                    break
                grid[h * i:h * (i + 1), w * j:w * (j + 1)] = images[index]
        return grid

    def run_with_translation(
        self, text: str, translate: bool, style: str, seed: int,
        only_first_stage: bool, num: int
    ) -> tuple[str | None, np.ndarray | None, list[np.ndarray] | None]:
        if translate:
            text = translated_text = self.translator(text)
        else:
            translated_text = None
        results = self.run(text, style, seed, only_first_stage, num)
        grid_image = self.make_grid(results)
        return translated_text, grid_image, results
