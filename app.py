#!/usr/bin/env python

from __future__ import annotations

import argparse
import os
import pathlib
import subprocess

import gradio as gr

from model import Model

DESCRIPTION = '''# CogView2 (text2image)

This is an unofficial demo for <a href="https://github.com/THUDM/CogView2">https://github.com/THUDM/CogView2</a>.
'''


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--only-first-stage', action='store_true')
    parser.add_argument('--share', action='store_true')
    return parser.parse_args()


def set_example_text(example: list) -> dict:
    return gr.Textbox.update(value=example[0])


def main():
    args = parse_args()
    model = Model(args.only_first_stage)

    with gr.Blocks(css='style.css') as demo:
        gr.Markdown(DESCRIPTION)

        with gr.Row():
            with gr.Column():
                with gr.Group():
                    text = gr.Textbox(label='Input')
                    style = gr.Dropdown(choices=[
                        'mainbody',
                        'photo',
                        'flat',
                        'comics',
                        'oil',
                        'sketch',
                        'isometric',
                        'chinese',
                        'watercolor',
                    ],
                                        label='Style')
                    seed = gr.Slider(0,
                                     100000,
                                     step=1,
                                     value=1234,
                                     label='Seed')
                    only_first_stage = gr.Checkbox(
                        label='Only First Stage',
                        value=args.only_first_stage,
                        visible=not args.only_first_stage)
                    num_images = gr.Slider(1,
                                           16,
                                           step=1,
                                           value=8,
                                           label='Number of Images')
                    with open('samples.txt') as f:
                        samples = [[line.strip()] for line in f.readlines()]
                    examples = gr.Dataset(components=[text], samples=samples)
                    run_button = gr.Button('Run')

            with gr.Column():
                result = gr.Gallery(label='Output')

        run_button.click(fn=model.run,
                         inputs=[
                             text,
                             style,
                             seed,
                             only_first_stage,
                             num_images,
                         ],
                         outputs=result)
        examples.click(fn=set_example_text,
                       inputs=examples,
                       outputs=examples.components)

    demo.launch(
        enable_queue=True,
        share=args.share,
    )


if __name__ == '__main__':
    main()
