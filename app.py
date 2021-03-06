#!/usr/bin/env python

from __future__ import annotations

import argparse

import gradio as gr

from model import AppModel

DESCRIPTION = '''# <a href="https://github.com/THUDM/CogView2">CogView2</a> (text2image)

This application accepts English or Chinese as input.
In general, Chinese input produces better results than English input.
If you check the "Translate to Chinese" checkbox, the app will use the English to Chinese translation results with [this Space](https://huggingface.co/spaces/chinhon/translation_eng2ch) as input.
But the translation model may mistranslate and the results could be poor.
So, it is also a good idea to input the translation results from other translation services.
'''


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--only-first-stage', action='store_true')
    parser.add_argument('--share', action='store_true')
    return parser.parse_args()


def set_example_text(example: list) -> list[dict]:
    return [
        gr.Textbox.update(value=example[0]),
        gr.Dropdown.update(value=example[1]),
    ]


def main():
    args = parse_args()
    model = AppModel(args.only_first_stage)

    with gr.Blocks(css='style.css') as demo:
        gr.Markdown(DESCRIPTION)

        with gr.Row():
            with gr.Column():
                with gr.Group():
                    text = gr.Textbox(label='Input Text')
                    translate = gr.Checkbox(label='Translate to Chinese',
                                            value=False)
                    style = gr.Dropdown(choices=[
                        'none',
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
                                        value='mainbody',
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
                    run_button = gr.Button('Run')

                    with open('samples.txt') as f:
                        samples = [
                            line.strip().split('\t') for line in f.readlines()
                        ]
                    examples = gr.Dataset(components=[text, style],
                                          samples=samples)

            with gr.Column():
                with gr.Group():
                    translated_text = gr.Textbox(label='Translated Text')
                    with gr.Tabs():
                        with gr.TabItem('Output (Grid View)'):
                            result_grid = gr.Image(show_label=False)
                        with gr.TabItem('Output (Gallery)'):
                            result_gallery = gr.Gallery(show_label=False)

        run_button.click(fn=model.run_with_translation,
                         inputs=[
                             text,
                             translate,
                             style,
                             seed,
                             only_first_stage,
                             num_images,
                         ],
                         outputs=[
                             translated_text,
                             result_grid,
                             result_gallery,
                         ])
        examples.click(fn=set_example_text,
                       inputs=examples,
                       outputs=examples.components)

    demo.launch(
        enable_queue=True,
        share=args.share,
    )


if __name__ == '__main__':
    main()
