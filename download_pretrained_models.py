#!/usr/bin/env python

import pathlib
import shutil
import zipfile

import huggingface_hub


def download_and_extract_cogview2_model(name: str,
                                        root_dir: pathlib.Path) -> None:
    path = huggingface_hub.hf_hub_download('THUDM/CogView2', name)
    stem = name.split('.')[0]
    model_dir = root_dir / stem
    if model_dir.exists():
        return
    with zipfile.ZipFile(path) as f:
        f.extractall('/tmp')
    shutil.move(f'/tmp/sharefs/cogview-new/{stem}', model_dir)
    shutil.rmtree('/tmp/sharefs')


def main():
    model_root_dir = pathlib.Path('pretrained')
    names = [
        'coglm.zip',
        'cogview2-dsr.zip',
        'cogview2-itersr.zip',
    ]
    for name in names:
        download_and_extract_cogview2_model(name, model_root_dir)


if __name__ == '__main__':
    main()
