# CogView2 demo
This is an unofficial demo app for [CogView2](https://github.com/THUDM/CogView2).

You can try out a web demo: [![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/THUDM/CogView2) (This version currently supports only the first stage.)

![screenshot](assets/screenshot.jpg)

It takes about 3 minutes to load models and about 1 minute to generate 8 images.

## Prerequisite
An A100 instance is required to run CogView2.

## Installation
### Change default-runtime of docker
First, put `"default-runtime": "nvidia"` in `/etc/docker/daemon.json`.
See: https://github.com/NVIDIA/nvidia-docker/issues/1033#issuecomment-519946473
```json
{
    "runtimes": {
        "nvidia": {
            "path": "/usr/bin/nvidia-container-runtime",
            "runtimeArgs": []
        }
    },
    "default-runtime": "nvidia"
}
```

Then, restart docker.
```bash
sudo systemctl restart docker
```

### Clone this repo
```bash
git clone --recursive https://github.com/hysts/CogView2_demo
cd CogView2_demo
```

### Build docker image
```bash
docker build . -t cogview2
```

### Apply patch to CogView2 repo
```bash
cd CogView2
patch -p1 < ../patch
```

### Download pretrained models (Optional)
You can download the pretrained models from [Hugging Face Hub](https://huggingface.co/THUDM/CogView2) with the following command:
```bash
pip install huggingface_hub

python download_pretrained_models.py
```

The total size of the models is approximately 40 GB.
The above script downloads and extracts zip files of them, so twice as much disk space is needed.

This repo assumes the pretrained models are stored in the `pretrained` directory as follows:
```
pretrained
├── coglm
│   ├── 432000
│   │   └── mp_rank_00_model_states.pt
│   ├── latest
│   └── model_config.json
├── cogview2-dsr
│   ├── 20000
│   │   └── mp_rank_00_model_states.pt
│   ├── latest
│   └── model_config.json
└── cogview2-itersr
    ├── 20000
    │   └── mp_rank_00_model_states.pt
    ├── latest
    └── model_config.json
```

The pretrained models will be downloaded automatically on the first run,
but it may take quite some time.
So you may want to download them in advance.
Also, downloading from the Hugging Face Hub using the command above may be way faster than the automatic download.

## Run
You can run the app with the following command:
```bash
docker compose run --rm app
```

The app will start up on port 7860 by default.
You can change the port using `GRADIO_SERVER_PORT` environment variable.
Use port forwarding when running on GCP, etc.
