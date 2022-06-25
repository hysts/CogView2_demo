# CogView2 demo
This is an unofficial demo app for [CogView2](https://github.com/THUDM/CogView2).

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

## Run
```bash
docker compose run --rm cogview2 python app.py
```
