List of commands (excluding basic commands for configuring the OS) used to set the requirements versions in the VM used:

```
# !sed -i '/flash-attn/d' requirements.txt
# !sed -i 's/cchardet/faust-cchardet/g' requirements.txt
# !sed -i 's/opencv-python/opencv-python-headless/g' requirements.txt
# !pip install -r requirements.txt
# !pip install -r requirements_demo.txt

# it seems that some packages are too old
# !pip install --upgrade accelerate peft transformers
# !pip install "numpy<2"
# !pip install opencv-python

# !pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu118
# !pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.3/flash_attn-2.7.3+cu11torch2.6cxx11abiFALSE-cp311-cp311-linux_x86_64.whl
```