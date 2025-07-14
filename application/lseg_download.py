import gdown
import os

def main():
    checkpoint_dir = "/Bench_LLM_Nav/vlmaps/lseg/checkpoints"
    checkpoint_path = "/Bench_LLM_Nav/vlmaps/lseg/checkpoints/demo_e200.ckpt"
    checkpoint_url = "https://drive.google.com/u/0/uc?id=1ayk6NXURI_vIPlym16f_RG3ffxBWHxvb"
    os.makedirs(checkpoint_dir, exist_ok=True)
    gdown.download(checkpoint_url, output=str(checkpoint_path))

if __name__ == '__main__':
    main()
