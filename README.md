<!--             
<style>
  .texttt {
    font-family: Consolas; /* Monospace font */
    font-size: 1em; /* Match surrounding text size */
    color: teal; /* Text color */
    letter-spacing: 0; /* Adjust if needed */
  }
</style> -->


## Installation

We tested our code using CUDA 12.8 and Python 3.10. To install requirements, run in the terminal:

```bash
conda create -n ddpm_training python=3.10

conda activate ddpm_training

pip install numpy tqdm pillow pyYaml
```
Linux: 
```bash
pip3 install torch torchvision torchaudio
```
Windows: 
```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```
Remaining libraries:
```bash
pip install opencv-python
pip install wandb
pip install wandb[media]
```
