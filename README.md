## Dataset Information
The LMT texture dataset can be downloaded from:
[https://zeus.lmt.ei.tum.de/downloads/texture](https://zeus.lmt.ei.tum.de/downloads/texture)

## Environment Setup
This project requires the following Python environment:

- Python 3.11

- PyTorch 2.5.0

- scikit-learn 1.4.2

- seaborn 0.13.2

- matplotlib 3.8.4

### Recommended Installation
You can set up the environment using pip:

```bash
pip install torch==2.5.0 scikit-learn==1.4.2 seaborn==0.13.2 matplotlib==3.8.4
```
### Usage
Run training:
```bash
python main.py --mode train \
               --data_path   \
               --epochs 100 \
               --batch_size 32 \
               --lr 0.001 \
               --model_save_path ./checkpoints/model.pth
```
Evaluation:
```bash
python main.py --mode eval \
               --data_path  \            
               --model_load_path ./checkpoints/model.pth
```
