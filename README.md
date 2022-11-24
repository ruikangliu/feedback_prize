# To start

- Download [Feedback Prize Dataset](https://www.kaggle.com/competitions/feedback-prize-english-language-learning/data) and change settings for dataset path in `config.py` (i.e., `CFG.train_file`, `CFG.test_file`, `CFG.submission_file`)
- Download transformer models (e.g. microsoft/deberta-v3-base) from [Huggingface](https://huggingface.co/) and change settings for model path in `config.py` (i.e., `CFG.model_dir`)

***

Train models

```bash
python train.py --model=microsoft/deberta-v3-base --loss_func=RMSE --pooling=mean --gpu_id=0
```

