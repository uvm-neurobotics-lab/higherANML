# Overview

The "training" of a few-shots meta-learned model like ANML consists of:
- train-train: the model is shown a random (initially new) task, i.e. 20 chars from one Omniglot class (a.k.a. learning phase)
- train-test: the model is tested on the last task it has just trained on and a sample of 64 random images from all the classes (a.k.a. remembering phase)
- test-train: at test time the model is shown a few examples (15) of a new class not seen during train-train. Labels are also provided and the model is asked to quickly learn this new class without forgetting (for 600 the learning sequence is 9000 gradient updates long)
- test-test: after learning from the few-shots the model performance is evaluated on the holdout examples.

The code for these phases can be found inside [this](anml.py) file.

# Prerequisites

- Python3 
- PyTorch 1.6.0 
- [higher](https://github.com/facebookresearch/higher)
- numpy
- tqdm

# Training

Phase       | Trained    |  Optimizer  | LR
 ---------- | ---------  | ----------  | -----
 inner loop | rln + fc   | SGD         | 0.1
 outer loop | everything | Adam        | 0.003

Training can be performed using

```
python train_omni.py
``` 
flags can be provided for exploring different setups.

# Evaluation
![evaluation results](evaluation_results.png)

Number of tasks | Accuracy | Eval LR
----| ---- | ------- 
10  | 0.92 | 0.0010 
50  | 0.88 | 0.0007 
75  | 0.87 | 0.0006 
100 | 0.84 | 0.0004 
200 | 0.80 | 0.0004 
300 | 0.76 | 0.0003 
400 | 0.71 | 0.0003 
600 | 0.65 | 0.0003

Evaluation can be performed using

```bash
python eval_omni.py --model trained_anmls/256_112_2304_ANML-29999.pth --classes 10 --lr 0.00085 --runs 10
```

The model is sensitive to learning rates used, the data shown in the above table has been gathered sweeping several learning rates over multiple repeats.
