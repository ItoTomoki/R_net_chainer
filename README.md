# R-NET: MACHINE READING COMPREHENSION WITH SELF MATCHING NETWORKS

Chainer implementation of https://www.microsoft.com/en-us/research/wp-content/uploads/2017/05/r-net.pdf

## Requirements
  * NumPy
  * tqdm
  * Chainer == 2.0

# Downloads and Setup
Once you clone this repo, run the following lines from bash **just once** to process the dataset (SQuAD).
```shell
$ git clone https://github.com/NLPLearn/R-net.git
$ cd R-net
$ pip install -r requirements.txt
$ bash setup.sh
$ python process.py --process True
```

# Training
To train the model, run the following line.

If you use cpu,
```shell
$ python R_net_execute.py
```

If you use gpu,
```shell
$ python R_net_execute_gpu.py
```

# Evaluation
```shell
$ python model_evaluation.py
```
