

## **Datasets**

Download datasets from these links and put them in the corresponding folder:

* [Twitter](https://goo.gl/5Enpu7)
* [Lap14](https://alt.qcri.org/semeval2014/task4)
* [Rest14](https://alt.qcri.org/semeval2014/task4)

## **Usage**

1. Install dependencies

```bash
pip install -r requirements.txt
```

2. Prepare RoBERTa model
Download RoBERTa base model and put it in ./models/Roberta/
3. The vocabulary files should be in each dataset directory

```
python prepare\_vocab.py
```

## **Training**

```
# Train on Laptop dataset
python train.py --model\_name dce-gtn --dataset laptop --cuda 0

# Train on Restaurant dataset
python train.py --model\_name dce-gtn --dataset restaurant --cuda 0

# Train on Twitter dataset
python train.py --model\_name dce-gtn --dataset twitter --cuda 0
```

