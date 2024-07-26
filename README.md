# OMLog: Online Log Anomaly Detection for Evolving Systems with Meta-learning

## Pre-requisites:

The experiment is based on a Pytorch implementation running in the following environment

```
 conda install --yes --file requirements.txt
```
## Dataset
Our approach follows the work in LogOnline. Therefore, all experiments were performed on two public log datasets, HDFS and BGL. 
where HDFS is a stable dataset, and BGL is a evolving dataset. However, due to Github's settings for space, we were unable to upload these two datasets. 
The open source datasets and their structured versions are available in [LogADEmpirical](https://github.com/LogIntelligence/LogADEmpirical/tree/icse2022). The downloaded files should be placed in the ‘/data’ folder in a directory named after the dataset.
For example, the parsed HDFS file ‘HDFS.log_structured.csv’ should be placed under ‘/data/HDFS/HDFS.log-structured.csv’.

## Log Parsing

The log parser used by LogOnline is [Spine](https://github.com/pfeak/spell), another work of the team, 
but details of Spine's implementation are missing. 
In addition, the paper proposing Spine does not open source the parser. 
Since log parsing is not the focus of our research. 
Therefore, OMLog makes use of the classic [Spell]() parser instead and places the source code of the implementation in src/spell.py.

## Pre-training Model
The normality detection model is derived from LogOnline, and we re-trained the normality detection model based on the Spell parsed dataset according to its source code. 
The code to implement the normality detection model is placed in src/aemodeltrain.py and src/aefeature.py.
Also, wiki-news-300d-1M.vec is available for download at [wiki-news-300d-1M-subword.vec](https://fasttext.cc/docs/en/english-vectors.html)

## Running of OMLog
You can run the code by clicking on OMLog.py directly in the root directory after installing the environment. 
All the parameters can be adjusted in OMLog.py.

```
python OMLog.py
```

## Methods of comparison

DeepLog | LogAnomaly:  [Code](https://github.com/xUhEngwAng/LogOnline)

LogRobust | PLELog|CNN: [Code](https://github.com/LogIntelligence/LogADEmpirical/tree/icse2022)

ROEAD: [Code](https://github.com/JasonHans/ROEAD-core-code)


## Acknowledgements
We acknowledge the work done by the LogOnline approach, and our code is implementation based on the [LogOnline](https://github.com/xUhEngwAng/LogOnline) .

