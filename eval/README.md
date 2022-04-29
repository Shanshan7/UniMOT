# use motmetrics
## 1.Set up  
Install py-motmetrics. Follow the instructions here: https://github.com/cheind/py-motmetrics  
or  
```
pip install motmetrics
```

## Data preparation
1. Need to prepare benchmark data 'gt.txt' and verification data 'result.txt' for each seq  
2. And prepare the data in the following structure:
```
seqs
   |——————seq1
   |        └——————gt.txt
   |        └——————result.txt
   └——————seq2
   └——————......
```
## 2.Run  
1. Modify ground_truth_path, result_path and summary_path in eval.sh according to your own path of eval  
2. run eval.sh to get the result:
```
sh eval.sh
```