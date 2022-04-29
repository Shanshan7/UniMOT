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
eval
   |——————seq1
   |        └——————gt.txt
   |        └——————result.txt
   └——————seq2
   └——————......
```
## 2.Run  
1. Modify ground_truth_dir(line 56) and hypothesis_dir(line 57) in run_metrics.py according to your own path of eval  
2. Modify xlsx_dir(line 78)  in run_metrics.py
3. run run_metrics.py to get the result:
```
python run_metrics.py
```