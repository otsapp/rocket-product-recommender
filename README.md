# Rocket product recommender  

## Purpose 

Collaborative filtering, implicit feedback approach to product recommendation and a chance to play around with LighFM library for matrix factorisation.

Had a go at going a different approach with word embeddings but the results wern't good enough to cluster. 

## Data

Only using the user event data for now (clicks, basket, transactions) `https://www.kaggle.com/retailrocket/ecommerce-dataset`

## Approach 

Inspired by Jay Do's Kernel: `https://www.kaggle.com/dohuan/retailrocket-data-train-test-with-lightfm`

## Run model training

1. Clone repo

2. Download events.csv into the `data` directory from `https://www.kaggle.com/retailrocket/ecommerce-dataset`.

3. install dependencies: `pip install -r requirements.txt`

4. With src as pwd, run `python train.py` in shell.

Model parameters that can be added as arguments to the command above: 

    --no-components 		(type=int, default=5)
    --loss-method 			(type=str, default='warp')
    --learning-schedule 	(type=str, default='adagrad')
    --epochs 				(type=int, default=100)
    --num_threads 			(type=int, default=1)

