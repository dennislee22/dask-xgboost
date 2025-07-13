# XGBoost with Dask vs Pandas

- XGBoost (eXtreme Gradient Boosting) is a gradient boosting algorithm. It builds decision trees sequentially, with each new tree attempting to correct the errors of the previous ones. The XGBoost algorithm operates by iteratively adding weak learner decision trees to build a strong ensemble model.
- Pandas Dataframe has widely been used for data manipulation and analysis in Python. It organizes data into 2-D arrays with rows and columns, similar to a SQL/tabular table.
- XGBoost can be seamlessly integrated with Pandas for ML tasks. Pandas provides powerful tools for data manipulation and preprocessing, which can be applied to prepare data for XGBoost models.
- Dask enables distributed computations, allowing you to scale your ML tasks across multiple pods/nodes/machines. It is equipped with Dask Dataframe (Pandas-like) with lazy execution in which it runs a computational graph of tasks rather than executing a particular task immediately.
- This article examines how Dask is able to build XGBoost model with larger-than-memory datasets in a distributed Kubernetes platform, effectively addressing the limitations of using XGBoost solely with Pandas DataFrames. The use case is training a XGBoost model against 3GB of csv dataset. The model is used to help telco to check if a particular MSISDN/user is fradulent based on the captured CDR. The steps to achieve this use case include:
    1. [Create synthetic dataset](create-synthetic-cdr.py) (in batch to prevent running into OOM problem).
    2. Use dataframe to create feature engineering of the dataset ([dask-train-xgboost.ipynb](dask-train-xgboost.ipynb)).
    3. Train and test the model using XGBoost ([dask-train-xgboost.ipynb](dask-train-xgboost.ipynb)).
    4. [Perform inference with the trained model](model-inference.py) to make prediction on the new dataset.

## Test 1: Train XGboost model with larger-than-memory datasets
- Train XGboost model with Pandas with 3GB of csv dataset using a node of 8G RAM.
```tel
$ python train-xboost-pandas.py 

Reading '3G_cdr_data.csv' with pandas...
Killed
```
- The process is killed almost instantly because the dataset is too large to fit into the memory of a node.
  
<img width="1182" height="217" alt="image" src="https://github.com/user-attachments/assets/ae627dff-0687-4445-ad95-1998957f96dd" />

- Train XGboost model with Pandas with 3GB of csv dataset using a node of 12G RAM. The script completes successfully with larger memory size.

```
$ python train-xboost-pandas.py 

Reading '3G_cdr_data.csv' with pandas...
Performing feature engineering with pandas...

Training the XGBoost model with pandas/scikit-learn...
Calculating scale_pos_weight for class imbalance...
scale_pos_weight determined to be: 19.00

Model Evaluation on Test Set...
Confusion Matrix:
[[44646     4]
 [    0  2350]]

Classification Report:
              precision    recall  f1-score   support

       False       1.00      1.00      1.00     44650
        True       1.00      1.00      1.00      2350

    accuracy                           1.00     47000
   macro avg       1.00      1.00      1.00     47000
weighted avg       1.00      1.00      1.00     47000


Feature Importances:
mobility                195.0
total_calls             115.0
avg_duration            110.0
outgoing_call_ratio      95.0
nocturnal_call_ratio     51.0
std_duration              3.0
dtype: float64

Trained XGBoost model saved to 'fraud_detection_model_xgb2.json'
Process complete in 229.39 seconds.
```

<img width="1180" height="211" alt="image" src="https://github.com/user-attachments/assets/0cf5a50f-a489-4cb3-846c-91229ce9289e" />

## Test 2: Train XGboost model in a distributed K8s platform using Dask (5 workers)

1. Ensure the associated libraries have been installed.
```
$ pip list | egrep 'dask|bokeh|xgb'
bokeh                              3.7.3
dask                               2025.5.1
dask-glm                           0.3.2
dask-ml                            2025.1.0
xgboost                            3.0.2
```

2. Create a cluster in K8s platform with 6 pods (1 Dask scheduler and 5 Dask workers). Each pod has 8GB RAM. Run the first cell in [dask-train-xgboost.ipynb](dask-train-xgboost.ipynb). 
```
# kubectl -n cmlws5-user-1  get pods -o wide
NAME               READY   STATUS    RESTARTS   AGE     IP            NODE                          NOMINATED NODE   READINESS GATES
4dow92pvmcytb8u1   5/5     Running   0          3m      10.42.1.216   ecs-w-03.dlee5.cldr.example   <none>           <none>
65tgiynk4xp546jg   5/5     Running   0          2m47s   10.42.2.53    ecs-w-02.dlee5.cldr.example   <none>           <none>
hodvksf7cx1xltvc   5/5     Running   0          2m47s   10.42.2.50    ecs-w-02.dlee5.cldr.example   <none>           <none>
lswkxh89cgszutto   5/5     Running   0          2m47s   10.42.1.217   ecs-w-03.dlee5.cldr.example   <none>           <none>
m0ng1uix3eegqny7   5/5     Running   0          2m47s   10.42.3.93    ecs-w-01.dlee5.cldr.example   <none>           <none>
qufc1uc6wxx3x4qz   5/5     Running   0          2m47s   10.42.3.92    ecs-w-01.dlee5.cldr.example   <none>           <none>
tewpg0qzx5gu0yhm   5/5     Running   0          3m59s   10.42.3.91    ecs-w-01.dlee5.cldr.example   <none>           <none>
```

3. By running the second cell in [dask-train-xgboost.ipynb](dask-train-xgboost.ipynb), Python functions are handed over to Dask scheduler which distributes the tasks among the workers. 
![dask-xgboost-5w](https://github.com/user-attachments/assets/e31e6444-1da4-4771-88ef-5156817e5b59)

4. The result shows the model has been trained successfully with 8G RAM in each worker. This demonstrates Dask distributes the dataset among workers, preventing OOM. However, the completion time is longer than previous test using Pandas dataframe because MSISDN is set as an index with high degree of cardinality. A shuffle is a computationally expensive process for rearranging data across partitions. Operations like merge (the equivalent of a SQL join), set_index on an unsorted column, and certain groupby aggregations trigger a shuffle. In a distributed environment with multiple nodes/pods, it involves sending significant amounts of data over the network.

Test 3: Train XGboost model in a distributed K8s platform using Dask (10 workers)

1. Run the first cell in [dask-train-xgboost.ipynb](dask-train-xgboost.ipynb) with the following modification. 
```
k8s_pods = 10
dask_workers = workers.launch_workers(
    n=k8s_pods,
    cpu=1,
    memory=8,
    code=f"!dask-worker {scheduler_url}",
)
```
<img width="800" height="748" alt="image" src="https://github.com/user-attachments/assets/bf99a3dc-ba26-4a14-85f9-c69b603988dc" />

2. Run the second cell in [dask-train-xgboost.ipynb](dask-train-xgboost.ipynb). The result shows the completion time is faster than using Dask with 5 workers.
- Train XGboost model with 10 Dask workers
```
Dask client connected: <Client: 'tcp://10.42.1.227:8786' processes=10 threads=320, memory=73.57 GiB>

Reading '3G_cdr_data.csv' with Dask...
Performing feature engineering with Dask...

Training the XGBoost model with Dask...
Calculating scale_pos_weight for class imbalance...
scale_pos_weight: 18.96

Model Evaluation on Test Set...
[[45094     3]
 [    0  2352]]
              precision    recall  f1-score   support

       False       1.00      1.00      1.00     45097
        True       1.00      1.00      1.00      2352

    accuracy                           1.00     47449
   macro avg       1.00      1.00      1.00     47449
weighted avg       1.00      1.00      1.00     47449


Feature Importances:
mobility                178.0
total_calls             101.0
avg_duration             96.0
outgoing_call_ratio      83.0
nocturnal_call_ratio     60.0
std_duration              3.0
dtype: float64

Trained XGBoost model saved to 'fraud_detection_model_xgb.json'
Process complete in 346.85 seconds.
```

## Model Inference

- [Perform inference with the trained model](model-inference.py) to make prediction on the new dataset.
```
$ python model-inference.py new_cdr_data.csv
Loading model from 'fraud_detection_model_xgb2.json'...
Model loaded successfully.

Reading inference data 'new_cdr_data.csv' with pandas...
Performing feature engineering for inference with pandas...

Performing inference with the loaded model...

Saving predictions to 'predictions_new_cdr_data.csv'...

--- Inference Summary ---
Total MSISDNs processed: 50000
Number of MSISDNs predicted as fraudulent: 2508

Top 5 most likely fraudulent MSISDNs (by probability):
            total_calls  outgoing_call_ratio  avg_duration  std_duration  nocturnal_call_ratio  mobility  fraud_probability  is_predicted_fraud
msisdn                                                                                                                                         
6590000004       1430.0             0.995105     15.046762      4.886439              0.904196       1.0           0.999999                   1
6590033097        958.0             0.990605     15.181286      4.894379              0.907098       1.0           0.999999                   1
6590032978        571.0             0.991243     14.848366      4.776497              0.910683       1.0           0.999999                   1
6590032981       1306.0             0.992343     14.979778      5.012120              0.908882       1.0           0.999999                   1
6590032990        700.0             0.985714     15.051479      4.611223              0.897143       1.0           0.999999                   1

Reasoning: Predictions are based on the model learning from features such as:
total_calls, outgoing_call_ratio, avg_duration, std_duration, nocturnal_call_ratio, and mobility.
High fraud probability suggests these users' activity patterns resemble those of known fraud cases in the training data.
--------------------------

Inference process complete in 40.96 seconds.
``` 

## Tips
- The following Dask dashboard depicts tasks have to rerun 4 times till it fail due to insufficient memory to complete the first batch of tasks.

<img width="800" alt="image" src="https://github.com/user-attachments/assets/78cf0201-7694-4a4c-adda-490f636b6606" />
```
KilledWorker: Attempted to run task ('shuffle-transfer-d80b45ae0d76f301654c24d36b0796fc', 20) on 4 different workers, but all those workers died while running it....
```

