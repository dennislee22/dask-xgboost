# XGboost with Dask

```
cdsw@df8gqpsco0hlse3g:~$ pip list | grep dask
dask                               2025.5.1
dask-glm                           0.3.2
dask-ml                            2025.1.0
cdsw@df8gqpsco0hlse3g:~$ pip list | grep xgboost
xgboost                            3.0.2
```

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
![dask-xgboost-5w](https://github.com/user-attachments/assets/e31e6444-1da4-4771-88ef-5156817e5b59)


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

- Train XGboost model with Pandas (8G RAM)
```
$ python train-xboost-pandas.py 

Reading '3G_cdr_data.csv' with pandas...
Killed
```

<img width="1182" height="217" alt="image" src="https://github.com/user-attachments/assets/ae627dff-0687-4445-ad95-1998957f96dd" />

- Train XGboost model with Pandas (12G RAM)

```
cdsw@reqythscmmghof9g:~$ python train-xboost-pandas.py 

Reading '3G_cdr_data.csv' with pandas...
Performing feature engineering with pandas...

Training the XGBoost model with pandas/scikit-learn...
Calculating scale_pos_weight for class imbalance...
/home/cdsw/train-xboost-pandas.py:58: FutureWarning: Series.__getitem__ treating keys as positions is deprecated. In a future version, integer keys will always be treated as labels (consistent with DataFrame behavior). To access a value by position, use `ser.iloc[pos]`
  scale_pos_weight = y_train.value_counts()[0] / y_train.value_counts()[1]
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
cdsw@reqythscmmghof9g:~$ 
```

<img width="1180" height="211" alt="image" src="https://github.com/user-attachments/assets/0cf5a50f-a489-4cb3-846c-91229ce9289e" />



## Best Practice

```
KilledWorker: Attempted to run task ('shuffle-transfer-d80b45ae0d76f301654c24d36b0796fc', 20) on 4 different workers, but all those workers died while running it....
```
<img width="800" alt="image" src="https://github.com/user-attachments/assets/78cf0201-7694-4a4c-adda-490f636b6606" />


