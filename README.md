# dask-xgboost

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


```
KilledWorker: Attempted to run task ('shuffle-transfer-d80b45ae0d76f301654c24d36b0796fc', 20) on 4 different workers, but all those workers died while running it....
```
<img width="800" alt="image" src="https://github.com/user-attachments/assets/78cf0201-7694-4a4c-adda-490f636b6606" />


