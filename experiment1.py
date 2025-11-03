from clearml import Task
import random, time

# ClearML タスク初期化
task = Task.init(project_name="demo-project", task_name="experiment1")

# ハイパーパラメータを登録
params = {"epochs": 5, "lr": 0.01}
task.connect(params)

# ダミーのループ
for epoch in range(params["epochs"]):
    loss = random.random()
    print(f"Epoch {epoch+1} | loss={loss:.3f}")
    time.sleep(0.5)

# 正常終了
task.mark_completed()
