from clearml import Task

# ClearMLに接続してタスクを作成
task = Task.init(project_name="Test Project", task_name="Execution Check")

# ログに出力
print("✅ ClearMLでの実行確認 OK！")

