import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 先下载模型
from paddlenlp import Taskflow

print("正在下载模型，请耐心等待...")
try:
    # 先下载一个简单的模型测试
    test_ie = Taskflow(
        task="information_extraction",
        schema=['测试'],
        model="uie-tiny",  # 先用小模型测试
        use_fast=False,
        device_id=-1,
        batch_size=1,
        max_seq_len=128
    )
    print("模型下载成功！")

    # 测试小模型
    result = test_ie("这是一个测试")
    print("测试结果:", result)

except Exception as e:
    print(f"下载失败: {e}")
    import traceback

    traceback.print_exc()