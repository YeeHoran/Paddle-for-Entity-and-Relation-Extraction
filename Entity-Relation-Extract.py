import os
from paddlenlp import Taskflow
from pprint import pprint

# 稳定运行设置
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ========== 1. 定义你的知识图谱Schema ==========
# 根据你的教材或文章内容，修改这里的实体和关系列表
# 示例：想从历史文章中提取人物、地点，以及他们的出生地、所属朝代关系
entities_for_ner = ['历史人物', '历史地点', '朝代']  # 第一步：要识别的实体类型
relations_for_re = {
    '历史人物': ['出生地', '所属朝代'],  # 第二步：要抽取的关系，键是主体实体
    # 可以添加更多，如：'历史地点': ['位于朝代', '相关事件']
}

# ========== 2. 初始化模型 ==========
# 注意：Taskflow对象一次只能执行一种schema，通常需要分别初始化
print("初始化实体抽取模型...")
ner = Taskflow("information_extraction", schema=entities_for_ner, model="uie-medium", device_id=-1)

print("初始化关系抽取模型...")
re = Taskflow("information_extraction", schema=relations_for_re, model="uie-medium", device_id=-1)

# ========== 3. 输入你的“一节内容” ==========
# 替换成你的实际文本，例如一段历史课文
your_text = """
唐太宗李世民（598年－649年），是唐朝的第二位皇帝。
他出生于武功别馆（今陕西省武功县）。
在位期间，首都为长安（今陕西西安）。
著名的宰相房玄龄（579年－648年）是齐州临淄（今山东省淄博市）人。
"""

print("待处理的文本内容：")
print(your_text)
print("\n" + "="*60)

# ========== 4. 第一步：抽取实体 ==========
print("第一步：实体抽取结果")
ner_results = ner(your_text)
pprint(ner_results)
print("="*60)

# ========== 5. 第二步：抽取关系 ==========
print("第二步：关系抽取结果")
re_results = re(your_text)
pprint(re_results)

# ========== 6. 数据后处理（示例） ==========
print("\n" + "="*60)
print("知识图谱三元组示例（需要从结果中解析）：")
# 这里只是演示如何理解结果，实际需要编写代码来组合实体和关系
for re_item in re_results:
    for entity_type, entities in re_item.items():
        for entity in entities:
            subject = entity['text']  # 主体，例如“唐太宗李世民”
            if 'relations' in entity:
                for relation_type, obj_list in entity['relations'].items():
                    for obj in obj_list:
                        object = obj['text']  # 客体，例如“武功别馆”
                        print(f"({subject}, {relation_type}, {object})")