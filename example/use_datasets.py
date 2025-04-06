from zero2hero.dataset.dataset import BaseIterableDataset, BaseMapDataset
from datasets import Dataset

def process_function(example):
    """清洗数据：小写化、去除标点"""
    processed_text = [text.lower().replace(".", "").replace("!", "").replace("...", "") for text in example["text"]]
    return {"text": processed_text, "label": example["label"]}

def filter_function(example):
    """过滤长度<20字符的评论"""
    return [len(text) >= 20 for text in example["text"]]


raw_data = Dataset.from_json('./data/local_movie_reviews.json')
print("原始数据：")
print(f"长度：{len(raw_data)}")
print(raw_data[0])
print("========")
dataset1 = BaseIterableDataset(
    data_source = raw_data,
    process_fn = process_function,
    filter_fn = filter_function,
    only_local = True,
    process_first = True,
    process_bs = 4,
    filter_bs = 4,
    metadata = {
        "task": "sentiment analysis",
        "language": "English"
    },
)
iterable_data_loader = dataset1.get_batch_loader(batch_size=4)
for i in iterable_data_loader:
    print(i)
print("===========")
# print(dataset1.dataset_card)


dataset2 = BaseMapDataset(
    data_source = './data/local_movie_reviews.json',
    only_local = True,
    process_fn = process_function,
    filter_fn = filter_function,
    metadata = {
        "task": "sentiment analysis",
        "language": "English"
    },
)
oapsdoao = dataset2.get_subset("train", list(range(10)))
for data in oapsdoao:
    print(data)

print(f"数据集分类：{dataset2.dataset.keys()}" )
print(f"训练数据集大小（处理后）: {len(dataset2.get_split('train'))}")
print(f"验证数据集大小（处理后）: {len(dataset2.get_split('valid'))}")
print(f"测试数据集大小（处理后）: {len(dataset2.get_split('test'))}")
print("处理后的样本示例：")
dataset_train = dataset2.get_split('train')
for i in range(2):
    print(dataset_train[i])

# gg = dataset2.get_subset('train', list(range(10)))
# print(type(gg.dataset))
# for i in gg:
#     print(i)

print("\n数据集卡片：")
print(dataset2.dataset_card)

print("如何获取子集：")
print(len(dataset_train.select(list(range(10)))))
