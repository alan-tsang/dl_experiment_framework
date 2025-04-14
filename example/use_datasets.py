from y_rgb.dataset.dataset import BaseIterableDataset, BaseMapDataset

def process_function(example):
    """清洗数据：小写化、去除标点"""
    processed_text = [text.lower().replace(".", "").replace("!", "").replace("...", "") for text in example["text"]]
    return {"text": processed_text, "label": example["label"]}

def filter_function(example):
    """过滤长度<20字符的评论"""
    return [len(text) >= 20 for text in example["text"]]


dataset_iterable = BaseIterableDataset(
    data_source = './data/local_movie_reviews.json',
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
print("====== 迭代型数据集(减少内存占用，maybe适合调试时使用) =====")
iterable_data_loader = dataset_iterable.get_batch_loader(batch_size=4)
for i in iterable_data_loader:
    print(i)
dataset_iterable.save_to_disk('./data/dataset_iterable')


print("====== Map Style数据集 =====")
dataset_map = BaseMapDataset(
    data_source = './data/local_movie_reviews.json',
    only_local = True,
    process_fn = process_function,
    filter_fn = filter_function,
    metadata = {
        "task": "sentiment analysis",
        "language": "English"
    },
)
dataset_train = dataset_map.get_split('train')
for i in range(2):
    print(dataset_train[i])

sample_dataset = dataset_map.sample('train', 5)
for data in sample_dataset:
    print(data)
dataset_map.save_to_disk('./data/dataset_map')
