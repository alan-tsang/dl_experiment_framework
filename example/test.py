"""不知道怎么用dataset.from_dict

"""
from zero2hero.dataset.base_dataset import BaseDataset
from zero2hero.dataset.dataset import BaseIterableDataset, BaseMapDataset
from datasets import Dataset

if __name__ == "__main__":
    def process_function(example):
        """清洗数据：小写化、去除标点"""
        processed_text = example["text"].lower().replace(".", "").replace("!", "").replace("...", "")
        return {"text": processed_text, "label": example["label"]}

    def filter_function(example):
        """过滤短于20字符的评论"""
        return len(example["text"]) >= 20

    x = Dataset.from_json('./local_movie_reviews.json').to_iterable_dataset()
    dataset1 = BaseIterableDataset(
        data_source = x,
        process_fn = process_function,
        filter_fn = filter_function,
        metadata = {
            "task": "sentiment analysis",
            "language": "English"
        },
    )
    iterable_data_loader = dataset1.get_batch_loader(batch_size=2)
    for i in iterable_data_loader:
        print(i)

    dataset2 = BaseMapDataset(
        data_source = './local_movie_reviews.json',
        process_fn = process_function,
        filter_fn = filter_function,
        metadata = {
            "task": "sentiment analysis",
            "language": "English"
        },
    )
    # 查看处理、过滤结果
    print(f"数据集大小（处理后）: {len(dataset2)}")
    print("处理后的样本示例：")
    for i in range(2):
        print(dataset2[i])

    # 验证过滤效果
    print("\n过滤掉的样本：")
    original_indices = [i for i, x in enumerate(dataset2) if not filter_function(process_function(x))]
    for i in original_indices:
        print(f"原数据索引 {i}: {dataset2[i]['text']}")

    print("\n数据集卡片：")
    print(dataset2.dataset_card)

    # dataset2.save_to_disk(save_path)


    # data = {
    #     "text": [
    #         "Great movie! Loved the plot.",  # 1
    #         "Terrible... Waste of time.",  # 0
    #         "The acting was phenomenal. Really enjoyed it.",  # 1
    #         "Not bad.",  # 0
    #         "The cinematography is stunning. A visual masterpiece.",  # 1
    #         "Absolutely fantastic! Best film of the year.",  # 1
    #         "The characters were flat and the story dragged on.",  # 0
    #         "A heartwarming story with brilliant performances.",  # 1
    #         "The dialogue felt forced and unnatural.",  # 0
    #         "This movie left me speechless. A must-watch!",  # 1
    #         "The pacing was too slow, couldn't finish it.",  # 0
    #         "The soundtrack was amazing and really added to the experience.",  # 1
    #         "Full of clichés and predictable twists.",  # 0
    #         "A perfect blend of humor and emotion.",  # 1
    #         "Awful acting and a nonsensical plot.",  # 0
    #         "Visually creative but the script was weak.",  # 0
    #         "Captivating from start to finish!",  # 1
    #         "Boring and unoriginal. Skip this one.",  # 0
    #         "The director's vision truly shines here.",  # 1
    #         "Technical excellence wasted on a shallow story.",  # 0
    #         # 新增20行
    #         "An unforgettable journey with deep character development.",  # 1
    #         "The special effects ruined the entire experience.",  # 0
    #         "Masterful storytelling that keeps you guessing.",  # 1
    #         "Painfully slow buildup with no payoff.",  # 0
    #         "The chemistry between lead actors was electric.",  # 1
    #         "Worst screenplay I've seen in decades.",  # 0
    #         "Innovative camera work that redefines the genre.",  # 1
    #         "Laughably bad CGI throughout the film.",  # 0
    #         "Powerful themes that resonate long after viewing.",  # 1
    #         "The plot holes were big enough to drive a truck through.",  # 0
    #         "A refreshing take on the classic hero's journey.",  # 1
    #         "Dialogue so cringey it made me physically uncomfortable.",  # 0
    #         "The costume design deserved an Oscar nomination.",  # 1
    #         "Editing was chaotic and disjointed.",  # 0
    #         "Emotionally raw and brutally honest storytelling.",  # 1
    #         "Overacted scenes destroyed any sense of realism.",  # 0
    #         "A thought-provoking exploration of human nature.",  # 1
    #         "The romantic subplot felt completely unnecessary.",  # 0
    #         "Every frame oozes artistic brilliance.",  # 1
    #         "Predictable from the first scene to the last.",  # 0
    #     ],
    #     "label": [
    #         1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0,  # 前20行
    #         1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0  # 新增20行
    #     ]
    # }
