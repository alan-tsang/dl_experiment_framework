from setuptools import setup, find_packages

setup(
    name='y_rgb',
    version='1.3.0',
    packages = find_packages(),
    url='https://www.github.com/alan-tsang/dl_experiment_framework',
    license='LGPL',
    author='zhicun zeng',
    author_email='leibzc@163.com',
    description='a personal deep learning experiment framework for pytorch',
    long_description='you are really a good boy! Y-RGB!',
    install_requires=[
        "torch>=1.11.0", # 支持torchrun的版本
        "datasets",
        "omegaconf>=1.4.0", # 之前是否支持from_cli没有查到
        "numpy",
        "pandas",
        "pyyaml"
    ],
    extras_require = {
        "rich": ["rich"], # for show model info more beautiful in analysis.*.py
        "rdkit": ["rdkit"], # for common.dl_util.py
        "wandb": ["wandb"], # for callback.wandb.py
        "transformers": ["transformers"], # for model.base_pretrained_model.py
        "deepspeed": ["deepspeed"],
        "torch_geometric": ["torch_geometric"],# for model.gnn.py
        "torch_scatter": ["torch_scatter"],# for model.gnn.py
        "other": ["matplotlib", "torchviz", "graphviz", "pynvml"]
    },
)
