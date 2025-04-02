from setuptools import setup

setup(
    name='zero2hero',
    version='1.3.0',
    packages=['zero2hero', 'zero2hero.dist', 'zero2hero.model', 'zero2hero.model.transformer', 'zero2hero.common', 'zero2hero.config', 'zero2hero.runner', 'zero2hero.callback'],
    url='https://www.github.com/alan-tsang',
    license='LGPL',
    author='zhicun zeng',
    author_email='leibzc@163.com',
    description='personal deep learning experiment framework for pytorch',
    install_requires=[
        "torch",
        "datasets",
        "omegaconf",
        "nvidia_ml_py",
        "scikit_learn",
        "pynvml",
        "numpy",
        "rich",
        "fonttools"
    ],
    extras_require = {
        "wandb": ["wandb"],
        "transformers": ["transformers"],
        "deepspeed": ["deepspeed"],
        "torch_geometric": ["torch_geometric"],
        "torch_scatter": ["torch_scatter"],
    },
)
