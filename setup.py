from setuptools import setup

setup(
    name='zero2hero',
    version='1.0.0',
    packages=['zero2hero', 'zero2hero.dist', 'zero2hero.model', 'zero2hero.model.transformer', 'zero2hero.common', 'zero2hero.config', 'zero2hero.runner', 'zero2hero.callback'],
    url='https://www.github.com/alan-tsang',
    license='LGPL',
    author='zhicun zeng',
    author_email='leibzc@163.com',
    description='personal deep learning experiment framework',
    install_requires=[
        "torch>=2.1.2",
        "wandb>=0.19.1",
        "datasets>=3.3.2",
        "omegaconf>2.3.0",
        "nvidia_ml_py>=12.570.86",
        "scikit_learn>=1.6.1"
        "pynvml>=12.0.0",
        "numpy>=1.21.1",
        "rich>=13.9.4",
        "fonttools>=4.55.6"
    ],
    extras_require = {
        "deep_speed": ["deepspeed>=0.16.2"],
        "torch_geometric": ["torch_geometric>=2.0.1"],
        "torch_scatter": ["torch_scatter>=2.1.2"],
    },
)
