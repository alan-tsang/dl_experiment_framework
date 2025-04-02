# Zero2Hero
personal deeplearning experiment framework for pytorch

## News
+ April 02, 2025: support evaluator offline and online
+ April 01, 2025: support base model based on nn.Module, PretrainedModel, and PretrainedModelConfig
+ March 11, 2025: support experiment dataset integration based on datasets

## Environment
Need to install the following packages:
+ [x] pytorch
+ [x] datasets
+ [x] omegaconf

Optional packages:
+ [x] deepspeed
+ [x] wandb
+ [x] transformers
+ [x] torch_scatter
+ [x] torch_geometric


## Feature
For Experiment(炼丹):
+ [x] multi experiment support based on omegaconf and wandb sweep
+ [x] multi monitor value for training, validation, and test
+ [x] dataset integration based on datasets, including map style and iter style
+ [x] different base model based on nn.Module, PretrainedModel, and PretrainedModelConfig
+ [x] Independent training, verification, testing
+ [x] tune hyperparameters based on wandb sweep
+ [x] evaluator offline and online
+ [x] distributed collect and dump results


For Training Trick:
+ [x] deepspeed support
+ [x] fp16 support
+ [x] grad accumulation
+ [x] grad clip
+ [x] early_stopping
+ [x] progress bar like pylightning
+ [x] warmup lr scheduler
+ [x] builtin epoch summary, model summary
+ [x] model analysis: complexity, flop, activation
+ [x] extendable callback

