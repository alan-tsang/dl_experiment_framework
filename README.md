# Y_RGB
You are really a good boy!

My personal deeplearning experiment framework for pytorch.

You can use as your wish, but I am too lazy and busy to provide documentation.

## News
+ April 14, 2025: support model activation checkpoint, better dataset integration
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
+ [x] multi experiment support based on omegaconf and wandb sweep. Therefore, we can easily compare the experimental results
+ [x] real-time multi monitor value for training, validation, and test
+ [x] dataset integration based on datasets and  pytorch, including map style and iter style. 
+ [x] different base model based on nn.Module, PretrainedModel, and PretrainedModelConfig which conclude basic functions like load, save and metric
+ [x] Independent training, verification, testing. This means you no longer need to write inference code.
+ [x] tune hyperparameters based on wandb sweep, so only a yaml is enough
+ [x] evaluator offline and online
+ [x] automatic distributed collect and dump results


For Training Trick:
+ [x] deepspeed support
+ [x] fp16 support
+ [x] activation checkpoint
+ [x] grad accumulation
+ [x] grad clip
+ [x] early_stopping
+ [x] progress bar like pylightning
+ [x] warmup lr scheduler
+ [x] builtin epoch summary, model summary
+ [x] model analysis: complexity, flop, activation
+ [x] extendable callback based on many lifecycle of runner

