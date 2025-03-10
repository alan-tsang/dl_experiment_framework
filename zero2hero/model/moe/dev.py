"""
Adapted from https://github.com/microsoft/Megatron-DeepSpeed/blob/main/megatron/model/transformer.py
Adapted from https://github.com/microsoft/Megatron-DeepSpeed/blob/main/megatron/model/gpt_model.py

"""
import deepspeed
# megatron-lm 1.1.5
from deepspeed.moe.layer import MoE
from omegaconf import OmegaConf
from mlp import MLP

configs = {"hidden_size": 512}
# global configs
_num_experts = configs.get("num_experts", "2 2 2 2 4 4")
_num_layers = configs.get("num_layers", 6)
_expert_interval = configs.get("expert_interval", 1)

def get_num_experts_per_layer(num_experts: list , num_layers: int, expert_interval: int, offset: int = 0) -> list:
    num_experts = [int(x) for x in num_experts.split()]
    assert len(num_experts) == 1 or len(num_experts) == num_layers // expert_interval, \
        'num_experts must be either a single value or a list of the same length as the number of MoE layers'
    if len(num_experts) == 1:
        num_experts = num_experts * (num_layers // expert_interval)
    experts_per_layer = []
    for i in range(num_layers):
        layer_num = i + 1 + offset
        n_e = num_experts[(layer_num-1) // expert_interval] if layer_num % expert_interval == 0 else 1
        experts_per_layer.append(n_e)
    return experts_per_layer

experts_per_layer = get_num_experts_per_layer(
    _num_experts, _num_layers, _expert_interval
)
print(experts_per_layer)
is_moe_model = any(n_experts > 1 for n_experts in experts_per_layer)
print(is_moe_model)


megatron_args = {
    'num_layers': 1,
    "hidden_size": 512,
    'num_attention_heads': 8,
    'max_position_embeddings': 1024,
    "use_cpu_initialization": False
}


ALL_LAYER_MOE = []
for index, num_experts in enumerate(experts_per_layer):
    ALL_LAYER_MOE.append(
        MoE(
            configs["hidden_size"],
            expert = MLP(512, 1024),
            num_experts = num_experts,
            use_residual = True
        )
    )

print(ALL_LAYER_MOE)
