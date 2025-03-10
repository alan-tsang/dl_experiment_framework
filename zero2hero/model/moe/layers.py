"""
Adapted from
https://github.com/microsoft/Megatron-DeepSpeed/blob/main/megatron/model/transformer.py
https://github.com/microsoft/Megatron-DeepSpeed/blob/main/megatron/model/gpt_model.py

"""
from deepspeed.moe.layer import MoE
from torch import nn


def get_num_experts_per_layer(num_experts: list | str , num_layers: int, expert_interval: int, offset: int = 0) -> list:
    if type(num_experts) == str:
        num_experts = [int(x) for x in num_experts.split()]
    assert len(num_experts) == 1 or len(num_experts) == num_layers // expert_interval, \
        'num_experts must be either a single value or a list of the same length as the number of MoE layers'
    if len(num_experts) == 1:
        num_experts = num_layers // expert_interval
    experts_per_layer = []
    for i in range(num_layers):
        layer_num = i + 1 + offset
        n_e = num_experts[(layer_num-1) // expert_interval] if layer_num % expert_interval == 0 else 1
        experts_per_layer.append(n_e)
    return experts_per_layer

def get_moe_layer_instance(d, d_ff, num_layers, num_experts, min_capacity, expert_interval):
    experts_per_layer = get_num_experts_per_layer(
        num_experts, num_layers, expert_interval
    )
    is_moe_model = any(n_experts > 1 for n_experts in experts_per_layer)
    print(f"{is_moe_model=}")

    moe_layers = nn.ModuleList()
    from mlp import MLP
    for index, num_experts in enumerate(experts_per_layer):
        moe_layers.append(
            MoE(
                d,
                expert = MLP(d, d_ff),
                num_experts = num_experts,
                min_capacity = min_capacity
            )
        )

    return moe_layers

if __name__ == "__main__":
    print(get_moe_layer_instance())