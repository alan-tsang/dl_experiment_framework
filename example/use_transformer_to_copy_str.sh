torchrun --nproc_per_node 1 --master_port 12345\
    ./use_transformer_to_copy_str.py\
    your_want_add_arg1='{key: [1, 2, {subkey: "value"}]}'\
    your_want_add_arg2=value2\
    your_want_add_arg3=value3\
