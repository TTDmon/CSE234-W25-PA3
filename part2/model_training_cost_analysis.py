import argparse
import json
import math

def model_training_cost_analysis_llama(model_config_path):
# Model size
# 6.74B params
# 6738415616
    # Model structure illustration:
    # Transformer-based Language Model
    #
    # Input Tokens
    #      |
    # Embedding Layer (embedding_dim × vocab_size)
    #      |
    # Transformer Layers (num_layers):
    # ┌───────────────────────────────────┐
    # │ Attention Layers                  │
    # │ └── (4 × embedding_dim²)          │
    # │ Feed Forward Network (FFN) Layers │
    # │ └── (3 × embedding_dim × ffn_dim) │
    # │ RMSNorm Layers                    │
    # │ └── (2 × embedding_dim)           │
    # └───────────────────────────────────┘
    #      |
    # Final RMSNorm Layer (embedding_dim)
    #      |
    # Language Modeling Head (embedding_dim × vocab_size)
    #      |
    # Output Tokens (probabilities)

    with open(model_config_path, 'r') as f:
            config = json.load(f)
    embedding_dim = config["hidden_size"]            # Typically 4096
    num_transformer_layers = config["num_hidden_layers"]  # Typically 32
    ffn_hidden_dim = config["intermediate_size"]     # Typically 11008
    vocabulary_size = config["vocab_size"]           # Typically 32000

    # Calculate Embedding Layer Parameters
    embedding_layer_params = embedding_dim * vocabulary_size

    # Calculate parameters per transformer layer
    attention_layer_params = 4 * (embedding_dim ** 2)
    ffn_layer_params = 3 * embedding_dim * ffn_hidden_dim
    rmsnorm_layer_params = 2 * embedding_dim

    # Sum parameters for a single transformer layer
    single_transformer_layer_params = (
        attention_layer_params + ffn_layer_params + rmsnorm_layer_params
    )

    # Parameters for all transformer layers
    total_transformer_layers_params = single_transformer_layer_params * num_transformer_layers

    # Final RMSNorm parameters
    final_rmsnorm_params = embedding_dim

    # Language Modeling Head parameters
    lm_head_params = embedding_dim * vocabulary_size

    # Calculate total parameters of the model
    total_params = (
        embedding_layer_params +
        total_transformer_layers_params +
        final_rmsnorm_params +
        lm_head_params
    )


# In recitation yesterday TA mentioned we can use batch size of 1 and seq_len of maximum sequence length(2048) in the config file.

# https://catalog.ngc.nvidia.com/orgs/nvidia/teams/dgxc-benchmarking/resources/llama31-405b-dgxc-benchmarking-a#notes
#     model flops = (sequence length) * ((attention flops) + (mlp flops) + (embedding flops))

# model flops breakdown:
#     attention flops = 12 * (number of layers) * (hidden size)^2 * (1 + (number of query groups)/(number of attention heads) + (sequence length)/(hidden size))
#     mlp flops = 18 * (number of layers) * (FFN size) * (hidden size)
#     embedding flops = 6 * (vocab size) * (hidden size)

# Llama 3.1 405b calculation:
#     sequence length = 8192
#     attention flops = 12 * 126 * 16384^2 * (1 + 16/128 + 8192/16384) = 659,545,915,392
#     mlp flops = 18 * 126 * 53248 * 16384 = 1,978,637,746,176
#     embedding flops = 6 * 128256 * 16384 = 12,608,077,824

#     model flops = 8129 * (659,545,915,392 + 1,978,637,746,176 + 12,608,077,824) = 2.17E16
    

    # Model parameters
    hidden_size = config["hidden_size"]
    intermediate_size = config["intermediate_size"]
    num_attention_heads = config["num_attention_heads"]
    seq_len = config["max_sequence_length"]

    head_dim = hidden_size // num_attention_heads

    # Attention FLOPs
    flops_qkv = 3 * (hidden_size * hidden_size) * seq_len
    flops_attention_scores = 2 * (seq_len * seq_len) * hidden_size
    flops_attention_proj = hidden_size * hidden_size * seq_len
    total_attention_flops = flops_qkv + flops_attention_scores + flops_attention_proj

    # Feed-forward FLOPs
    flops_ffn_gate = hidden_size * intermediate_size * seq_len
    flops_ffn_up = hidden_size * intermediate_size * seq_len
    flops_ffn_down = intermediate_size * hidden_size * seq_len
    total_ffn_flops = flops_ffn_gate + flops_ffn_up + flops_ffn_down

    # Total FLOPs per layer
    total_flops_per_layer = total_attention_flops + total_ffn_flops

    # Convert FLOPs to TFLOPs (1 TFLOP = 1e12 FLOPs)
    flops_layer_TF = total_flops_per_layer / 1e12

    peak_memory_GB = 0



    

    return total_params, flops_layer_TF, peak_memory_GB

def model_training_cost_analysis_deepseek(model_config_path):
    #TODO you code here.
    
    total_params, flops_layer_TF, peak_memory_GB = 0,0,0
    return total_params, flops_layer_TF, peak_memory_GB

def get_optimal_N_D_from_cost(cost_budget):
    """
    cost_budget:  a monetary training budget (in dollars)
    Returns:
        N: Optimal total model parameters (in absolute numbers)
        D: Optimal number of training tokens (in absolute numbers)
        training_budget_flops: Effective total training FLOPs (in FLOPs)
        best_gpu: name of the selected GPU (one of 'A100', 'V100', 'T4')
    """
    #TODO you code here

    return N, D, training_budget_flops, best_gpu


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Model training cost analysis')
    parser.add_argument('--model_config', type=str, help='Path to model config file')
    parser.add_argument('--training_budget', type=float, default=None, help='Training budget')
    args = parser.parse_args()

    if args.model_config:
        if 'deepseek' in args.model_config:
            num_parameters, num_flops, memory_cost = model_training_cost_analysis_deepseek(args.model_config)
        elif 'llama' in args.model_config:
            num_parameters, num_flops, memory_cost = model_training_cost_analysis_llama(args.model_config)
        else:
            print('Unknown LLM Type!')
            exit()
        print(f"Number of parameters: {num_parameters}")
        print(f"Number of TFLOPs: {num_flops}")
        print(f"Peak memory cost: {memory_cost} GBs")

    if args.training_budget:    
        N, D, training_budget_flops, best_gpu = get_optimal_N_D_from_cost(args.training_budget)
        print(f"best_gpu: {best_gpu}")
        print(f"training_budget_flops: {training_budget_flops}")
        print(f"Optimal N: {N}")
        print(f"Optimal D: {D}")

    