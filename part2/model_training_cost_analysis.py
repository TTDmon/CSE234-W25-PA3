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

    # In recitation yesterday TA mentioned we can use 
    # batch size of 1 and seq_len of maximum sequence length(2048) 
    # in the config file.
    # flops per attention layer:
    # 6bsh2 + 4bs2h + 3bs2n +2bsh2 + 6bshi
    # From Page50 of
    # https://hao-ai-lab.github.io/cse234-w25/assets/slides/feb27.pdf 
    b = 1 # batch size
    s = config["max_sequence_length"] # seq_len
    h = config["hidden_size"] # hidden size
    i = config["intermediate_size"] # SwiGLU intermediate size
    n = config["num_attention_heads"] # Number of attention heads
    flops_layer_TF = (6 * b * s * h * h 
                      + 4 * b * s * s * h
                      + 3 * b * s *  s * n 
                      + 2 * b * s * h * h 
                      + 6 * b * s * h * i)/1e12

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

    