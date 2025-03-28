import argparse
import json
import math
from scipy.optimize import minimize_scalar
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
    # 0.898050818048
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
    
    # 2 b s h / Giga
    # Calculate peak memory cost (in GB) assuming fp16 precision
    bytes_per_param = 2  # fp16 = 2 bytes
    activation_memory_attention = b * s * h * bytes_per_param

    # Peak memory includes model weights + activations
    peak_memory_bytes = single_transformer_layer_params * bytes_per_param + activation_memory_attention 
    peak_memory_GB = peak_memory_bytes / (1024**3)  # Convert bytes to GB

    return total_params, flops_layer_TF, peak_memory_GB

def model_training_cost_analysis_deepseek(model_config_path):
    with open(model_config_path, 'r') as f:
        config = json.load(f)

    # Extract necessary parameters from the config
    hidden_size = config["hidden_size"]
    intermediate_size = config["intermediate_size"]
    num_hidden_layers = config["num_hidden_layers"]
    vocab_size = config["vocab_size"]
    num_attention_heads = config["num_attention_heads"]
    moe_layer_freq = config["moe_layer_freq"]
    n_routed_experts = config["n_routed_experts"]
    moe_intermediate_size = config["moe_intermediate_size"]
    num_experts_per_tok = config["num_experts_per_tok"]

    # Calculate params per layer (non-MoE layers)
    attn_params = 3 * hidden_size * hidden_size + hidden_size * hidden_size
    mlp_params = 2 * hidden_size * intermediate_size
    layer_norm_params = 2 * hidden_size

    params_per_standard_layer = attn_params + mlp_params + layer_norm_params

    # Calculate params for MoE layers
    moe_params_per_expert = 2 * hidden_size * moe_intermediate_size
    total_moe_params_per_layer = moe_params_per_expert * n_routed_experts
    
    num_moe_layers = num_hidden_layers // moe_layer_freq
    num_standard_layers = num_hidden_layers - num_moe_layers

    total_params = (
        num_standard_layers * params_per_standard_layer +
        num_moe_layers * (attn_params + total_moe_params_per_layer + layer_norm_params)
    )

    # Adding Embeddings parameters
    embedding_params = vocab_size * hidden_size
    total_params += embedding_params

    # FLOPs estimation (Forward pass only)
    flops_layer = 2 * total_params
    flops_layer_TF = flops_layer / 1e12  # Convert to TeraFLOPs

    # Peak Memory estimation (simplified)
    peak_memory = total_params * 2  # bytes for bf16 (2 bytes per param)
    peak_memory_GB = peak_memory / (1024**3)  # Convert bytes to GB

    return total_params, flops_layer_TF, peak_memory_GB

def get_optimal_N_D_from_cost(cost_budget):
    """
    cost_budget: a monetary training budget (in dollars)
    Returns:
        N: Optimal total model parameters (in absolute numbers)
        D: Optimal number of training tokens (in absolute numbers)
        training_budget_flops: Effective total training FLOPs (in FLOPs)
        best_gpu: name of the selected GPU (one of 'A100', 'V100', 'T4')
    """
    # GPU configuration
    config_gpus = {
        'A100': {'price': 4.0, 'TFLOPs': 312},
        'V100': {'price': 2.5, 'TFLOPs': 125},
        'T4': {'price': 1.0, 'TFLOPs': 65},
    }

    TFLOPs_per_dollar = {gpu: cfg['TFLOPs'] / cfg['price'] for gpu, cfg in config_gpus.items()}
    best_gpu = max(TFLOPs_per_dollar, key=TFLOPs_per_dollar.get)

    training_budget_flops = (
        cost_budget / config_gpus[best_gpu]['price'] * 3600
        * config_gpus[best_gpu]['TFLOPs'] * 0.4 * 1e12
    )

    # Objective function for optimization
    def objective(N):
        D = training_budget_flops / (6 * N)
        return (406.4 / (N ** 0.34)) + (410.7 / (D ** 0.29)) + 1.69

    # Perform optimization using scalar minimization
    result = minimize_scalar(objective, bounds=(1e5, 1e15))

    # Calculate optimal N and D
    N_optimal = int(result.x)
    D_optimal = int(training_budget_flops / (6 * N_optimal))

    return N_optimal, D_optimal, training_budget_flops, best_gpu

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
        elif 'my_model_config' in args.model_config:
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

    