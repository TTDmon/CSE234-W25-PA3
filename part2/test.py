# couldn't run yet
# pip install --upgrade calflops
from calflops import calculate_flops_hf

batch_size, max_seq_length = 1, 2048
# model_name = "https://huggingface.co/THUDM/glm-4-9b-chat" # THUDM/glm-4-9b-chat
model_name = "https://huggingface.co/huggyllama/llama-7b"
flops, macs, params = calculate_flops_hf(model_name=model_name, input_shape=(batch_size, max_seq_length))
print("%s FLOPs:%s  MACs:%s  Params:%s \n" %(model_name, flops, macs, params))
