import argparse
from utils import prompt_handling, schema_template
from utils.neo4jutils.neo4j_schema import *
from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig, AutoModelForCausalLM, pipeline
from unsloth import FastLanguageModel
from huggingface_hub import login
import torch
import warnings
import gc
import GPUtil
import time
warnings.filterwarnings("ignore")

IGNORE_INDEX = -100
EOT_TOKEN = "<|EOT|>"

def huggingface_login():
    login(token="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")
    
def take_user_input():
    parser = argparse.ArgumentParser(description="Get the user query and other relevant information")
    
    parser.add_argument("--query", type=str, default= "Which stream has most users?", help="Question of the user")
    parser.add_argument("--graph", type=str, default= "neo4j", help="name of the database")
    parser.add_argument("--augmentModel", type=str, default= "mistralai/Mistral-7B-Instruct-v0.2", help="LLM augment model name")
    parser.add_argument("--cypherModel", type=str, default= "specter_cypher_model", help="Cypher finetuned Model")
    parser.add_argument("--graphURL", type=str, default= "neo4j://98.80.176.211:7687", help="Cypher finetuned Model")
    parser.add_argument("--graphUser", type=str, default= "neo4j", help="Cypher finetuned Model")
    parser.add_argument("--graphPWD", type=str, default= "runouts-recognitions-acceleration", help="Cypher finetuned Model")
    parser.add_argument("--running_on_laptop", type=bool, default= True, help="To clear the cuda memory if running on laptop.")
    
    args = parser.parse_args()
    
    return args
 
def generate_graph_schema_template(graphName, graphURL, usr, pwd):
    gutils = Neo4jSchema(url = graphURL, username = usr, password=pwd, database = graphName)
    
    jschema = gutils.get_structured_schema
    schema = schema_template.generate_full_schema_template(jschema)
    
    return schema
    
def _LLM_augment(ModelName, user_query, Gschema, desc_tokenizer, pipeline):
    '''LLM augmentation generation'''
    template = prompt_handling.desc_template(ModelName)
    chat = desc_tokenizer.apply_chat_template(template, tokenize = False)
    input = chat.format(usr = user_query, schema = Gschema)
    # print(input)
    
    desc = prompt_handling.build_description(input, pipeline)
    
    return desc

def generate_cypher(user_query, schema, description, cypher_model, cypher_tokenizer):
    '''Generate cypher query and some post processsing'''
    
    cypher_prompt = "Convert the following question into a Cypher query using the provided graph schema!"
    input = prompt_handling.build_cypher_inst_prompt(cypher_prompt, schema, description, user_query)
    inputs = cypher_tokenizer(input, return_tensors="pt").to(cypher_model.device)
    
    # inputs
    outputs = cypher_model.generate(**inputs, max_new_tokens = 500)
    generated_cypher = cypher_tokenizer.decode(outputs[0], skip_special_tokens=True)[len(input):]
    
    return generated_cypher
    
def self_heal():
    return False

def clear_model_memory(model):
    del model  # Delete the model variable
    torch.cuda.empty_cache()  # Clear the GPU cache
    gc.collect()

def print_gpu_utilization():
    gpus = GPUtil.getGPUs()
    for gpu in gpus:
        print(f"GPU {gpu.id}:")
        print(f"  Memory Used: {gpu.memoryUsed} MB")
        print(f"  Memory Total: {gpu.memoryTotal} MB")
        print(f"  Memory Utilization: {gpu.memoryUtil * 100}%")
        print(f"  GPU Utilization: {gpu.load * 100}%")
    return gpus

def main():
    args = take_user_input()
    huggingface_login()
    print("User inputs read.")
    
    max_seq_length = 32000 
    dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
    load_in_4bit = True 
    
    jschema = generate_graph_schema_template(args.graph, args.graphURL, args.graphUser, args.graphPWD)
    print("Schema generated!")
    
    aug_tokenizer = AutoTokenizer.from_pretrained(args.augmentModel)
    repetition_penalty = 1.1
    
    bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
    )

    aug_tokenizer = AutoTokenizer.from_pretrained(args.augmentModel)
    aug_model = AutoModelForCausalLM.from_pretrained(args.augmentModel, 
                                                 quantization_config=bnb_config,
                                                 device_map = "auto",
                                                #  attn_implementation="flash_attention",
                                                 )

    # Create a text generation pipeline with specified parameters
    llm_aug_pipeline = pipeline(
        "text-generation",
        model=aug_model,
        tokenizer=aug_tokenizer,
        max_length=8192*2,
        repetition_penalty=repetition_penalty,
        pad_token_id=aug_tokenizer.eos_token_id,
        return_full_text = False,
        clean_up_tokenization_spaces = True, 
        batch_size = 16,
    )
    print("LLM augmentation model and Pipeline defined.")
    
    llm_augmentation = _LLM_augment(args.augmentModel, args.query, jschema, aug_tokenizer, llm_aug_pipeline)
    print("LLM Augmentation generated")
    
    print("Gpu utilization before clearing memory")
    print("="*100)
    print_gpu_utilization()
    
    if args.running_on_laptop:
        print("Code running on laptop, need to clear the cache before moving on.")
        clear_model_memory(aug_model)
        time.sleep(4)

    print("Gpu utilization after clearing memory")
    print("="*100)
    print_gpu_utilization()
    # exit()

    cypher_model, cypher_tokenizer = FastLanguageModel.from_pretrained(
        model_name = "lora_model",
        max_seq_length = max_seq_length,
        dtype = dtype,
        load_in_4bit = load_in_4bit,
    )
    
    print("Cypher finetuned model loaded.")
    
    FastLanguageModel.for_inference(cypher_model)
    
    print("Model setup for inference")
    
    cypher = generate_cypher(args.query, jschema, llm_augmentation, cypher_model, cypher_tokenizer)
    
    print("Cypher generated.")
    
    return cypher
    
    
    

if __name__ == "__main__":
    
    cypher = main()
    
    print(cypher)