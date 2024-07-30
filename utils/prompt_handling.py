"""Template selector and description builder"""
# import description_api

def desc_template(model_name):
    
    ## Mistral
    if model_name == "mistralai/Mistral-7B-Instruct-v0.2":
        template = [
        {"role": "user", "content": """{schema} <s>[INST] You are a cypher graph problem solver, you build strategy to solve programming problem. [/INST] \n [INST] Using the given full graph schema previously, write three single line clear instruction or steps to solve the question. [/INST] \n [INST] Do not write code. [/INST] \n [INST]Do not add complexities. [/INST] \n [INST] Make the steps as simple as possible, this will be used as pseudo code. [/INST]</s> \n\n [INST] Question: {usr} [/INST]"""}
        ]
    
    ## LLama3
    elif model_name == "meta-llama/Meta-Llama-3-8B-Instruct":
        template = [
            {"role": "system","content": """You are a cypher graph problem solver, you build strategy to solve programming problem. Using the given full graph schema write three single line clear instruction or steps to solve the question. Do not write code. Do not add complexities. \n{schema}"""},
            {"role": "user", "content": """Question: {usr}"""}
        ]
    
    else:
        return "Either model name is incorrect or it is not supported."
    
    return template

def build_description(input, pipeline):
    
    ## When pipeline is given
    if pipeline:
        # Generate response using the pipeline
        response = pipeline(input)[0]["generated_text"]

        if response[:11].lower() == "assistant\n\n":
            return response[11:]
    
    ## In case the description generator is through API call
    # response = description_api(input)[0]["generated_text"]
    # if response[:11].lower() == "assistant\n\n":
    #     return response[11:]
    
def build_cypher_inst_prompt(Prompt, Schema, Instruction, Question):
    return '''
You are an AI Cypher programming assistant, utilizing the DeepSeek Coder model. You only return code. Do not provide explanation along with it. Write cypher queries taking help from the graph schema and strategy given below. \n {schema}. \n {strategy}.
### Instruction:
{prompt}.
{question}.
### Response:
'''.format(prompt = Prompt,schema = Schema, strategy = Instruction ,question = Question).lstrip()

if __name__ == "__main__":
    desc_template("mistralai/Mistral-7B-Instruct-v0.2")