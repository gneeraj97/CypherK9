import time
import os
import openai
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


if "AUGMENT_MODEL" not in os.environ:
    # Other models:
    #   "mistralai/Mistral-7B-Instruct-v0.2"
    #   "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"
    os.environ['AUGMENT_MODEL'] = "mistralai/Mistral-7B-Instruct-v0.2"
    print(f"[!] WARNING 'AUGMENT_MODEL' env varable not set, using the default of: {os.environ['AUGMENT_MODEL']}")


if "LITELLM_API_KEY" not in os.environ or "LITELLM_BASE_URL" not in os.environ:
    raise Exception("Both 'LITELLM_API_KEY' and 'LITELLM_BASE_URL' environment variables must be set!")

litellm_client = openai.OpenAI(
    api_key=os.environ["LITELLM_API_KEY"],
    base_url=os.environ["LITELLM_BASE_URL"]
)


##################################################################
#
# Prompt generation helpers
#
##################################################################

def build_augmentation_prompt(schema, query):
    """Builds the augmentation prompt for the stock LLM."""
    if "Mistral-7B-Instruct" in os.environ["AUGMENT_MODEL"]:
        return f"""<s> {schema} [INST] You are a Cypher graph problem solver, you build strategies to plan Cypher query generation. Using the given full graph schema previously, write three single line clear instructions or steps to solve the question. Do not write code. Do not add complexities. Make the steps as simple as possible, this will be used as pseudo code. Question: {query} [/INST]"""
    else:
        return "ERROR IN AUGMENTATION PROMPT GENERATION"


def build_cypher_inst_prompt(prompt, schema, strategy, query):
    """Builds the Cypher instruction prompt for the tuned Deepseek Coder LLM."""
    return f'''
You are an AI Cypher programming assistant, utilizing the DeepSeek Coder model. You only return code. Do not provide explanation along with it. Write cypher queries taking help from the graph schema and strategy given below. \n {schema}. \n {strategy}.
### Instruction:
{prompt}.
{query}.
### Response:
'''


app = FastAPI()

class QueryRequest(BaseModel):
    query: str

# get the stored schema
with open("bh_schema.txt", 'r') as f:
    schema = f.read()


def llm_augmentation(user_query, schema):
    """
    Take a user query and the stored schema, and call the augmentation 
    LLM to generate a strategy for generating the Cypher query.
    """

    # build the prompt based on the query and passed schema
    prompt = build_augmentation_prompt(schema, user_query)

    # call the LLM to get the augmentation
    litellm_model = os.environ["AUGMENT_MODEL"].split("/")[1]
    start = time.time()
    response = litellm_client.chat.completions.create(
        model=litellm_model,
        messages = [
            {
                "role": "user",
                "content": prompt
            }
        ],
    )
    print(f"[*] Augmentation took: {(time.time()-start):.2f} seconds")
    return response.choices[0].message.content
    

def generate_cypher(user_query, schema, strategy):
    '''Generate cypher query and some post processsing'''
    
    # build the cypher generation prompt based on the passed schema, strategy,and question
    cypher_prompt = "Convert the following question into a Cypher query using the provided graph schema!"
    input = build_cypher_inst_prompt(cypher_prompt, schema, strategy, user_query)

    # call the fine-tuned Deepseek Cypher LLM to get the final Cypher query
    start = time.time()
    response = litellm_client.chat.completions.create(
        model="deepseek-cypher",
        messages = [
            {
                "role": "user",
                "content": input
            }
        ],
        # temperature=0.1
    )
    print(f"[*] Generation took: {(time.time()-start):.2f} seconds")
    return response.choices[0].message.content


@app.post("/generate_cypher/")
async def generate_cypher_query(query_request: QueryRequest):
    try:
        print(f"\n[*] Input query: '{query_request.query}'")
        strategy = llm_augmentation(query_request.query, schema)
        print(f"[*] Augmentation strategy:\n\n{strategy}\n\n")
        cypher = generate_cypher(query_request.query, schema, strategy).strip()
        print(f"[*] Final Cypher: \n\n{cypher}")
        return {"generation_strategy": strategy, "cypher_query": cypher}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)