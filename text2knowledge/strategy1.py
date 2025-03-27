import re
import os
import json
import contextlib
# from strictjson import strict_json
# import text2knowledge.ollama.client as client
import text2knowledge.openai.client as client
from text2knowledge.prompt_template import ENTITY_EXTRACTION_PROMPT_TEMPLATE, RELATION_EXTRACTION_PROMPT_TEMPLATE


def extract_concepts(prompt: str, metadata={}, model="mistral-openorca:latest", base_url=None):
    response, _ = client.generate(model_name=model, system=ENTITY_EXTRACTION_PROMPT_TEMPLATE, prompt=prompt, base_url=base_url)
    # prompt = f"{ENTITY_EXTRACTION_PROMPT_TEMPLATE}\n\n{prompt}"
    # response, _ = client.generate(model_name=model, prompt=prompt, options={
    #     "temperature": 0.6,
    # })
    try:
        result = json.loads(response)
        result = [dict(item, **metadata) for item in result]
    except:
        print("\n\nERROR ### Here is the buggy response: ", response, "\n\n")
        print("If you get `404 Client Error: Not Found`, please check if the model name is correct or you have installed the model. If you get `500 Server Error: Internal Server Error`, please check if the model is running.")
        result = None
    return result


def graph_prompt(input: str, metadata={}, model="mistral-openorca:latest", base_url=None):
    if model == None:
        model = "mistral-openorca:latest"

    # model_info = client.show(model_name=model)
    # print( chalk.blue(model_info))

    # USER_PROMPT = f"context: ```{input}``` \n\n output: "
    # response, _ = client.generate(model_name=model, system=RELATION_EXTRACTION_PROMPT_TEMPLATE, prompt=USER_PROMPT)
    
    USER_PROMPT = f"{RELATION_EXTRACTION_PROMPT_TEMPLATE}\n\ncontext: ```{input}``` \n\n output: "
    # with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
    response, _ = client.generate(model_name=model, prompt=USER_PROMPT, base_url=base_url)
    response = response[response.index('['):response.rindex(']')+1]
    # print(response)

    try:
        
        # print(response)
        result = json.loads(response)
        result = [dict(item, **metadata) for item in result]
    except:
        print("\n\nERROR ### Here is the buggy response: ", response, "\n\n")
        result = None
    return result