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


def graph_prompt(input: str, metadata={}, model="mistral-openorca:latest", base_url=None, entity_types=[], relation_types=[]):
    if model == None:
        model = "mistral-openorca:latest"
    # model_info = client.show(model_name=model)
    # print( chalk.blue(model_info))
    # USER_PROMPT = f"context: ```{input}``` \n\n output: "
    if model == None:
        model = "mistral-openorca:latest"

    # model_info = client.show(model_name=model)
    # print( chalk.blue(model_info))

    # USER_PROMPT = f"context: ```{input}``` \n\n output: "
    # response, _ = client.generate(model_name=model, system=RELATION_EXTRACTION_PROMPT_TEMPLATE, prompt=USER_PROMPT)
    SYSTEM = RELATION_EXTRACTION_PROMPT_TEMPLATE\
        .replace("###ENTITY_TYPE###", str(entity_types))\
       .replace('###RELATION_TYPE###', str(relation_types))
    USER_PROMPT = f"{SYSTEM}\n\ncontext: ```{input}``` \n\n output: "
    # with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):

    # [{
    #     "source_name": "Congenital long QT syndrome (LQTS)",
    #     "source_type": "Disease",
    #     "target_name": "fetal bradycardia",
    #     "target_type": "Condition",
    #     "relation_type": "BioMedGPS::AssociatedWith::Disease:Symptom",
    #     "key_sentence": "In this study we investigated a newborn patient with fetal bradycardia, 2:1 atrioventricular block and ventricular tachycardia soon after birth.",
    # }, ...]


    response_format = {
        "type": "json_schema",
        "json_schema": {
            "name": "relation_schema",
            "strict": "true",
            "schema": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "source_name": {"type": "string"},
                        "source_type": {"type": "string", "enum": entity_types},
                        "target_name": {"type": "string"},
                        "target_type": {"type": "string", "enum": entity_types},
                        "relation_type": {"type": "string", "enum": relation_types},
                        "key_sentence": {"type": "string"},
                    },
                    "required": ["source_name", "source_type", "target_name", "target_type", "relation_type", "key_sentence"],
                }
            }
        }
    }


    response, _ = client.generate(model_name=model, prompt=USER_PROMPT, base_url=base_url, response_format=response_format)
    # print(response)
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