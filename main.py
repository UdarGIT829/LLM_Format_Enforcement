import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import pandas as pd
from lmformatenforcer import JsonSchemaParser, CharacterLevelParser, RegexParser, StringParser
from lmformatenforcer.integrations.transformers import generate_enforced, build_token_enforcer_tokenizer_data

from pydantic import BaseModel
from typing import Tuple, Optional, Union, List

import json

from utils import get_prompt, extract_json_objects, test_cuda, display_header, display_content

StringOrManyStrings = Union[str, List[str]]

def run(message: StringOrManyStrings,
        system_prompt: str,
        max_new_tokens: int = 1024,
        temperature: float = 0.8,
        top_p: float = 0.95,
        top_k: int = 50,
        num_beams: int = 1,
        required_regex: Optional[str] = None,
        required_str: Optional[str] = None,
        required_json_schema: Optional[dict] = None,
        required_json_output: Optional[bool] = None) -> Tuple[StringOrManyStrings, Optional[pd.DataFrame]]:
    is_multi_message = isinstance(message, list)
    messages = message if is_multi_message else [message]
    prompts = [get_prompt(msg, system_prompt) for msg in messages]
    inputs = tokenizer(prompts, return_tensors='pt', add_special_tokens=False, return_token_type_ids=False, padding=is_multi_message).to(device)
    
    generate_kwargs = dict(
        inputs,
        # streamer=streamer,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        top_p=top_p,
        top_k=top_k,
        temperature=temperature,
        num_beams=num_beams,
        output_scores=True,
        return_dict_in_generate=True
    )

    parser: Optional[CharacterLevelParser] = None
    if required_regex:
        parser = RegexParser(required_regex)
    if required_str:
        parser = StringParser(required_str)
    if required_json_schema:
        parser = JsonSchemaParser(required_json_schema)
    if required_json_output:
        parser = JsonSchemaParser(None)

    if parser:
        output = generate_enforced(model, tokenizer_data, parser, **generate_kwargs)
    else:
        output = model.generate(**generate_kwargs)

    sequences = output['sequences']
    # skip_prompt=True doesn't work consistenly, so we hack around it.
    string_outputs = [tokenizer.decode(sequence, skip_special_tokens=True) for sequence in sequences]
    string_outputs = [string_output.replace(prompt[3:], '') for string_output, prompt in zip(string_outputs, prompts)]
    if parser and not is_multi_message:
        enforced_scores_dict = output.enforced_scores
        enforced_scores = pd.DataFrame(enforced_scores_dict)
        pd.set_option('display.width', 1000)
        pd.set_option('display.max_columns', 10)
        pd.set_option('display.max_rows', 999)
        pd.set_option('display.float_format', ' {:,.5f}'.format)
    else:
        enforced_scores = None
    return string_outputs if is_multi_message else string_outputs[0], enforced_scores


model_id = "solidrust/Hermes-2-Pro-Llama-3-8B-AWQ"

device = 'cuda'

if test_cuda():
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map='auto')
else:
    raise Exception('GPU not available')
tokenizer = AutoTokenizer.from_pretrained(model_id)
if tokenizer.pad_token_id is None:
    # Required for batching example
    tokenizer.pad_token_id = tokenizer.eos_token_id 


tokenizer_data = build_token_enforcer_tokenizer_data(tokenizer)

DEFAULT_SYSTEM_PROMPT = """\
You are a helpful assistant to a storyteller. Describe only the following situation, Do not describe anything outside the provided information:\n\
"""
DEFAULT_MAX_NEW_TOKENS = 150

class AnswerFormat(BaseModel):
    situation: str
    

question = """{'player_cosmetic_details': ['wet'], 'player_state_details': ['cold'], 'environment': 'wilderness, at a cave entrance with a small stream flowing in and the forest in the distance with a river flowing out', 'company': 'alone', 'danger_level': 'low'}\n\n    You MUST answer using the following json schema: """

question_with_schema = f'{question}{AnswerFormat.model_json_schema()}'

display_header("Question:")
display_content(question_with_schema)

display_header("Answer, With json schema enforcing:")
result, enforced_scores = run(question_with_schema, system_prompt=DEFAULT_SYSTEM_PROMPT,max_new_tokens=DEFAULT_MAX_NEW_TOKENS, required_json_schema=AnswerFormat.model_json_schema())
display_content(result)

print("_="*9)
try:
    myJSON_output = json.loads(result.strip())
except Exception as err:
    print("Attempting JSON Extraction")
    myJSON_output = json.loads(extract_json_objects(result.strip())[-1])

print(myJSON_output)
print(enforced_scores)