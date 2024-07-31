# LLM_Format_Enforcement

Restrict LLM output to proper JSON formatting.

Currently restricted to Huggingface Transformers pipeline.

Wrapper for `noamgat/lm-format-enforcer` designed for function calling.

## Installation

For installation please use 
```ps
pip install torch --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

## Planned Features

- Support for Langchain, CTransformers, and llama.cpp
- Dynamic JSON Schema generation using text-based prompting
- Validation of JSON Schema

## Future work

- Screenshots of working functionality
- TinyLlama finetuning pipeline using JSON schema 
- Generation of artificial datasets using online LLM's (such as ChatGPT, Mistral, Copilot, etc.)
