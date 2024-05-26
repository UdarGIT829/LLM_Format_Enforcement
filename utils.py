import re
import torch


def test_cuda():
    if result := torch.cuda.is_available():  # This should return True if a GPU is available
        print("CUDA using GPU device.")
    else:
        print("GPU not available to CUDA")
    return result

def extract_json_objects(text):
    # Regular expression to find all occurrences of {...}
    pattern = r'\{[^{}]*\}'
    results = re.findall(pattern, text)
    return results

def get_prompt(message: str, system_prompt: str) -> str:
    texts = [f'<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n']
    # The first user input is _not_ stripped
    do_strip = False
    message = message.strip() if do_strip else message
    texts.append(f'{message} [/INST]')
    return ''.join(texts)


def display_header(text):
    # Using ANSI escape codes to apply bold style in the console
    print(f'\033[1m{text}\033[0m')

def display_content(text):
    # Print the text within a simple box for clarity, mimicking the Markdown code block style
    print(f'```\n{text.strip()}\n```')
