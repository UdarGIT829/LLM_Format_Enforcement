def get_prompt(message: str, system_prompt: str) -> str:
    texts = [f'<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n']
    # The first user input is _not_ stripped
    do_strip = False
    message = message.strip() if do_strip else message
    texts.append(f'{message} [/INST]')
    return ''.join(texts)