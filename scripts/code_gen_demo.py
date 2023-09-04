from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer
)
from argparse import ArgumentParser
import typing
from typing import Optional

def get_args():
    arg_parser = ArgumentParser(description = "Generate text with model.")
    arg_parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Path to directory with model and pretrained tokenizer."
    )
    arg_parser.add_argument(
        "--tokenizer_path",
        type=str,
        default=None,
        help="Path to pretrained tokenizer, if different from model."
    )
    arg_parser.add_argument(
        "--text",
        type=str,
        default=None,
        help="Optionally pass text as flag."
    )
    return arg_parser.parse_args()

def generate(model: AutoModelForCausalLM, tokenizer: AutoTokenizer, text: str):
    input_ids = tokenizer(text, return_tensors="pt").input_ids
    generated_ids = model.generate(input_ids, max_length=128)
    return tokenizer.decode(generated_ids[0], skip_special_tokens=True)

def run(model_path: str, tokenizer_path: Optional[str]=None, text: Optional[str]=None):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(tokenizer_path if tokenizer_path is not None else model_path)
    if text is not None:
        print(generate(model, tokenizer, text))
    else:
        print("CODE GENERATION DEMO")
        print(f"Generating code with model: {model_path}")
        print("Enter a prompt to get started (e or exit to quit)")
        print()
        
        text = input("Prompt: ")
        while text.lower() not in ['e', 'exit']: 
            generated_code = generate(model, tokenizer, text)
            print(generated_code)
            print()
            text = input("Prompt: ")

        print("Exiting code generator")

if __name__=='__main__':
    args = get_args()
    run(args.model_path, args.tokenizer_path, args.text)
