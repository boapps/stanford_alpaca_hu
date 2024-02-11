"""
batch_selfinstruct_generate.py

run:
python -m generate_instruction generate_instruction_following_data \
  --output_dir ./ \
  --num_instructions_to_generate 10 \
  --model_name="text-davinci-003" \
"""
import time
import json
import os
import random
import re
import string
from functools import partial
from multiprocessing import Pool

import numpy as np
import openai
import tqdm
import textdistance
import utils
from nltk.stem import SnowballStemmer

import fire
from transformers import BertForSequenceClassification, BertTokenizer
import torch
import torch.nn.functional as F


model = BertForSequenceClassification.from_pretrained('./backups/merged_model').to('cuda')
tokenizer = BertTokenizer.from_pretrained('SZTAKI-HLT/hubert-base-cc')

stemmer = SnowballStemmer("hungarian")


def encode_prompt(prompt_instructions):
    """Encode multiple prompt instructions into a single string."""
    prompt = ''

    for idx, task_dict in enumerate(prompt_instructions):
        (instruction, input, output) = task_dict["instruction"], task_dict["input"], task_dict["output"]
        instruction = re.sub(r"\s+", " ", instruction).strip().rstrip(":")
        input = "<üres>" if input.lower() == "" else input
        if idx != 0:
            prompt += f"###\n"
        prompt += f"{idx + 1}. Utasítás: {instruction}\n"
        prompt += f"{idx + 1}. Bemenet:\n{input}\n"
        prompt += f"{idx + 1}. Válasz:\n{output}\n"
    return prompt


def post_process_gpt3_response(num_prompt_instructions, response):
    if response is None:
        return []
    print(response.message.content)
    raw_instructions = response.message.content
    raw_instructions = re.split("###", raw_instructions)
    instructions = []
    for idx, inst in enumerate(raw_instructions):
        # if the decoding stops due to length, the last example is likely truncated so we discard it
        if idx == len(raw_instructions) - 1 and response.finish_reason == "length":
            continue
        splitted_data = re.split(f"\d+\.\s+(Utasítás|Bemenet|Válasz):", inst)
        print(splitted_data)
        if len(splitted_data) != 7:
            continue
        else:
            inst = splitted_data[2].strip()
            input = splitted_data[4].strip()
            input = "" if input.lower() == "<üres>" else input
            output = splitted_data[6].strip()
        # filter out too short or too long instructions
        if len(inst.split()) <= 3 or len(inst.split()) > 150:
            continue
        # filter based on keywords that are not suitable for language models.
        if 'Kontextus: ' in inst or 'kedvenc' in inst or 'Kontextus: ' in input:
            continue
        # filter those starting with punctuation
        if inst[0] in string.punctuation:
            continue
        instructions.append({"instruction": inst, "input": input, "output": output})
    return instructions


def find_word_in_string(w, s):
    return stemmer.stem(w) == stemmer.stem(s)

def generate_instruction_following_data(
    output_dir="./",
    seed_tasks_path="./seed_tasks.jsonl",
    num_instructions_to_generate=15000,
    model_name="gpt-3.5-turbo-0125",
    num_prompt_instructions=3,
    request_batch_size=1,
    temperature=1.0,
    top_p=1.0,
    num_cpus=1,
):
    seed_tasks = [json.loads(l) for l in open(seed_tasks_path, "r")]
    seed_instruction_data = [
        {"instruction": t["instruction"], "input": t["input"], "output": t["output"]}
        for t in seed_tasks
    ]
    print(f"Loaded {len(seed_instruction_data)} human-written seed instructions")

    os.makedirs(output_dir, exist_ok=True)
    request_idx = 0
    # load the LM-generated instructions
    machine_instruction_data = []
    if os.path.exists(os.path.join(output_dir, "regen.json")):
        machine_instruction_data = utils.jload(os.path.join(output_dir, "regen.json"))
        print(f"Loaded {len(machine_instruction_data)} machine-generated instructions")

    # similarities = {}

    # now let's generate new instructions!
    progress_bar = tqdm.tqdm(total=num_instructions_to_generate)
    if machine_instruction_data:
        progress_bar.update(len(machine_instruction_data))

    # first we tokenize all the seed instructions and generated machine instructions
    all_instructions = [d["instruction"] for d in seed_instruction_data] + [
        d["instruction"] for d in machine_instruction_data
    ]

    while len(machine_instruction_data) < num_instructions_to_generate:
        request_idx += 1

        batch_inputs = []
        for _ in range(request_batch_size):
            # only sampling from the seed tasks
            prompt_instructions = random.sample(seed_instruction_data, num_prompt_instructions)
            prompt = encode_prompt(prompt_instructions)
            batch_inputs.append(prompt)
        request_start = time.time()
        results = utils.openai_completion(
            prompts=batch_inputs,
            model_name=model_name,
            batch_size=request_batch_size,
            logit_bias={"50256": -100},  # prevent the <|endoftext|> token from being generated
        )
        request_duration = time.time() - request_start

        process_start = time.time()
        instruction_data = []
        for result in results:
            new_instructions = post_process_gpt3_response(num_prompt_instructions, result)
            instruction_data += new_instructions

        total = len(instruction_data)
        keep = 0
        for instruction_data_entry in instruction_data:
            # computing similarity with the pre-tokenzied instructions
            similarities = [textdistance.levenshtein.normalized_similarity(inst, instruction_data_entry["instruction"]) for inst in all_instructions]
            most_similar_instructions = {
                all_instructions[i]: similarities[i] for i in np.argsort(similarities)[-10:][::-1]
            }
            if max(similarities) > 0.7:
                continue
            else:
                keep += 1
            instruction_data_entry["most_similar_instructions"] = most_similar_instructions
            instruction_data_entry["avg_similarity_score"] = float(np.mean(similarities))
            machine_instruction_data.append(instruction_data_entry)
            all_instructions.append(instruction_data_entry["instruction"])
            progress_bar.update(1)
        process_duration = time.time() - process_start
        print(f"Request {request_idx} took {request_duration:.2f}s, processing took {process_duration:.2f}s")
        print(f"Generated {total} instructions, kept {keep} instructions")
        utils.jdump(machine_instruction_data, os.path.join(output_dir, "regen.json"))


def main(task, **kwargs):
    globals()[task](**kwargs)


if __name__ == "__main__":
    fire.Fire(main)
