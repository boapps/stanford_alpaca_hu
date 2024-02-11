import dataclasses
from email import message
import logging
import math
import os
import io
from pyexpat.errors import messages
import sys
import time
import json
from typing import Optional, Sequence, Union

import openai
import tqdm
import copy
from openai import OpenAI

openai_org = os.getenv("OPENAI_ORG")
if openai_org is not None:
    openai.organization = openai_org
    logging.warning(f"Switching to organization: {openai_org} for OAI API key.")


@dataclasses.dataclass
class OpenAIDecodingArguments(object):
    max_tokens: int = 1800
    temperature: float = 0.2
    top_p: float = 1.0
    n: int = 1
    stream: bool = False
    stop: Optional[Sequence[str]] = None
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    suffix: Optional[str] = None
    logprobs: Optional[int] = None
    echo: bool = False

client = OpenAI()

def openai_completion(
    prompts: Union[str, Sequence[str], Sequence[dict[str, str]], dict[str, str]],
    model_name="gpt-3.5-turbo",
    sleep_time=2,
    batch_size=1,
    max_instances=sys.maxsize,
    max_batches=sys.maxsize,
    return_text=False,
    **decoding_kwargs,
):
    """Decode with OpenAI API.

    Args:
        prompts: A string or a list of strings to complete. If it is a chat model the strings should be formatted
            as explained here: https://github.com/openai/openai-python/blob/main/chatml.md. If it is a chat model
            it can also be a dictionary (or list thereof) as explained here:
            https://github.com/openai/openai-cookbook/blob/main/examples/How_to_format_inputs_to_ChatGPT_models.ipynb
        decoding_args: Decoding arguments.
        model_name: Model name. Can be either in the format of "org/model" or just "model".
        sleep_time: Time to sleep once the rate-limit is hit.
        batch_size: Number of prompts to send in a single request. Only for non chat model.
        max_instances: Maximum number of prompts to decode.
        max_batches: Maximum number of batches to decode. This argument will be deprecated in the future.
        return_text: If True, return text instead of full completion object (which contains things like logprob).
        decoding_kwargs: Additional decoding arguments. Pass in `best_of` and `logit_bias` if you need them.

    Returns:
        A completion or a list of completions.
        Depending on return_text, return_openai_object, and decoding_args.n, the completion type can be one of
            - a string (if return_text is True)
            - an openai_object.OpenAIObject object (if return_text is False)
            - a list of objects of the above types (if decoding_args.n > 1)
    """
    is_single_prompt = isinstance(prompts, (str, dict))
    if is_single_prompt:
        prompts = [prompts]

    if max_batches < sys.maxsize:
        logging.warning(
            "`max_batches` will be deprecated in the future, please use `max_instances` instead."
            "Setting `max_instances` to `max_batches * batch_size` for now."
        )
        max_instances = max_batches * batch_size

    prompts = prompts[:max_instances]
    num_prompts = len(prompts)
    prompt_batches = [
        prompts[batch_id * batch_size : (batch_id + 1) * batch_size]
        for batch_id in range(int(math.ceil(num_prompts / batch_size)))
    ]

    completions = []
    for batch_id, prompt_batch in enumerate(prompt_batches):
        while True:
            try:
                shared_kwargs = dict(
                    model=model_name,
                    **decoding_kwargs,
                )
                messages=[
                    {
                    "role": "system",
                    "content": "A feladatod, hogy kitalálj egy adott számú különböző feladatból álló utasításcsomagot! Ezeket a feladatutasításokat egy GPT modellnek adjuk meg, és értékelni fogjuk a GPT asszisztenst az utasítások teljesítésén. Az elválasztáshoz használj '###'-t.\n\nItt vannak a követelmények:\n1. Próbáld meg nem ugyanazt az igét használni minden utasításnál, hogy növeld a változatosságot!\n2. Az utasításokhoz használt nyelvezetnek is változatosnak kell lennie! Például használj a felszólító mód mellett kérdéseket is!\n3. Az utasítások típusának is sokfélének kell lennie. A listának változatos feladattípusokat kell tartalmaznia, például nyílt végű generálást, osztályozást, szerkesztést stb.\n4. A GPT nyelvi modellnek képesnek kell lennie az utasítás teljesítésére. Például ne kérd az asszisztenstől, hogy hozzon létre vizuális vagy hangból álló kimenetet! Egy másik példa: ne kérd meg az asszisztenst, hogy ébresszen fel délután 5 órakor, vagy állítson be emlékeztetőt, mert képes semmilyen művelet végrehajtására.\n5. Az utasítás legyen magyar nyelven!\n6. Az utasítások 1-2 mondat hosszúak legyenek! Felszólító mód és kérdés is megengedett.\n7. Bizonyos utasításhoz bemenetnek is tartoznia kell. Ennek a bemenet mezőnek tartalmaznia kell az utasításhoz megadott konkrét példát. A bemenetnek érdemi tartalmat kell nyújtania ahhoz, hogy az utasítás kihívást jelentsen, de ideális esetben nem haladhatja meg a 100 szót.\n8. Nem minden utasításhoz van szükség bemenetre. Például, ha egy utasítás valamilyen általános információra kérdez rá, hogy \"mennyi 15+93?\", nem szükséges konkrét bemenetet megadni. Ebben az esetben egyszerűen írjuk be az \"<üres>\" feliratot a bemenet mezőbe, de a bemenet mezőnek ekkor is léteznie kell!\n9. Válasznak meg kell válaszolnia az utasítást a bemenet tekintetében. Ügyelj arra, hogy a kimenet ne nagyon haladja meg a 100 szót álljon!\n10. Az utasítások és válaszok esetében is ügyelj a nyelvtanilag helyes és magyaros szövegalkotásra, valamint a megfelelő ragozásra!"
                    },
                    {
                    "role": "user",
                    "content": "Adj meg 3 darab feladatot!"
                    },
                    {
                    "role": "assistant",
                    "content": prompt_batch[0]
                    },
                    {
                    "role": "user",
                    "content": "Adj meg 20 darab feladatot ugyanilyen formátumban!"
                    }
                ]
                response = client.chat.completions.create(
                model="gpt-3.5-turbo-0125",
                messages=messages,
                temperature=1,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0,
                stop=["21. Instrukció", "\\n21."]
                )

                choices = response.choices

                completions.extend(choices)
                break
            except Exception as e:
                logging.warning(f"OpenAIError: {e}.")
                logging.warning("Hit request rate limit; retrying...")
                time.sleep(sleep_time)  # Annoying rate limit on requests.

    if return_text:
        completions = [completion.text for completion in completions]
    if is_single_prompt:
        # Return non-tuple if only 1 input and 1 generation.
        (completions,) = completions
    return completions


def _make_w_io_base(f, mode: str):
    if not isinstance(f, io.IOBase):
        f_dirname = os.path.dirname(f)
        if f_dirname != "":
            os.makedirs(f_dirname, exist_ok=True)
        f = open(f, mode=mode)
    return f


def _make_r_io_base(f, mode: str):
    if not isinstance(f, io.IOBase):
        f = open(f, mode=mode)
    return f


def jdump(obj, f, mode="w", indent=4, default=str):
    """Dump a str or dictionary to a file in json format.

    Args:
        obj: An object to be written.
        f: A string path to the location on disk.
        mode: Mode for opening the file.
        indent: Indent for storing json dictionaries.
        default: A function to handle non-serializable entries; defaults to `str`.
    """
    f = _make_w_io_base(f, mode)
    if isinstance(obj, (dict, list)):
        json.dump(obj, f, indent=indent, default=default)
    elif isinstance(obj, str):
        f.write(obj)
    else:
        raise ValueError(f"Unexpected type: {type(obj)}")
    f.close()


def jload(f, mode="r"):
    """Load a .json file into a dictionary."""
    f = _make_r_io_base(f, mode)
    jdict = json.load(f)
    f.close()
    return jdict
