import torch
from accelerate import init_empty_weights, infer_auto_device_map, load_checkpoint_and_dispatch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
import os
import argparse

from save_weights import save_weights

def main(fp16:bool=False,bf16:bool=False):

    model_name = "EleutherAI/gpt-neox-20b"
    weights_path = "./gptneox"
    if not os.path.exists(weights_path):
        os.makedirs(weights_path)
        save_weights(fp16=fp16,bf16=bf16)

    config = AutoConfig.from_pretrained(model_name)

    config.use_cache = False

    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(config)

    tokenizer = AutoTokenizer.from_pretrained(model_name)


    if torch.cuda.is_bf16_supported():
        load_checkpoint_and_dispatch(
            model,
            weights_path,
            device_map="auto",
            no_split_module_classes=["GPTNeoXLayer"]
            offload_folder=None,
            offload_state_dict=False,
            dtype="bfloat16"
        )
    else:
        load_checkpoint_and_dispatch(
            model,
            weights_path,
            device_map="auto",
            no_split_module_classes=["GPTNeoXLayer"]
            offload_folder=None,
            offload_state_dict=False,
            dtype="float16"
        )



    prompt = 'Machine learning is '
    input_tokenized = tokenizer(prompt, return_tensors="pt")
    output = model.generate(input_tokenized["input_ids"].to(0), do_sample=True,max_length=100,temperature=0.9,top_k=50,top_p=0.9)
    output_text = tokenizer.decode(output[0].tolist())
    print(output_text)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fp16", default=False,action="store_true")
    parser.add_argument("--bf16", default=False,action="store_true")
    args = parser.parse_args()
    if args.fp16:
        main(fp16=True)
    elif args.bf16:
        main(bf16=True)
    else:
        main()


