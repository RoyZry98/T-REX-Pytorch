from transformers import AutoTokenizer, AutoConfig, AddedToken
import torch
from loguru import logger
import copy
import argparse

import sys
# sys.path.append("../../")
from component.utils import ModelUtils
from component.template import template_dict
import os
from datasets import load_dataset
from tqdm import tqdm
from prompt_template import *
from torch.utils.data import DataLoader, Subset
import numpy as np
import re

# Manually building the dict based on the provided function names and the user's requirement
FUNC_POOL = {
    "boolq":prompt_one_example_on_boolq,
    "cb":prompt_one_example_on_cb,
    "multirc":prompt_one_example_on_multirc,
    "wic":prompt_one_example_on_wic,
    "wsc":prompt_one_example_on_wsc,
    "rte":prompt_one_example_on_rte,
    "copa":prompt_one_example_on_copa,
    "WinoGrande":prompt_one_example_on_WinoGrande,
    "openbookqa":prompt_one_example_on_openbookqa,
    "anli":prompt_one_example_on_anli,
    "record":prompt_one_example_on_record,
    "hellaswag":prompt_one_example_on_hellaswag,
    "piqa":prompt_one_example_on_piqa,
    "mmlu":prompt_one_example_on_mmlu,
    "arce":prompt_one_example_on_arce,
    "commonsense_qa":prompt_one_example_on_commonsense_qa,
    "siqa":prompt_one_example_on_siqa,
    "boolean_expressions":prompt_one_example_on_bbh_boolean_expressions,
    "causal_judgement":prompt_one_example_on_bbh_causal_judgement,
    "date_understanding":prompt_one_example_on_bbh_date_understanding,
    "disambiguation_qa":prompt_one_example_on_bbh_disambiguation_qa,
    "dyck_languages":prompt_one_example_on_bbh_dyck_languages,
    "formal_fallacies":prompt_one_example_on_bbh_formal_fallacies,
    "geometric_shapes":prompt_one_example_on_bbh_geometric_shapes,
    "hyperbaton":prompt_one_example_on_bbh_hyperbaton,
    "logical_deduction_five_objects":prompt_one_example_on_bbh_logical_deduction_five_objects,
    "logical_deduction_seven_objects":prompt_one_example_on_bbh_logical_deduction_seven_objects,
    "logical_deduction_three_objects":prompt_one_example_on_bbh_logical_deduction_three_objects,
    "movie_recommendation":prompt_one_example_on_bbh_movie_recommendation,
    "multistep_arithmetic_two":prompt_one_example_on_bbh_multistep_arithmetic_two,
    "navigate":prompt_one_example_on_bbh_navigate,
    "object_counting":prompt_one_example_on_bbh_object_counting,
    "penguins_in_a_table":prompt_one_example_on_bbh_penguins_in_a_table,
    "reasoning_about_colored_objects":prompt_one_example_on_bbh_reasoning_about_colored_objects,
    "ruin_names":prompt_one_example_on_bbh_ruin_names,
    "salient_translation_error_detection":prompt_one_example_on_bbh_salient_translation_error_detection,
    "snarks":prompt_one_example_on_bbh_snarks,
    "sports_understanding":prompt_one_example_on_bbh_sports_understanding,
    "temporal_sequences":prompt_one_example_on_bbh_temporal_sequences,
    "tracking_shuffled_objects_five_objects":prompt_one_example_on_bbh_tracking_shuffled_objects_five_objects,
    "tracking_shuffled_objects_seven_objects":prompt_one_example_on_bbh_tracking_shuffled_objects_seven_objects,
    "tracking_shuffled_objects_three_objects":prompt_one_example_on_bbh_tracking_shuffled_objects_three_objects,
    "web_of_lies":prompt_one_example_on_bbh_web_of_lies,
    "word_sorting":prompt_one_example_on_bbh_word_sorting,
    
}
all_resultsp = {}

def build_prompt_chatglm3(tokenizer, query, history, system=None):
    history.append({"role": 'user', 'message': query})
    # system
    input_ids = tokenizer.get_prefix_tokens() + \
                [tokenizer.get_command(f"<|system|>")] + \
                tokenizer.encode(system, add_special_tokens=False)
    # convs
    for item in history:
        role, message = item['role'], item['message']
        if role == 'user':
            tokens = [tokenizer.get_command(f"<|user|>")] + \
                     tokenizer.encode(message, add_special_tokens=False) + \
                     [tokenizer.get_command(f"<|assistant|>")]
        else:
            tokens = tokenizer.encode(message, add_special_tokens=False) + [tokenizer.eos_token_id]
        input_ids += tokens

    return input_ids


def build_prompt(tokenizer, template, query, history, system=None, verbose=False):
    template_name = template.template_name
    system_format = template.system_format
    user_format = template.user_format
    assistant_format = template.assistant_format
    system = system if system is not None else template.system

    if template_name == 'chatglm2':
        prompt = tokenizer.build_prompt(query, history)
        input_ids = tokenizer.encode(prompt)
    elif template_name == 'chatglm3':
        input_ids = build_prompt_chatglm3(tokenizer, query, history, system)
    else:
        history.append({"role": 'user', 'message': query})
        input_ids = []

        # setting system information
        if system_format is not None:
            # system信息不为空
            if system is not None:
                system_text = system_format.format(content=system)
                if verbose:
                    print(system_text)
                input_ids = tokenizer.encode(system_text, add_special_tokens=False)
        # concat conversation
        for item in history:
            role, message = item['role'], item['message']
            message = user_format.format(content=message, stop_token=tokenizer.eos_token)
            if verbose:
                print(message)
            # if role == 'user':
            #     message = user_format.format(content=message, stop_token=tokenizer.eos_token)
            # else:
            #     message = assistant_format.format(content=message, stop_token=tokenizer.eos_token)
            tokens = tokenizer.encode(message, add_special_tokens=False)
            input_ids += tokens
    input_ids = torch.tensor([input_ids], dtype=torch.long)

    return input_ids


def load_tokenizer(model_name_or_path):
    # config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
    # 加载tokenzier
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
        use_fast=False
        # llama不支持fast
        # use_fast=False if config.model_type == 'llama' else True
    )

    if 'gemma' in model_name_or_path.lower():
        tokenizer.add_special_tokens({'additional_special_tokens': ['<start_of_turn>', '<end_of_turn>']})
        print("add special tokens")

    if tokenizer.__class__.__name__ == 'QWenTokenizer':
        tokenizer.pad_token_id = tokenizer.eod_id
        tokenizer.bos_token_id = tokenizer.eod_id
        tokenizer.eos_token_id = tokenizer.eod_id
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # assert tokenizer.pad_token_id is not None, "pad_token_id should not be None"
    return tokenizer


def main():
    # 使用合并后的模型进行推理
    # Initialize the parser
    parser = argparse.ArgumentParser(description="Arguments for running the model with specific configurations.")

    # Add arguments
    parser.add_argument("--model_name_or_path", type=str, default="llama2-13b_model_path", help="Path to the model directory.")
    parser.add_argument("--template_name", type=str, default="llama2", help="Name of the template to use.")
    parser.add_argument("--adapter_name_or_path", type=str, default="./output/llama2-13b-trex", help="Path to the adapter or name of the adapter.")
    parser.add_argument("--cluster_file", type=str, default="./datasets/task_embs_32cluster_nomic-embed-text-v1.pt", help="")
    parser.add_argument("--datasets", default=None, nargs="+", metavar="DATASET", help="List of datasets")
    parser.add_argument("--moe_mode", type=str, required=True, help="\
                        lora \
                        rank1 \
                        rank1_flex \
                        none")
    parser.add_argument("--rank1_flex_topk", type=int, default=-1, help="")
    parser.add_argument("--verbose", action="store_true", help="")
    parser.add_argument("--rank1_flex_rank_allocation", nargs='+', type=int, default=None, help="lora_A's dim")

    parser.add_argument("--gen_embeds", action="store_true", help="")
    parser.add_argument("--embedder", type=str, default=None, help="")
    parser.add_argument("--router_mapping", default=False, action="store_true", help="add router mapper")
    parser.add_argument("--ablation", type=str, default=None, help="intuition , rank1")
    # Parse the arguments
    args = parser.parse_args()

    if args.gen_embeds and args.embedder=="nomic":
        from component.embedder import init_embedder, gen_embs
        embedder, emb_tokenizer = init_embedder()

    external_cfg = {
        "moe_mode": args.moe_mode,
        "rank1_flex_topk": args.rank1_flex_topk,
        "router_mapping": args.router_mapping,
        "ablation": args.ablation,
        "rank1_flex_rank_allocation": args.rank1_flex_rank_allocation,
    }
    with open("global_external_config.json","w") as fp:
        json.dump(external_cfg, fp)

    if args.moe_mode in ["rank1", "rank1_flex"]:
        centroids = torch.load(args.cluster_file, map_location='cpu').numpy()

    template_name = args.template_name
    model_name_or_path = args.model_name_or_path
    adapter_name_or_path = args.adapter_name_or_path

    template = template_dict[template_name]
    # 是否使用4bit进行推理，能够节省很多显存，但效果可能会有一定的下降
    load_in_4bit = False
    # 生成超参配置
    max_new_tokens = 1
    top_p = 0.9
    temperature = 0.35
    repetition_penalty = 1.0

    # 加载模型
    logger.info(f'Loading model from: {model_name_or_path}')
    logger.info(f'adapter_name_or_path: {adapter_name_or_path}')
    model = ModelUtils.load_model(
        model_name_or_path,
        load_in_4bit=load_in_4bit,
        adapter_name_or_path=adapter_name_or_path
    ).eval()
    if "bloom" in model_name_or_path:
        tokenizer = load_tokenizer(model_name_or_path)
    else:
        print(model_name_or_path if adapter_name_or_path is None else adapter_name_or_path)
        tokenizer = load_tokenizer(model_name_or_path if adapter_name_or_path is None else adapter_name_or_path)
        # tokenizer = load_tokenizer(model_name_or_path)
    if template_name == 'chatglm2':
        stop_token_id = tokenizer.eos_token_id
    elif template_name == 'chatglm3':
        stop_token_id = [tokenizer.eos_token_id, tokenizer.get_command("<|user|>"), tokenizer.get_command("<|observation|>")]
    else:
        if template.stop_word is None:
            template.stop_word = tokenizer.eos_token
        # import pdb;pdb.set_trace()
        stop_token_id = tokenizer.encode(template.stop_word, add_special_tokens=False)
        # assert len(stop_token_id) == 1
        stop_token_id = stop_token_id[0]
    
    if 'internlm2' in model_name_or_path.lower():
        tokenizer._added_tokens_encoder.update({'<|im_start|>': 92543})
        tokenizer._added_tokens_encoder.update({'<|im_end|>': 92542})
        tokenizer._added_tokens_decoder.update({92543: AddedToken('<|im_start|>')})
        tokenizer._added_tokens_decoder.update({92542: AddedToken('<|im_end|>')})
        tokenizer.add_special_tokens({'additional_special_tokens': ['<|im_start|>', '<|im_end|>']})
    elif 'orion' in model_name_or_path.lower():
        tokenizer.add_special_tokens({'bos_token': '<s>', 'eos_token': '</s>'})
    elif 'gemma' in model_name_or_path.lower():
        tokenizer.add_special_tokens({'additional_special_tokens': ['<start_of_turn>', '<end_of_turn>']})

    if tokenizer.__class__.__name__ == 'QWenTokenizer':
        tokenizer.pad_token_id = tokenizer.eod_id
        tokenizer.bos_token_id = tokenizer.eod_id
        tokenizer.eos_token_id = tokenizer.eod_id
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    assert tokenizer.pad_token_id is not None, "pad_token_id should not be None"
    assert tokenizer.eos_token_id is not None, "eos_token_id should not be None"

    dataset_list =  args.datasets if args.datasets is not None else [
            "wic", 
            "WinoGrande",
            "wsc", 
            "anli",
            "hellaswag",
            'piqa', 
            "siqa",
            'rte', 
            'copa',
            "openbookqa",
            "multirc", 
            "mmlu",
            "commonsense_qa",
            "boolq", 
            # 'boolean_expressions', 'causal_judgement', 'date_understanding', 'disambiguation_qa', 'dyck_languages', 'formal_fallacies', 'geometric_shapes', 'hyperbaton', 'logical_deduction_five_objects', 'logical_deduction_seven_objects', 'logical_deduction_three_objects', 'movie_recommendation', 'multistep_arithmetic_two', 'navigate', 'object_counting', 'penguins_in_a_table', 'reasoning_about_colored_objects', 'ruin_names', 'salient_translation_error_detection', 'snarks', 'sports_understanding', 'temporal_sequences', 'tracking_shuffled_objects_five_objects', 'tracking_shuffled_objects_seven_objects', 'tracking_shuffled_objects_three_objects', 'web_of_lies', 'word_sorting',
            # "cb", 
        ] 
    print(args.datasets)
    print("Validate on:", dataset_list)
    for sub_name in dataset_list:
        if sub_name in ["boolq", "cb", "multirc", "wic", "wsc", 'rte', 'copa',"record"]:
            print(f"Loading super_glue/{sub_name}/validation ...")
            dataset = load_dataset("super_glue", sub_name)
            val_dataset = dataset["validation"]
            print(val_dataset)
        elif sub_name == "WinoGrande":
            print(f"Loading {sub_name}/validation ...")
            dataset = load_dataset("winogrande", "winogrande_xl")
            val_dataset = dataset["validation"]
            print(val_dataset)
        elif sub_name=="anli":
            print(f"Loading {sub_name}/dev_r1 ...")
            dataset = load_dataset("anli")
            val_dataset = dataset["dev_r1"]
            print(val_dataset)
        elif sub_name=="hellaswag":
            print(f"Loading {sub_name}/validation ...")
            dataset = load_dataset("hellaswag")
            val_dataset = dataset["validation"]
            print(val_dataset)
        elif sub_name=="mmlu":
            print(f"Loading {sub_name}/validation ...")
            dataset = load_dataset("cais/mmlu", "all")
            val_dataset = dataset["validation"]
            print(val_dataset)
        elif sub_name=="arc-e":
            dataset = load_dataset('ai2_arc', 'ARC-Easy')
            print(f"Loading {sub_name}/validation ...")
            print(dataset)
            val_dataset = dataset["validation"]
        elif sub_name=="siqa":
            print(f"Loading {sub_name}/validation ...")
            dataset = load_dataset("social_i_qa")
            val_dataset = dataset["validation"]
            print(val_dataset)
        elif sub_name=="piqa":
            print(f"Loading {sub_name}/validation ...")
            dataset = load_dataset("piqa")
            val_dataset = dataset["validation"]
            print(val_dataset)
        elif sub_name=="openbookqa":
            print(f"Loading {sub_name}/validation ...")
            dataset = load_dataset("openbookqa", "main")
            val_dataset = dataset["validation"]
            print(val_dataset)
        elif sub_name in ['boolean_expressions', 'causal_judgement', 'date_understanding', 'disambiguation_qa', 'dyck_languages', 'formal_fallacies', 'geometric_shapes', 'hyperbaton', 'logical_deduction_five_objects', 'logical_deduction_seven_objects', 'logical_deduction_three_objects', 'movie_recommendation', 'multistep_arithmetic_two', 'navigate', 'object_counting', 'penguins_in_a_table', 'reasoning_about_colored_objects', 'ruin_names', 'salient_translation_error_detection', 'snarks', 'sports_understanding', 'temporal_sequences', 'tracking_shuffled_objects_five_objects', 'tracking_shuffled_objects_seven_objects', 'tracking_shuffled_objects_three_objects', 'web_of_lies', 'word_sorting']:
            print(f"Loading {sub_name}/test ...")
            dataset = load_dataset("lukaemon/bbh", sub_name)
            val_dataset = dataset["test"]
        elif  sub_name == "imdb":
            print(f"Loading {sub_name}/test ...")
            val_dataset = load_dataset(sub_name)
            val_dataset = val_dataset["test"]
            val_dataset = sorted(val_dataset, key=lambda x: len(x["text"]), reverse=True)
        else:
            dataset = load_dataset(sub_name)
            print(f"Loading {sub_name}/validation ...")
            val_dataset = dataset["validation"]
        correct = 0
        total = 0

        # load dataset
        total_size = len(val_dataset)
        subset_dataset = Subset(val_dataset,list(range(total_size)))
        # Create the dataloader for the current split
        validation_dataloader = DataLoader(subset_dataset, shuffle=False, batch_size=1, 
                                        #    sampler=sampler
                                           )
        pbar = tqdm(total=len(validation_dataloader), desc=sub_name, unit="batch")
        for batch_index, batch in enumerate(validation_dataloader):
            prompt, gt, idx = FUNC_POOL[sub_name](batch, raw=True)
            if args.moe_mode in ["lora", "sira", "none"]:
                pass
            elif args.moe_mode in ["rank1", "rank1_flex"]:
                npy_path = args.cluster_file.replace(args.cluster_file.split("/")[-1], f"npys_val_{sub_name}/{batch_index:09d}.npy")
                if args.gen_embeds and (args.embedder is not None) and not os.path.exists(npy_path):
                    intuitions = gen_embs(emb_tokenizer, embedder, [prompt])
                    # import ipdb;ipdb.set_trace()
                    relate_score = (torch.from_numpy(centroids).to(model.device)@intuitions).view(1,-1)
                else:
                    intuitions = np.load(args.cluster_file.replace(args.cluster_file.split("/")[-1], f"npys_val_{sub_name}/{batch_index:09d}.npy"))
                    relate_score = torch.from_numpy(centroids@intuitions).view(1,-1).to(model.device)
                for name, module in model.named_modules():
                    if hasattr(module, "relate_score_placeholder"):
                        setattr(module, "relate_score_placeholder", relate_score)
            else:
                raise NotImplementedError
            if sub_name in ["object_counting","dyck_languages","multistep_arithmetic_two"]:
                max_new_tokens = 20
            elif sub_name in ["word_sorting"]:
                max_new_tokens = 40
            else:
                max_new_tokens = 1

            pred = inference(prompt, model, tokenizer, template, max_new_tokens, stop_token_id, dataset=sub_name, verbose=args.verbose)
            # import ipdb;ipdb.set_trace()
            correct += int(pred==gt.lower())
            total += 1
            pbar.update(1)
            pbar.set_postfix(accuracy=f"{correct/total:.2f}")
        print()
        print("ACC:", sub_name, correct/total)
        all_resultsp[sub_name] = correct/total
    for key, val in (all_resultsp).items():
        print(key, val)

def inference(query, model, tokenizer, template, max_new_tokens, stop_token_id, dataset, verbose):
    query = query.strip()
    input_ids = build_prompt(tokenizer, template, query, [], system=None, verbose=verbose).to(model.device)
    with torch.autocast(device_type="cuda"):
        outputs = model.generate(
            input_ids=input_ids, max_new_tokens=max_new_tokens, do_sample=False,
            # top_p=top_p, temperature=temperature, repetition_penalty=repetition_penalty,
            eos_token_id=stop_token_id,
            pad_token_id=tokenizer.eos_token_id
        )
    outputs = outputs.tolist()[0][len(input_ids[0]):]
    response = tokenizer.decode(outputs)
    # import ipdb;ipdb.set_trace()
    response = response.split(tokenizer.eos_token)[0]
    if dataset=="object_counting":
        numbers = re.findall(r'\d+', response)
        response = numbers[0] if numbers else "none"
    elif dataset=="word_sorting":
        response = response.replace(",","")
        response = response.replace(".","")
    response = response.strip().lower()
    if verbose:
        print(response)
        print("--"*30)
    if response not in ["a","b","c","d","e","f","g","h","i","j","k","l","m","n","o","p","q","r","s","t","u","v","w","x","y","z"]:
        if response in ["yes","no"]:
            response = {
                "yes":"b",
                "no":"a"
            }[response]
        elif response in ["true","false"]:
            response = {
                "false":"a",
                "true":"b"
            }[response]
        elif dataset=="multirc":
            raw_answers = query.split("Candidate answers:")[-1].split("\n")[0]
            if response in raw_answers.lower():
                response = 'a'
            else:
                response = 'b'
        else:
            response = response
        # else:
        #     print("query:",query,"\nresponse:",response)
        #     raise NotImplementedError
    return response

if __name__ == '__main__':
    main()

