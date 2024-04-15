# -*- coding: utf-8 -*-
# 240411
# 1. 更新了模型加载方式；
# 2. 更新了int4模型的模型函数(quantize +@classmethod; 改init部分)，增加了lora_utils.py

# 231010
# 1. 修改了training_args的json文件：将eval和save的参数都改为了epoch

# 231008
# 1. 修改了前面的args为指定路径 减少了执行时候参数的输入
# 2. 修改了后面dataset的表头 和自定义数据集统一 这个很重要



import os
import argparse
from typing import List, Dict, Optional

import torch
import torch.nn as nn
from loguru import logger
from datasets import load_dataset
from transformers import (
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    HfArgumentParser,
    set_seed,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig
)
from peft import (
    TaskType,
    LoraConfig,
    get_peft_model,
    set_peft_model_state_dict,
    prepare_model_for_kbit_training,

)
from peft.utils import TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING


_compute_dtype_map = {
    'fp32': torch.float32,
    'fp16': torch.float16,
    'bf16': torch.bfloat16
}


def parse_args():
    parser = argparse.ArgumentParser(description='ChatGLM-6B LoRA')

    # parser.add_argument('--train_args_json', type=str, required=True, help='TrainingArguments的json文件')
    parser.add_argument('--train_args_json', type=str, default='chatGLM_6B_LoRA.json', help='TrainingArguments的json文件')

    parser.add_argument('--model_name_or_path', type=str, default='glm2-6b-int4-lora', help='模型id或local path')
    parser.add_argument('--model_bin_path', type=str, default='glm2-6b-int4-lora/pytorch_model.bin',
                        help='模型bin local path')

    # parser.add_argument('--train_data_path', type=str, required=True, help='训练数据路径')
    # parser.add_argument('--eval_data_path', type=str, default=None, help='验证数据路径')

    parser.add_argument('--train_data_path', type=str, default='data/dev.json', help='训练数据路径')
    parser.add_argument('--eval_data_path', type=str, default='data/dev.json', help='验证数据路径')

    parser.add_argument('--seed', type=int, default=42)

    parser.add_argument('--max_input_length', type=int, default=1024, help='instruction + input的最大长度')
    parser.add_argument('--max_output_length', type=int, default=1024, help='output的最大长度')

    parser.add_argument('--lora_rank', type=int, default=4, help='lora rank')
    parser.add_argument('--lora_alpha', type=int, default=32, help='lora_alpha')
    parser.add_argument('--lora_dropout', type=float, default=0.05, help='lora dropout')
    parser.add_argument('--resume_from_checkpoint', type=str, default='saved_files/chatGLM2_6B_int4_LoRA', help='恢复训练的checkpoint路径')

    # parser.add_argument('--prompt_text', type=str, default='请完成下述医疗文本事件信息提取任务：根据所提供的时间戳，提取医疗文本中包含的#影像检查、基因检测、手术、用药#四种类型的临床事件：请返回所有从文本中提取的属于#影像检查、基因检测、手术、用药#类型的临床事件的时间、事件类型、事件项目、结论；若有信息项为空，则以NONE填充；请勿自行编撰事件及时间；一项时间戳可能对应多个事件，请全部提取。时间戳及待分析文本如下：\n***', help='统一添加在所有数据前的指令文本')
    parser.add_argument('--prompt_text', type=str,
                        default='请完成下述医疗文本事件信息提取任务：根据所提供的医疗文本，判断其中是否包含#影像检查、基因检测、手术、用药#四种类型的临床事件；如果有，请返回所有从文本中提取的属于#影像检查、基因检测、手术、用药#类型的临床事件的时间、事件类型、事件项目、结论；若有信息项为空，则以NONE填充；请勿自行编撰事件及时间。待分析文本如下：\n***',
                        help='统一添加在所有数据前的指令文本')
    parser.add_argument('--end_text', type=str,
                        default='***\n请直接以jsonl输出结果，格式为：{"time": "","event_type": "","project": "","conclusion": ""}',
                        help='统一添加在所有数据后的指令文本')

    # parser.add_argument('--end_text', type=str,
    #                     default='***\n请只根据判断结果返回"Yes"或"No"。',
    #                     help='统一添加在所有数据后的指令文本')
    # parser.add_argument('--prompt_text', type=str,
    #                     default='请判断待分析文本中是否包含#影像检查、基因检测、手术、用药#四种类型的完整临床事件；如果有，请直接返回"Yes"；否则请直接返回"No"。待分析文本如下：\n***',
    #                     help='统一添加在所有数据前的指令文本')

    parser.add_argument('--compute_dtype', type=str, default='fp32',
                        choices=['fp32', 'fp16', 'bf16'], help='计算数据类型')
    return parser.parse_args()


def tokenize_func(example, tokenizer, global_args, ignore_label_id=-100):
    """单样本tokenize处理
    注意要改成适合的dataset的形式
    """
    question = global_args.prompt_text + example['input_text'] + global_args.end_text
    if example.get('input', None):
        if example['input'].strip():
            question += f"\n{example['input']}"
    answer = example['output_text']
    q_ids = tokenizer.encode(text=question, add_special_tokens=False)
    a_ids = tokenizer.encode(text=answer, add_special_tokens=False)
    if len(q_ids) > global_args.max_input_length - 2:  # 2 - gmask, bos
        q_ids = q_ids[: global_args.max_input_length - 2]
    if len(a_ids) > global_args.max_output_length - 1:  # 1 - eos
        a_ids = a_ids[: global_args.max_output_length - 1]
    input_ids = tokenizer.build_inputs_with_special_tokens(q_ids, a_ids)
    # question_length = input_ids.index(tokenizer.bos_token_id)
    question_length = len(q_ids) + 2  # chatglm1 - gmask, bos, chatglm2 - gmask, sop
    labels = [ignore_label_id] * question_length + input_ids[question_length:]
    return {'input_ids': input_ids, 'labels': labels}


def get_dataset(data_path, tokenizer, global_args):
    """读取本地数据文件，并tokenize，shuffle，返回datasets.dataset"""
    data = load_dataset('json', data_files=data_path)
    column_names = data['train'].column_names
    dataset = data['train'].map(lambda example: tokenize_func(example, tokenizer, global_args),
                                batched=False, remove_columns=column_names)
    dataset = dataset.shuffle(seed=global_args.seed)
    dataset = dataset.flatten_indices()
    return dataset


class DataCollatorForChatGLM:
    def __init__(self,
                 pad_token_id: int,
                 max_length: int = 2048,
                 ignore_label_id: int = -100):
        self.pad_token_id = pad_token_id
        self.ignore_label_id = ignore_label_id
        self.max_length = max_length

    def __call__(self, batch_data: List[Dict[str, List]]) -> Dict[str, torch.Tensor]:
        """根据batch最大长度做padding"""
        len_list = [len(d['input_ids']) for d in batch_data]
        batch_max_len = max(len_list)
        input_ids, labels = [], []
        for len_of_d, d in sorted(zip(len_list, batch_data), key=lambda x: -x[0]):
            pad_len = batch_max_len - len_of_d
            ids = d['input_ids'] + [self.pad_token_id] * pad_len
            label = d['labels'] + [self.ignore_label_id] * pad_len
            if batch_max_len > self.max_length:
                ids = ids[: self.max_length]
                label = label[: self.max_length]
            input_ids.append(torch.LongTensor(ids))
            labels.append(torch.LongTensor(label))
        input_ids = torch.stack(input_ids)
        labels = torch.stack(labels)
        return {'input_ids': input_ids, 'labels': labels}


class CastOutputToFloat(nn.Sequential):
    def forward(self, x):
        return super().forward(x).to(torch.float32)

def prepare_model_for_half_training(model, output_embedding_layer_name="lm_head",
        use_gradient_checkpointing=True, layer_norm_names=["layer_norm"]):
    r"""
    This method wrapps the entire protocol for preparing a model before running a training. This includes:
        1- Cast the layernorm in fp32 2- making output embedding layer require grads 3- Add the upcasting of the lm
        head to fp32

    Args:
        model, (`transformers.PreTrainedModel`):
            The loaded model from `transformers`
    """
    #  不要使用 model.half(), 这样会先截取精度再训练了, 最初data就要保持half
    for name, param in model.named_parameters():
        # freeze base model's layers
        param.requires_grad = False
        # cast layer norm in fp32 for stability for 8bit models
        if param.ndim == 1 and any(layer_norm_name in name for layer_norm_name in layer_norm_names):
            param.data = param.data.to(torch.float32)
        elif output_embedding_layer_name in name:  # lm_head也需要是tf.float32(最后一层)
            param.data = param.data.to(torch.float32)
        else:
            param.data = param.data.to(torch.float32)
            # param.data = param.data.to(torch.half)

    if use_gradient_checkpointing:
        # For backward compatibility
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
        # enable gradient checkpointing for memory efficiency
        model.gradient_checkpointing_enable()
    return model




class LoRATrainer(Trainer):

    def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False):
        """只保存adapter"""
        if output_dir is None:
            output_dir = self.args.output_dir
        self.model.save_pretrained(output_dir)
        torch.save(self.args, os.path.join(output_dir, "training_args.bin"))


def train(global_args):

    hf_parser = HfArgumentParser(TrainingArguments)
    hf_train_args, = hf_parser.parse_json_file(json_file=global_args.train_args_json)


    set_seed(global_args.seed)
    hf_train_args.seed = global_args.seed
    model_max_length = global_args.max_input_length + global_args.max_output_length

    # LoRA
    target_modules = TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING['chatglm']
    lora_config = LoraConfig(
        r=global_args.lora_rank,
        lora_alpha=global_args.lora_alpha,
        target_modules=target_modules,
        lora_dropout=global_args.lora_dropout,
        bias='none',
        inference_mode=False,
        task_type=TaskType.CAUSAL_LM
    )

    tokenizer = AutoTokenizer.from_pretrained(global_args.model_name_or_path, trust_remote_code=True)

    conf = AutoConfig.from_pretrained(global_args.model_name_or_path, trust_remote_code=True)

    model = AutoModel.from_config(conf, trust_remote_code=True)
    model.attach_lora(
        lora_r=global_args.lora_rank,
        lora_alpha=global_args.lora_alpha,
        lora_dropout_rate=global_args.lora_dropout,
    )
    stdc = torch.load(global_args.model_bin_path, map_location=torch.device('cpu'))
    model.load_state_dict(stdc, False)
    # model = model.cpu()

    # model = AutoModel.from_pretrained(global_args.model_name_or_path,
    #                                   device_map='auto',
    #                                   trust_remote_code=True)
    # model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    # model = prepare_model_for_half_training(model,
    #                                         use_gradient_checkpointing=True,
    #                                         output_embedding_layer_name="lm_head",
    #                                         layer_norm_names=["post_attention_layernorm",
    #                                                           "final_layernorm",
    #                                                           "input_layernorm",
    #                                                           ],
    #                                         )
    #
    # model.gradient_checkpointing_enable()
    # model.enable_input_require_grads()
    # model.is_parallelizable = True
    # model.model_parallel = True
    # # model.lm_head = CastOutputToFloat(model.lm_head)
    model.config.use_cache = (
        False
    )

    # model = get_peft_model(model, lora_config)

    resume_from_checkpoint = global_args.resume_from_checkpoint
    if resume_from_checkpoint is not None:
        checkpoint_name = os.path.join(resume_from_checkpoint, 'pytorch_model.bin')
        if not os.path.exists(checkpoint_name):
            checkpoint_name = os.path.join(
                resume_from_checkpoint, 'adapter_model.bin'
            )
            resume_from_checkpoint = False
        if os.path.exists(checkpoint_name):
            logger.info(f'Restarting from {checkpoint_name}')
            adapters_weights = torch.load(checkpoint_name)
            set_peft_model_state_dict(model, adapters_weights)
        else:
            logger.info(f'Checkpoint {checkpoint_name} not found')

    # model.print_trainable_parameters()

    # data
    train_dataset = get_dataset(global_args.train_data_path, tokenizer, global_args)
    eval_dataset = None
    if global_args.eval_data_path:
        eval_dataset = get_dataset(global_args.eval_data_path, tokenizer, global_args)

    data_collator = DataCollatorForChatGLM(pad_token_id=tokenizer.pad_token_id,
                                           max_length=model_max_length)

    # train
    trainer = LoRATrainer(
        model=model,
        args=hf_train_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator
    )

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    trainer.model.save_pretrained(hf_train_args.output_dir)


if __name__ == "__main__":
    args = parse_args()
    train(args)

