import torch
from torchmetrics.functional.classification.accuracy import accuracy
from trainer import MyCustomTrainer

import lightning as L
from lightning.pytorch.loggers import WandbLogger

from typing import Any, Optional
from lightning.pytorch.utilities.types import STEP_OUTPUT
from transformers import AutoModelForCausalLM
from peft import get_peft_config, get_peft_model, PromptTuningInit, PromptTuningConfig, TaskType, PeftType
import torch
from datasets import load_dataset
import os
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from transformers import default_data_collator, get_linear_schedule_with_warmup
from tqdm import tqdm
from datasets import load_dataset

model_name_or_path = "bigscience/bloomz-560m"
tokenizer_name_or_path = "bigscience/bloomz-560m"
text_column = "Tweet text"
label_column = "text_label"
max_length = 64
lr = 3e-2
num_epochs = 50
batch_size = 8

peft_config = PromptTuningConfig(
    task_type=TaskType.CAUSAL_LM,
    prompt_tuning_init=PromptTuningInit.TEXT,
    num_virtual_tokens=8,
    prompt_tuning_init_text="Classify if the tweet is a complaint or not:",
    tokenizer_name_or_path=model_name_or_path,
)

class LitModel(L.LightningModule):
    def __init__(self):
        super().__init__()

        # creating model
        model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
        self.model = model

    def training_step(self, batch, batch_idx):
        outputs = self.model(**batch)
        loss = outputs.loss
        return loss

    validation_step = training_step
    test_step = training_step

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
        return optimizer


def train(fabric):
    dataset_name = "twitter_complaints"
    checkpoint_name = f"{dataset_name}_{model_name_or_path}_{peft_config.peft_type}_{peft_config.task_type}_v1.pt".replace(
        "/", "_"
    )

    dataset = load_dataset("ought/raft", dataset_name)

    classes = [k.replace("_", " ") for k in dataset["train"].features["Label"].names]
    print(classes)
    dataset = dataset.map(
        lambda x: {"text_label": [classes[label] for label in x["Label"]]},
        batched=True,
        num_proc=1,
    )
    print(dataset)
    dataset["train"][0]

    # %%
    # data preprocessing
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    target_max_length = max([len(tokenizer(class_label)["input_ids"]) for class_label in classes])
    print(target_max_length)


    def preprocess_function(examples):
        batch_size = len(examples[text_column])
        inputs = [f"{text_column} : {x} Label : " for x in examples[text_column]]
        targets = [str(x) for x in examples[label_column]]
        model_inputs = tokenizer(inputs)
        labels = tokenizer(targets)
        for i in range(batch_size):
            sample_input_ids = model_inputs["input_ids"][i]
            label_input_ids = labels["input_ids"][i] + [tokenizer.pad_token_id]
            # print(i, sample_input_ids, label_input_ids)
            model_inputs["input_ids"][i] = sample_input_ids + label_input_ids
            labels["input_ids"][i] = [-100] * len(sample_input_ids) + label_input_ids
            model_inputs["attention_mask"][i] = [1] * len(model_inputs["input_ids"][i])
        # print(model_inputs)
        for i in range(batch_size):
            sample_input_ids = model_inputs["input_ids"][i]
            label_input_ids = labels["input_ids"][i]
            model_inputs["input_ids"][i] = [tokenizer.pad_token_id] * (
                max_length - len(sample_input_ids)
            ) + sample_input_ids
            model_inputs["attention_mask"][i] = [0] * (max_length - len(sample_input_ids)) + model_inputs[
                "attention_mask"
            ][i]
            labels["input_ids"][i] = [-100] * (max_length - len(sample_input_ids)) + label_input_ids
            model_inputs["input_ids"][i] = torch.tensor(model_inputs["input_ids"][i][:max_length])
            model_inputs["attention_mask"][i] = torch.tensor(model_inputs["attention_mask"][i][:max_length])
            labels["input_ids"][i] = torch.tensor(labels["input_ids"][i][:max_length])
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs


    processed_datasets = dataset.map(
        preprocess_function,
        batched=True,
        num_proc=1,
        remove_columns=dataset["train"].column_names,
        load_from_cache_file=False,
        desc="Running tokenizer on dataset",
    )

    train_dataset = processed_datasets["train"]
    eval_dataset = processed_datasets["test"]

    def test_preprocess_function(examples):
        batch_size = len(examples[text_column])
        inputs = [f"{text_column} : {x} Label : " for x in examples[text_column]]
        model_inputs = tokenizer(inputs)
        # print(model_inputs)
        for i in range(batch_size):
            sample_input_ids = model_inputs["input_ids"][i]
            model_inputs["input_ids"][i] = [tokenizer.pad_token_id] * (
                max_length - len(sample_input_ids)
            ) + sample_input_ids
            model_inputs["attention_mask"][i] = [0] * (max_length - len(sample_input_ids)) + model_inputs[
                "attention_mask"
            ][i]
            model_inputs["input_ids"][i] = torch.tensor(model_inputs["input_ids"][i][:max_length])
            model_inputs["attention_mask"][i] = torch.tensor(model_inputs["attention_mask"][i][:max_length])
        return model_inputs


    test_dataset = dataset["test"].map(
        test_preprocess_function,
        batched=True,
        num_proc=1,
        remove_columns=dataset["train"].column_names,
        load_from_cache_file=False,
        desc="Running tokenizer on dataset",
    )

    train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=default_data_collator, batch_size=batch_size, pin_memory=True
    )

    eval_dataloader = DataLoader(eval_dataset, collate_fn=default_data_collator, batch_size=batch_size, pin_memory=True)

    

    trainer = MyCustomTrainer(
        fabric, 
        # limit_train_batches=10, 
        # limit_val_batches=20, 
        max_epochs=num_epochs,
        checkpoint_frequency=num_epochs,
    )
    model = LitModel()
    trainer.fit(model, train_dataloader, eval_dataloader)


if __name__ == "__main__":
    fabric = L.Fabric(
            accelerator='auto',
            strategy='auto',
            devices='auto',
            precision='32-true',
            plugins=None,
            callbacks=None,
            loggers=WandbLogger(),
        )
    fabric.launch(train)


    
