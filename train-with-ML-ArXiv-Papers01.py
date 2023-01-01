import time
import numpy as np
from transformers import Trainer, TrainingArguments, AutoModelForCausalLM, DataCollatorForLanguageModeling
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import evaluate


def tokenize_function(examples):
    current_tokenizer_result = tokenizer(examples["abstract"], padding="max_length", truncation=True)
    return current_tokenizer_result


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


if __name__ ==  '__main__':

    start = time.time()

    GPT_FINE_TUNED_FILE = "fine_tuned_models/gpt-neo-125M-ML-ArXiv-Papers"

    print("Loading model")
    model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neo-125M", low_cpu_mem_usage=True)
    model.config.pad_token_id = model.config.eos_token_id

    print("Loading tokenizer")
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M")
    tokenizer.pad_token = tokenizer.eos_token

    print("Loading dataset")
    current_dataset = load_dataset("CShorten/ML-ArXiv-Papers", split="train")
    current_dataset = current_dataset.select(range(1200))


    print("Splitting and tokenizing dataset")
    tokenized_datasets = current_dataset.map(tokenize_function, batched=True)
    small_train_dataset = tokenized_datasets.select(range(100))

    print("Preparing training arguments")

    training_args = TrainingArguments(output_dir=GPT_FINE_TUNED_FILE,
                                      report_to='all',
                                      logging_dir='./logs', # directory for storing logs
                                      label_names=['input_ids', 'attention_mask'],  # 'logits', 'past_key_values'
                                      no_cuda=True,
                                      num_train_epochs=1,  # total # of training epochs
                                      per_device_train_batch_size=1,  # batch size per device during training
                                      per_device_eval_batch_size=1,  # batch size for evaluation
                                      warmup_steps=200,  # number of warmup steps for learning rate scheduler
                                      weight_decay=0.01,  # strength of weight decay
                                      prediction_loss_only=True,
                                      save_steps=10000)

    metric = evaluate.load("accuracy")

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=small_train_dataset,
        compute_metrics=compute_metrics,
        data_collator=data_collator,
    )

    print("Starting training")
    trainer.train()
    print(f"Finished fine-tuning in {time.time() - start}")

    trainer.save_model()
    tokenizer.save_pretrained(GPT_FINE_TUNED_FILE)