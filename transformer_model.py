from transformers import BartForConditionalGeneration, BartTokenizerFast, AutoTokenizer, AutoModel, T5ForConditionalGeneration, T5Tokenizer
from transformers import (
    AdamW,
    T5ForConditionalGeneration,
    T5Tokenizer,
    get_linear_schedule_with_warmup,
    T5Tokenizer
)
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import torch
import tensorflow as tf
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
import config
import os
import pandas as pd
from tqdm import tqdm
from datasets import load_dataset, load_metric
from pprint import pprint
import copy
import argparse
import numpy as np
import matplotlib.pyplot as plt
import pickle
import gc
from memory_profiler import profile

gc.collect()
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

class QuestionGenerationDataset(Dataset):
    def __init__(self, tokenizer, filepath, sapmle_num):
        self.path = filepath
        self.passage_column = "context"
        self.answer = "answer"
        self.question = "question"
        self.data = pd.read_parquet(self.path).iloc[:sapmle_num,:]
        self.max_len_input = config.max_len_input
        self.max_len_output = config.max_len_output
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data.iloc[index]
        passage, answer, question = row[self.passage_column], row[self.answer], row[self.question]

        input_text = f"context: {passage} answer: {answer}"
        target_text = f"question: {question}"

        'Tokenize input and target on the fly'
        tokenized_inputs = self.tokenizer(
            input_text, 
            max_length=self.max_len_input,
            padding='max_length',
            truncation=True,
            return_tensors="pt"
        )
        tokenized_targets = self.tokenizer(
            target_text, 
            max_length=self.max_len_output,
            padding='max_length',
            truncation=True,
            return_tensors="pt"
        )

        source_ids = tokenized_inputs['input_ids'].squeeze()
        target_ids = tokenized_targets['input_ids'].squeeze()
        src_mask = tokenized_inputs['attention_mask'].squeeze()
        target_mask = tokenized_targets['attention_mask'].squeeze()

        labels = target_ids.clone()
        labels[labels == 0] = -100

        return {
            "source_ids": source_ids, 
            "source_mask": src_mask, 
            "target_ids": target_ids, 
            "target_mask": target_mask,
            "labels":labels
        }

@profile
class T5Tuner(pl.LightningModule):
    def __init__(self, t5model, t5tokenizer, train_dataset, validation_dataset):
        super().__init__()
        self.model = t5model
        self.tokenizer = t5tokenizer
        self.batch_size = config.batch_size
        self.train_dataset = train_dataset
        self.validation_dataset = validation_dataset
        self.train_losses = []
        self.val_losses = []
        self.rl_train_losses = []
        self.rl_val_losses = []
        self.rl_train_scores = []
        self.rl_val_scores = []
        self.secondary_loss_weight = config.secondary_loss_weight
        self.curr_training_step = 0
        self.curr_validation_step = 0

    def forward(self, input_ids, attention_mask=None, 
                decoder_attention_mask=None, 
                lm_labels=None):
      
         outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_attention_mask=decoder_attention_mask,
            labels=lm_labels,
        )
         return outputs

    def generate_question(self, input_ids, attention_mask):
        """Sampled question generation (using top-k sampling for diversity)"""
        sampled_output = self.model.generate(
            input_ids=input_ids, 
            attention_mask=attention_mask,
            max_length=config.max_len_output, 
            do_sample=True, 
            top_k=50
        )
 
        sampled_question = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in sampled_output]

        return sampled_question
    
    def prepare_data_for_reward(self, old_candidate, old_reference):
        reference = []
        candidate = []
        for sample in old_candidate:
            candidate.append(' '.join(sample))
        for sample in old_reference:
            reference.append(' '.join(sample))
        reference = " ".join(reference)
        candidate = " ".join(candidate)
        return candidate, reference

    '''Calculate ROUGE-L and and METEOR scores'''
    def compute_reward(self, generated_question, reference_question):
        if isinstance(reference_question, tf.Tensor):
            reference_question = reference_question.numpy()
    
        if isinstance(reference_question, bytes):
            reference_question = reference_question.decode('utf-8')
        
        vocab_size = self.tokenizer.vocab_size
        reference_question = [token_id for token_id in reference_question if 0 <= token_id < vocab_size]
        reference_question = self.tokenizer.decode(reference_question,  truncation=True)

        candidate = [str(generated_question).split()]
        reference = [reference_question.split()]
        candidate, reference = self.prepare_data_for_reward(candidate, reference)
        scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        rouge_l = scorer.score(reference, candidate)['rougeL'].fmeasure
        return rouge_l

    def calculate_feedback(self, batch):
        sampled_question = self.generate_question(batch['source_ids'], batch["source_mask"])
        sampled_rewards = [self.compute_reward(sq, ref) for sq, ref in zip(sampled_question, batch['labels'])]
        rewards = {'sampled': torch.tensor(sampled_rewards, dtype=torch.float32, device=self.device)}
        mean_reward = rewards['sampled'].mean()
        return mean_reward

    def training_step(self, batch, batch_idx):
        outputs = self.forward(
            input_ids=batch["source_ids"],
            attention_mask=batch["source_mask"],
            decoder_attention_mask=batch['target_mask'],
            lm_labels=batch['labels']
        )
        loss = outputs[0]
        feedback = self.calculate_feedback(batch)
        final_loss = torch.clamp(config.alpha * loss + (1 - config.alpha) * (1-feedback), min=0)
        
        self.train_losses.append(loss)
        self.rl_train_losses.append(final_loss)
        self.rl_train_scores.append(feedback)
        self.log('train_loss', final_loss)
        self.curr_training_step += 1
        return final_loss

    def validation_step(self, batch, batch_idx):
        outputs = self.forward(
            input_ids=batch["source_ids"],
            attention_mask=batch["source_mask"],
            decoder_attention_mask=batch['target_mask'],
            lm_labels=batch['labels']
        )
        
        loss = outputs[0]
        feedback = self.calculate_feedback(batch)
        final_loss = torch.clamp(config.alpha * loss + (1 - config.alpha) * (1-feedback), min=0)
        
        self.val_losses.append(loss)
        self.rl_val_losses.append(final_loss)
        self.rl_val_scores.append(feedback)
        self.log('val_loss', final_loss)
        self.curr_training_step += 1
        return final_loss

    def train_dataloader(self):
        return DataLoader(
            train_dataset, 
            batch_size=self.batch_size,
            num_workers=config.num_workers,
            persistent_workers=True
        )

    def val_dataloader(self):
        return DataLoader(validation_dataset,
            batch_size=self.batch_size,
            num_workers=config.num_workers,
            persistent_workers=True
        )

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=3e-4, eps=1e-8)
        return optimizer

def get_question(context, answer, model, tokenizer):

    ''' function to generate questions. Takes a sentence,answer,
        model and tokenizer 
    '''

    text = "context: {} answer: {}".format(context, answer)
    encoding = tokenizer.encode_plus(
        text, 
        max_length=config.max_len_input, 
        pad_to_max_length=False, 
        truncation=True, 
        return_tensors="pt"
    )

    input_ids, attention_mask = encoding["input_ids"], encoding["attention_mask"]

    outputs = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        early_stopping=True,
        num_beams=5,
        num_return_sequences=1,
        no_repeat_ngram_size=2,
        max_length=config.max_len_input
    )
    decoded_output = [tokenizer.decode(ids, skip_special_tokens=True) for ids in outputs]

    Question = decoded_output[0].replace("question:", "")
    Question = decoded_output[0].strip()
    return Question

def compute_score(y_pred, y_true):
    bleu_scores = []
    rougel_scores = []
    meteor_scores = []
    for candidate, reference in zip(y_pred, y_true):
        print(f'reference {reference[0]}')
        print(f'candidate {candidate[0]}:')
        if isinstance(reference, list):
            reference = " ".join(reference)
        if isinstance(candidate, list):
            candidate = " ".join(candidate)
        scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        rouge_l = scorer.score(reference, candidate)['rougeL'].fmeasure
        meteor = meteor_score([reference], candidate)
        rougel_scores.append(rouge_l)
        meteor_scores.append(meteor)
    return rougel_scores, meteor_scores

def save_result(rougel_scores, meteor_scores, score):
    with open(f'result.txt', 'a') as f:
        f.write(f'\n*\t*\t*\t*\t*\t*\t*\t*\t*\t*\t*\t*\t*\t*\t*\t*\t*\t*\t\n')
        f.write(f'Evaluation number {config.num_of_train} Done With These Parameters:\n\tbatch size: {config.batch_size}\n\tnumber of samples: {config.samples}\n\tepoch num: {config.epoch_num}\n\tmax len input: {config.max_len_input}\n\tmax len output: {config.max_len_output}\n\tmodel name: {config.model_name}\n')
        f.write(f'The Achived Scores Are Shown Below:\n\trouge-l: {np.mean(rougel_scores)}\n\tmeteor: {np.mean(meteor_scores)}')
        f.write(f'\n*\t*\t*\t*\t*\t*\t*\t*\t*\t*\t*\t*\t*\t*\t*\t*\t*\t*\t\n')

def remove_extra_char(sentence):
    return [word.replace('▁', '') for word in sentence]

def prepare_data_for_bleu(old_y_pred, old_y_true):
    y_true = []
    y_pred = []
    for sample in old_y_pred:
        y_pred.append([' '.join(sample)])
    y_true = old_y_true
    return y_pred, y_true

def evaluation(loader, model, tokenizer, device='cpu'):
    y_true = []
    y_pred = []
    for i, batch in enumerate(loader):
        tmp = batch['target_ids']
        encoded_input = [line for line in batch['source_ids']]
        encoded_input = torch.stack([torch.tensor(item) for item in encoded_input]).to(device)

        outputs = model.generate(encoded_input, max_length=175)
        batch_pred = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        for sentence in batch['target_ids']:
            if isinstance(sentence, tf.Tensor):
                sentence = sentence.numpy()    
            if isinstance(sentence, bytes):
                sentence = sentence.decode('utf-8')
            if isinstance(sentence, torch.Tensor):
                sentence = tokenizer.decode(sentence, skip_special_tokens=True)
            else:
                sentence = tokenizer.tokenize(sentence)
            y_true.append([sentence])

        for sentence in batch_pred:
            if isinstance(sentence, str):
                sentence = tokenizer.tokenize(sentence)
            sentence = remove_extra_char(sentence)
            y_pred.append(sentence)
    
    y_pred, y_true = prepare_data_for_bleu(y_pred, y_true)

    rougel_scores, meteor_scores = compute_score(y_pred, y_true)
    save_result(rougel_scores, meteor_scores)

def plot_losses(train_losses, val_losses, rl_train_losses=None, rl_val_losses=None, score=None):
    val_epochs = range(1, len(val_losses) + 1)
    train_epochs = range(1, len(train_losses) + 1)
    
    val_losses = [val_loss.detach().numpy() if isinstance(val_loss, torch.Tensor) else val_loss for val_loss in val_losses]
    train_losses = [train_loss.detach().numpy() if isinstance(train_loss, torch.Tensor) else train_loss for train_loss in train_losses]
    
    if rl_train_losses is not None:
        rl_val_losses = [rl_loss.detach().numpy() if isinstance(rl_loss, torch.Tensor) else rl_loss for rl_loss in rl_val_losses]
        rl_train_losses = [rl_loss.detach().numpy() if isinstance(rl_loss, torch.Tensor) else rl_loss for rl_loss in rl_train_losses]
    
    plt.plot(val_epochs, val_losses, label='Validation Loss')
    plt.plot(train_epochs, train_losses, label='Training Loss')
    
    if rl_train_losses is not None:
        plt.plot(val_epochs, rl_val_losses, label='Validation Loss for RL')
        plt.plot(train_epochs, rl_train_losses, label='Training Loss for RL')
    
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig(os.path.join(config.model_path, f'plot_{config.num_of_train}.png'))
    plt.show()
    plt.close()
    

def plot_scores(train_scores, val_scores):
    val_epochs = range(1, len(val_scores) + 1)
    train_epochs = range(1, len(train_scores) + 1)
    
    val_scores = [val_score.detach().numpy() if isinstance(val_score, torch.Tensor) else val_score for val_score in val_scores]
    train_scores = [train_score.detach().numpy() if isinstance(train_score, torch.Tensor) else train_score for train_score in train_scores]

    plt.plot(val_epochs, val_scores, label='Validation Score')
    plt.plot(train_epochs, train_scores, label='Training Score')
    plt.xlabel('Steps')
    plt.ylabel('Score')
    plt.title('Training and Validation Scores')
    plt.legend()
    plt.savefig(os.path.join(config.model_path, f'score_plot_{config.num_of_train}.png'))
    plt.show()
    plt.close()

def save_model(model):
    model.model.save_pretrained(os.path.join(config.model_path, f't5_trained_model_{config.num_of_train}'))
    model.tokenizer.save_pretrained(os.path.join(config.model_path, f't5_tokenizer_{config.num_of_train}'))
    with open(os.path.join(config.model_path, f'train_losses_{config.num_of_train}.pkl'), 'wb') as f:
        pickle.dump(model.train_losses, f)
    with open(os.path.join(config.model_path, f'val_losses_{config.num_of_train}.pkl'), 'wb') as f:
        pickle.dump(model.val_losses, f)
    with open(os.path.join(config.model_path, f'rl_train_losses_{config.num_of_train}.pkl'), 'wb') as f:
        pickle.dump(model.rl_train_losses, f)
    with open(os.path.join(config.model_path, f'rl_val_losses_{config.num_of_train}.pkl'), 'wb') as f:
        pickle.dump(model.rl_val_losses, f)
    with open(os.path.join(config.model_path, f'rl_train_scores_{config.num_of_train}.pkl'), 'wb') as f:
        pickle.dump(model.rl_train_scores, f)
    with open(os.path.join(config.model_path, f'rl_val_scores_{config.num_of_train}.pkl'), 'wb') as f:
        pickle.dump(model.rl_val_scores, f)

if __name__ == "__main__":

    if 'train' in config.run_model:
        t5_model = T5ForConditionalGeneration.from_pretrained(config.model_name)
        t5_tokenizer = T5Tokenizer.from_pretrained(config.model_name)
        train_dataset = QuestionGenerationDataset(t5_tokenizer, config.train_path, config.samples)
        validation_dataset = QuestionGenerationDataset(t5_tokenizer, config.validation_path, int(np.floor(config.val_amount * config.samples)))
        model = T5Tuner(t5_model, t5_tokenizer, train_dataset, validation_dataset)
        checkpoint_callback = ModelCheckpoint(
            dirpath= config.checkpoint_path,
            filename='best-checkpoint',  
            save_top_k=1,  
            verbose=True,
            mode='min',
            save_weights_only=True
        )

        trainer = pl.Trainer(
            max_epochs=config.epoch_num, 
            accelerator='cpu', 
            callbacks=[checkpoint_callback], 
            val_check_interval=config.val_amount
        )
        trainer.fit(model)
        save_model(model)
        loader = DataLoader(validation_dataset, batch_size=config.batch_size)
        evaluation(loader, t5_model, t5_tokenizer)
        plot_losses(model.train_losses, model.val_losses, model.rl_train_losses, model.rl_val_losses)
        plot_scores(model.rl_train_scores, model.rl_val_scores)

    elif 'evaluation' in config.run_model:
        'Prediction / Inference'    
        t5_model = T5ForConditionalGeneration.from_pretrained(os.path.join(config.model_path, f't5_trained_model_{config.num_of_train}'))
        t5_tokenizer = T5Tokenizer.from_pretrained(os.path.join(config.model_path, f't5_tokenizer_{config.num_of_train}'))
        with open(os.path.join(config.model_path, f'train_losses_{config.num_of_train}.pkl'), 'rb') as f:
            train_losses = pickle.load(f)
        with open(os.path.join(config.model_path, f'val_losses_{config.num_of_train}.pkl'), 'rb') as f:
            val_losses = pickle.load(f)
        with open(os.path.join(config.model_path, f'rl_train_losses_{config.num_of_train}.pkl'), 'rb') as f:
            rl_train_losses = pickle.load(f)
        with open(os.path.join(config.model_path, f'rl_val_losses_{config.num_of_train}.pkl'), 'rb') as f:
            rl_val_losses = pickle.load(f)
        with open(os.path.join(config.model_path, f'rl_train_scores_{config.num_of_train}.pkl'), 'rb') as f:
            rl_train_scores = pickle.load(f)
        with open(os.path.join(config.model_path, f'rl_val_scores_{config.num_of_train}.pkl'), 'rb') as f:
            rl_val_scores = pickle.load(f)
        print(f'train_losses: {train_losses}')
        
        validation_dataset = QuestionGenerationDataset(t5_tokenizer, config.validation_path, int(np.floor(config.val_amount * config.samples)))
        loader = DataLoader(validation_dataset, batch_size=config.batch_size)
        evaluation(loader, t5_model, t5_tokenizer)
        plot_losses(train_losses, val_losses, rl_train_losses, rl_val_losses)
        plot_scores(rl_train_scores, rl_val_scores)

    elif 'experiment' in config.run_model: 
        t5_model = T5ForConditionalGeneration.from_pretrained(os.path.join(config.model_path, f't5_trained_model_{config.num_of_train}'))
        t5_tokenizer = T5Tokenizer.from_pretrained(os.path.join(config.model_path, f't5_tokenizer_{config.num_of_train}'))
        
        # context = "Albert Einstein was a German-born theoretical physicist who developed the theory of relativity."
        # answer = "Albert Einstein"
        # text = "context: "+ context + " " + "answer: " + answer
        
        context = "The Pacific Ocean is the largest and deepest of Earth's oceanic divisions."
        answer = "The Pacific Ocean"
        text = "context: " + context + " " + "answer: " + answer
        
        input_ids = t5_tokenizer.encode(text, return_tensors="pt", max_length=400, truncation=True)
        output_ids = t5_model.generate(input_ids=input_ids, max_length=40)
        generated_question = t5_tokenizer.decode(output_ids[0], skip_special_tokens=True)
        print("Generated Question: ", generated_question)