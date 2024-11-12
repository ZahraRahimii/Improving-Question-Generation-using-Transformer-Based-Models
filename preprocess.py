import json
import requests
import pandas as pd
from datasets import load_dataset
import config
import os
import re

class Preprocess:
    def remove_additional_chars(self, example):
        if example is not None:
            return re.sub(r"[',\n?]", "", example)

    def extract_data(self, example):
        context = self.remove_additional_chars(example['context'])
        question = self.remove_additional_chars(example['question'])
        answer = example['answers']['text'][0] if example['answers']['text'] else None
        answer = self.remove_additional_chars(answer)

        return {'context': context, 'question': question, 'answer': answer}

    def create_data(self, squad_v2, name):
        data = squad_v2[name].map(self.extract_data)
        df = pd.DataFrame(data)
        
        if 'answers' in df.columns:
            df.drop(columns=['answers'], inplace=True)
        if 'title' in df.columns:
            df.drop(columns=['title'], inplace=True)
        if 'id' in df.columns:
            df.drop(columns=['id'], inplace=True)
        df.dropna(subset=['answer'], inplace=True)
        df.to_csv(os.path.join(config.data_path, f'{name}.csv'))
        df.to_parquet(os.path.join(config.data_path, f'{name}_squad.parquet'))
        

if __name__ == "__main__":
    squad_v2 = load_dataset("squad_v2")
    preprocess = Preprocess()
    preprocess.create_data(squad_v2, 'train')
    preprocess.create_data(squad_v2, 'validation')