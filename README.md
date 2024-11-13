## Project Overview
This project aims to enhance question generation by introducing a feedback mechanism within the loss function that calculates the similarity between generated and target questions. This feedback-driven approach helps the model produce more accurate and contextually relevant questions.

This project includes the following main steps:
<details>
<summary> Data Preprocessing </summary>
  
Prepare the [SQuAD](https://www.kaggle.com/datasets/stanfordu/stanford-question-answering-dataset) dataset with fields for 'context,' 'question,' and 'answer.'
<p align=center>
  <img src = "https://github.com/user-attachments/assets/a1589919-ea36-48ec-9ce9-6104df7f478b"
</p>
</details>

<details>
<summary> Fine-tuning T5 on SQuAD </summary>
Train a T5 model on the SQuAD dataset, focusing on generating questions from context and answer pairs.

<p align=center>
  <img src = "https://github.com/user-attachments/assets/28bb2f23-223f-43e6-943c-20b34654937f"
</p>
</details>

<details>
<summary> Feedback-Enhanced Loss </summary>
  
Modify the loss function to include feedback by measuring the similarity between generated and target questions, which allows for iterative improvement in question relevance.

<p align=center>
  <img src = "https://github.com/user-attachments/assets/8a93eabc-f775-4124-91f1-823d68670d3a"
</p>

The enhanced loss function incorporates feedback to achive more accurate and contextually aligned question generation, is defined as follows:


$Total \ Loss = \alpha * loss + (1 - \alpha) * secondary \ loss \ weight * reward$


where:

* $loss$: The primary loss from the model's output.
* $reward$: A feedback term that measures the similarity between the generated question and the target question.
* $\alpha$: A weighting factor that balances the influence of the primary loss and the feedback.
* $secondary \ loss \ weight$: A coefficient that adjusts the impact of the feedback term.
</details>

## Evaluation
The model's performance is evaluated using two automated metrics: METEOR and ROUGE-L, which assess accuracy and alignment between generated and target questions. Below are the results:
<p align=center>

| Model | METEOR | ROUGE-L  |
| :---:   | :---: | :---: |
| T5 | 0.31   | 0.20  |
| T5 + Feedback-Enhanced Loss | 0.33   | 0.21  |
</p>
These results illustrate the model's improvement in question generation accuracy and relevance after implementing the feedback mechanism.

## Ablation Study
An ablation study was conducted to experiment with various parameters and configurations to optimize model performance. This study involved adjusting parameters such as dataset size and feedback weights ($\alpha$ and $secondary \ loss \ weight$ to reach the best-performing model. The findings provide insights into which configurations most effectively enhance question generation accuracy and model stability.

<p align=center>
  <img src = "https://github.com/user-attachments/assets/40dd4edf-8df7-4413-a2f6-06a46b965bf9"
</p>

## Set-Up
1. Create Directories for Data and Model
``` python
mkdir data model
```
2. Install the dependencies
``` python
pip install -r requirements.txt
```
3. Run python preprocessing.py to preprocess the data
``` python
python preprocessing.py
```

4. Run python transformer_model.py to train/eval the model with hyper-parameters found in config.py
``` python
python transformer_model.py
```


