# Improving Question Generation with Transformer Based Models

This project aims to enhance question generation by introducing a feedback mechanism within the loss function that calculates the similarity between generated and target questions. This feedback-driven approach helps the model produce more accurate and contextually relevant questions.

## Project Overview
This project includes the following main steps:

### Data Preprocessing: 
Prepare a dataset with "context", "question", and "answer" fields.
<div align=center>
  <img src = "https://github.com/user-attachments/assets/a1589919-ea36-48ec-9ce9-6104df7f478b"
</div>

### Fine-tuning T5 on SQuAD: 
Train a T5 model on the SQuAD dataset, focusing on generating questions from context and answer pairs.
<div align=center>
  <img src = "https://github.com/user-attachments/assets/a1589919-ea36-48ec-9ce9-6104df7f478b"
</div>

### Feedback-Enhanced Loss: 
Modify the loss function to include feedback by measuring the similarity between generated and target questions, which allows for iterative improvement in question relevance.

<div align=center>
  <img src = "https://github.com/user-attachments/assets/8a93eabc-f775-4124-91f1-823d68670d3a"
</div>
