![image](https://github.com/user-attachments/assets/5c772e27-9102-4b0a-afd1-dbcb6bc0b9c4)# Improving Question Generation with Transformer Based Models

This project aims to enhance question generation by introducing a feedback mechanism within the loss function that calculates the similarity between generated and target questions. This feedback-driven approach helps the model produce more accurate and contextually relevant questions.

## Project Overview
This project includes the following main steps:

### Data Preprocessing: 
Prepare a dataset with "context", "question", and "answer" fields.
<p align=center>
  <img src = "https://github.com/user-attachments/assets/a1589919-ea36-48ec-9ce9-6104df7f478b"
</p>

### Fine-tuning T5 on SQuAD: 
Train a T5 model on the SQuAD dataset, focusing on generating questions from context and answer pairs.

<p align=center>
  <img src = "https://github.com/user-attachments/assets/28bb2f23-223f-43e6-943c-20b34654937f"
</p>


### Feedback-Enhanced Loss: 
Modify the loss function to include feedback by measuring the similarity between generated and target questions, which allows for iterative improvement in question relevance.

<p align=center>
  <img src = "https://github.com/user-attachments/assets/8a93eabc-f775-4124-91f1-823d68670d3a"
</p>

The enhanced loss function, incorporating feedback for improved question relevance, is defined as follows:
``` math
Total Loss = \alpha * loss + (1 - \alpha) * secondary loss weight * reward
```

where:

* $loss$: The primary loss from the model's output.
* $reward$: A feedback term that measures the similarity between the generated question and the target question.
* $\alpha$: A weighting factor that balances the influence of the primary loss and the feedback.
* $secondary \ loss \ weight$: A coefficient that adjusts the impact of the feedback term.
