![image](https://github.com/SiddarthaKoppaka/Unmasking_hate/assets/95752301/5113db50-74b6-457f-aca2-e40b327a2a28)
# Unmasking Hate: Telugu Language Hate Speech Recognition

## Abstract
In the digital era, the spread of abusive language and hate speech on social media is alarming, especially for low-resource languages like Telugu. Our project, "Unmasking Hate," aims to combat abusive language in Telugu by collecting a comprehensive dataset from Twitter and training Transformer models. Our goal is to contribute towards a safer online environment by leveraging advanced natural language processing techniques.

## Introduction
Hate speech, targeting individuals or groups based on characteristics such as race, religion, or gender, has grown with social media's rise. For Telugu, a language spoken by millions yet underrepresented in research, we've taken a step forward by creating datasets and developing hate speech recognition models to address this issue head-on.

## Related Work
Our work draws inspiration from advancements in hate speech detection in languages like Hindi and Marathi, utilizing CNNs, LSTMs, and Transformer models like BERT and RoBERTa. We extend these methodologies to Telugu, a language with distinct linguistic and cultural nuances.

## Data Collection and Preprocessing
### Gathering and Preparing the Tweet Dataset
We collected ~50,000 tweets using snscrape, focusing on keywords related to abusive language in Telugu. Post-collection, we embarked on rigorous data cleaning and manual annotation to ensure the dataset's quality for model training.

### Data Cleaning & Preprocessing
Through regular expressions, we removed mentions, user IDs, emojis, and irrelevant English words, aiming for a balanced dataset of positive and negative labels to train our models effectively.

## Training the Model
### Model Description
We explored a variety of models, from RNNs and LSTMs to state-of-the-art Transformers, aiming to identify the most effective architecture for hate speech detection in Telugu.

### Training Approaches
- **RNN & LSTM**: Utilized for their capability to capture sequential patterns and long-term dependencies within text data.
- **Transformers**: Implemented models like mBERT, DistilBERT, Indic-BERT, NLLB, and MuRIL, leveraging their self-attention mechanisms and pre-trained knowledge for better performance in language understanding and hate speech detection.

The Modelling phase went through a lot of cahnges, we tried implementing the RNN + LSTM model at first, which made us realise that we can use better model & then decided to use the recent state-of-the-art models the `TRANFSFORMERS`.
Starting with BERT with Multilingual Support, we then implemented the DistilBERT, RoBERTa. Later, we've got to know about the Mutilingual transformer model which is a fine-tuned version of BERT, developed by google. Later, we trided Implementing the ai4Bharat's Indic-BERT model ( a fine tuned Albert model) , which is specifically fine tuned for the Indian Languages. In our research, we came across a great model which is NLLB ( No Language Left Behind ) By the Meta, which supports almost every language in the world. But the problem with NLLB, it is not designed to directly support classification, so we modified the architecture by adding a classification head to leverage it best.

## Evaluation and Metrics
We evaluated the models using F1 score, recall, precision, and accuracy. Our results show significant promise, with most Transformer models achieving over 95% accuracy, and mBERT leading at 98.2%.

Results :

| Model         | F1 Score | Precision | Recall | Accuracy |
|---------------|----------|-----------|--------|----------|
| RNN + LSTM    | 91       | 92        | 91     | 91       |
| mBERT         | 98.2126  | 98.2368   | 98.2124| 98.2124  |
| DistilBERT    | 98.0283  | 98.0451   | 98.0283| 98.0283  |
| XLM-RoBERTa   | 85.3617  | 85.5656   | 85.3838| 85.3838  |
| mBART         | 91       | 92        | 91     | 91       |
| MuRIL         | 97.9495  | 97.9656   | 97.9495| 97.9495  |
| Indic-BERT    | 98.16    | 98.1842   | 98.1598| 98.1598  |
| Indic-BART    | 33.9     | 25.7      | 50     | 51.3     |
| NLLB          | 97.3     | 97.3      | 97.3   | 97.3     |


## Conclusion
Our research presents a significant step towards detecting hate speech in Telugu. Through the creation of a custom dataset and the application of Transformer models, we demonstrated the effectiveness of these approaches in addressing hate speech, with mBERT showcasing outstanding performance. This Repo Contains the implementation code, but not the trained models nor the data ( We are still in the process of publishing the paper, so 😅).

## References

1. BERT - https://arxiv.org/abs/1810.04805
2. DistilBERT - https://arxiv.org/abs/1907.11692
3. RoBERTa - https://arxiv.org/abs/1910.01108
4. NLLB - https://arxiv.org/abs/2207.04672
5. Indic-BERT - https://indicnlp.ai4bharat.org/pages/indic-bert/
6. MuRIL - https://arxiv.org/abs/2103.10730
7. mBART - https://arxiv.org/abs/2001.08210
8. Indic-BART - https://arxiv.org/abs/2109.02903

We could pull this off, thanks to Hugging Face - https://huggingface.co/ & Pytorch - https://pytorch.org/.
