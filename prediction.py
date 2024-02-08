from transformers import AutoTokenizer, AutoModelForSequenceClassification, BertForSequenceClassification, BertTokenizer, DistilBertForSequenceClassification, XLMRobertaForSequenceClassification,pipeline, DistilBertTokenizer, TextClassificationPipeline
import torch

def make_prediction(text, model_name):
    if model_name == 'DistilBERT':
        tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-multilingual-cased')
        pipe = TextClassificationPipeline(model="SiddarthaKoppaka/hate-speech-telugu-distilbert",tokenizer=tokenizer , top_k=1)
    if model_name == 'BERT':
        tokenizer = BertTokenizer.from_pretrained("Bert-base-multilingual-cased")
        pipe = TextClassificationPipeline(model="SiddarthaKoppaka/hate-speech-telugu-distilbert",tokenizer=tokenizer, top_k=1)
    if model_name == 'MuRIL':
        tokenizer = BertTokenizer.from_pretrained('google/muril-base-cased')
        pipe = TextClassificationPipeline(model="SiddarthaKoppaka/hate-speech-telugu-muril",tokenizer=tokenizer, top_k=1)
    if model_name == 'RoBERTa':
        tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-emotion")
        pipe = pipeline("text-classification", model="SiddarthaKoppaka/hate-speech-telugu-roberta",tokenizer=tokenizer, top_k=1)
    if model_name == 'Indic-BERT':
        tokenizer = BertTokenizer.from_pretrained('ai4bharat/indic-bert')
        pipe = pipeline("text-classification", model="SiddarthaKoppaka/hate-speech-telugu-indicbert",tokenizer=tokenizer, top_k=1)
    if model_name == 'NLLB':
        return 'Coming soon dudu'
    
    pred = pipe(input)
    prediction = True if pred[0][0]['label'] == 'LABEL_1' else False
    return 'This sentence contains Hate' if prediction == True else "Does not contain Hate"