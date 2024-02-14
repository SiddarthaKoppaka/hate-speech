from transformers import TextClassificationPipeline, DistilBertTokenizer, DistilBertForSequenceClassification, BertForSequenceClassification, BertTokenizer, AutoTokenizer, XLMRobertaForSequenceClassification
import torch

def check_model(model_name):
    if model_name == "DistilBERT":
        model = DistilBertForSequenceClassification.from_pretrained('SiddarthaKoppaka/hate-speech-telugu-distilbert')
        tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-multilingual-cased')
        return (model,tokenizer)
    if model_name == "BERT":
        model =  BertForSequenceClassification.from_pretrained('SiddarthaKoppaka/Hate-speech-telugu-bert')
        tokenizer = BertTokenizer.from_pretrained("Bert-base-multilingual-cased")
        return (model,tokenizer)
    if model_name == "MuRIL":
        model = BertForSequenceClassification.from_pretrained('SiddarthaKoppaka/hate-speech-telugu-muril')
        tokenizer = BertTokenizer.from_pretrained('google/muril-base-cased')
        return (model,tokenizer)
    if model_name == "RoBERTa":
        model = XLMRobertaForSequenceClassification.from_pretrained('SiddarthaKoppaka/hate-speech-telugu-roberta')
        tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-emotion")
        return (model,tokenizer)
    if model_name == "Indic-BERT":
        model = BertForSequenceClassification.from_pretrained('SiddarthaKoppaka/hate-speech-telugu-indicbert')
        tokenizer = BertTokenizer.from_pretrained('ai4bharat/indic-bert')
        return (model,tokenizer)
    if model_name == "NLLB":
        # model = torch.load_model('models/NLLB_tel/')
        # tokenizer = AutoTokenizer.from_pretrained('facebook/nllb-200-distilled-600M')
        return 'cominggggg sooonnnn'
    if model_name == 'BART':
        return 'Not Saved, Try another model.....'

def make_prediction(input, model_name):
    # if model_name == 'NLLB' or 'BART':
    #     return 'Coming Soon...'

    model,tokenizer = check_model(model_name)
    pipe = TextClassificationPipeline(model=model, tokenizer=tokenizer, top_k=1)
    pred = pipe(input)
    # print(pred)
    # print(pred[0][0])
    prediction = True if pred[0][0]['label'] == 'LABEL_1' else False
    # print(prediction)
    return 'This sentence contains Hate' if prediction == True else "Does not contain Hate" 