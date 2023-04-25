import torch
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.preprocessing import LabelEncoder
import pickle
import torch.nn.functional as F

with open("label_encoder.pkl", "rb") as f:
    loaded_label_encoder = pickle.load(f)

model_load_path = "trained_intent_model"
loaded_tokenizer = BertTokenizer.from_pretrained(model_load_path)
loaded_model = BertForSequenceClassification.from_pretrained(model_load_path, output_hidden_states=True)

def predict_intent_with_confidence(prompt, model, tokenizer, label_encoder, device):
    model.eval()
    model.to(device)

    encoding = tokenizer.encode_plus(
        prompt,
        add_special_tokens=True,
        max_length=128,
        return_token_type_ids=False,
        padding="max_length",
        truncation=True,
        return_attention_mask=True,
        return_tensors="pt",
    )

    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        hidden_states = outputs.hidden_states[-1]

        # Calculate softmax probabilities
        probabilities = F.softmax(logits, dim=-1)
        _, preds = torch.max(logits, dim=1)

        # Calculate confidence percentage
        confidence = torch.max(probabilities, dim=1).values.cpu().numpy()[0]

    return {
        "intent": label_encoder.inverse_transform(preds.cpu().numpy())[0],
        "confidence": confidence,
        "hidden_states": hidden_states.cpu().numpy()
    }

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
prompt = "EL manejo de las relaciones pasadas y el dejar el apego al pasado es el principal tema que maneja la ultima pelicula de Makoto Shinkay, alejándose un poco del romance que tanto se presentaba en sus ultimas películas"
predicted_result = predict_intent_with_confidence(prompt, loaded_model, loaded_tokenizer, loaded_label_encoder, device)
print(f"Predicted intent: {predicted_result['intent']}")
print(f"Confidence: {predicted_result['confidence'] * 100:.2f}%")
#print(f"Hidden states: {predicted_result['hidden_states']}")