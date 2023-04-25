import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import pickle

# LEER ARCHIVO
df = pd.read_csv("datamovie.csv", sep="|", names=["intent", "prompt"], header=None)

# Obtener los valores únicos de la columna "intent"
unique_intents = df["intent"].unique()

# Crear un DataFrame vacío para almacenar los datos reducidos
reduced_df = pd.DataFrame(columns=["intent", "prompt"])

# Calcular el conteo de filas por "intent" y filtrar solo aquellos con al menos 50 filas
intent_counts = df["intent"].value_counts()
filtered_intents = intent_counts[intent_counts >= 200].index

# Encontrar el mínimo número de filas para los intents filtrados
min_num_rows = min(intent_counts[filtered_intents])

# Iterar sobre los intents filtrados
for intent in filtered_intents:
    # Obtener las filas que corresponden a este valor de "intent"
    intent_rows = df[df["intent"] == intent]

    # Tomar el mínimo número de filas para cada "intent" y agregarlas al DataFrame reducido
    reduced_df = pd.concat([reduced_df, intent_rows.iloc[:min_num_rows]])

# Verificar el tamaño del DataFrame reducido
print(len(reduced_df))

# Calcular el conteo de filas por "intent" en el DataFrame reducido
reduced_intent_counts = reduced_df["intent"].value_counts()

# Crear un gráfico de barras con los conteos de "intent"
plt.figure(figsize=(12, 6))
plt.bar(reduced_intent_counts.index, reduced_intent_counts.values)
plt.xlabel("Intents")
plt.ylabel("Cantidad de datos")
plt.title("Cantidad de datos por intent en el DataFrame reducido")
plt.xticks(rotation=90)

# Mostrar el gráfico
plt.show()



# Codificar las etiquetas de intención como números enteros
label_encoder = LabelEncoder()
reduced_df["encoded_intent"] = label_encoder.fit_transform(reduced_df["intent"])

# Guardar el label_encoder en un archivo pickle
with open("label_encoder.pkl", "wb") as f:
    pickle.dump(label_encoder, f)

#PREPROCESAMIENTO DE DATOS

# Dividir los datos en conjuntos de entrenamiento y prueba
train_df, test_df = train_test_split(reduced_df, test_size=0.1, random_state=42)

# Crear un conjunto de datos personalizado para PyTorch
class IntentDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len):
        self.dataframe = dataframe
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        row = self.dataframe.iloc[index]
        encoded_intent, prompt = row["encoded_intent"], row["prompt"]
        encoding = self.tokenizer.encode_plus(
            prompt,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": torch.tensor(encoded_intent, dtype=torch.long),
        }


# Inicializar el tokenizador
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Crear conjuntos de datos
max_len = 128
train_dataset = IntentDataset(train_df, tokenizer, max_len)
test_dataset = IntentDataset(test_df, tokenizer, max_len)

# Crear DataLoaders
batch_size = 16
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


#CREAR Y ENTRENAR MODELO

# Crear el modelo
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=len(df.intent.unique()))

# Mover el modelo a la GPU si está disponible
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

epochs = 10
# Configurar el optimizador y el planificador de tasa de aprendizaje
optimizer = AdamW(model.parameters(), lr=5e-5)  # Prueba diferentes valores de tasa de aprendizaje
total_steps = len(train_dataloader) * epochs
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

# Función de entrenamiento
def train_epoch(model, dataloader, optimizer, scheduler, device):
    model.train()
    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()

        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss

        loss.backward()
        optimizer.step()
        scheduler.step()

# Función de evaluación
def evaluate(model, dataloader, device):
    model.eval()
    correct_predictions = 0
    total_predictions = 0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            _, preds = torch.max(outputs.logits, dim=1)

            correct_predictions += (preds == labels).sum().item()
            total_predictions += labels.size(0)

    return correct_predictions / total_predictions

# Entrenar el modelo

epochs = 10
for epoch in range(epochs):
    print(f"Epoch {epoch + 1}/{epochs}")
    train_epoch(model, train_dataloader, optimizer, scheduler, device)
    accuracy = evaluate(model, test_dataloader, device)
    print(f"Accuracy: {accuracy:.4f}\n")

def get_predictions(model, dataloader, device):
    model.eval()
    predictions = []
    true_labels = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            _, preds = torch.max(outputs.logits, dim=1)

            predictions.extend(preds.tolist())
            true_labels.extend(labels.tolist())

    return predictions, true_labels

predictions, true_labels = get_predictions(model, test_dataloader, device)
print(classification_report(true_labels, predictions))


model_save_path = "trained_intent_model"
model.save_pretrained(model_save_path)
tokenizer.save_pretrained(model_save_path)

from transformers import BertTokenizer, BertForSequenceClassification

model_load_path = "trained_intent_model"
loaded_tokenizer = BertTokenizer.from_pretrained(model_load_path)
loaded_model = BertForSequenceClassification.from_pretrained(model_load_path)

def predict_intent(prompt, model, tokenizer, label_encoder, device):
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
        _, preds = torch.max(outputs.logits, dim=1)

    return label_encoder.inverse_transform(preds.cpu().numpy())[0]

prompt = "EL manejo de las relaciones pasadas y el dejar el apego al pasado es el principal tema que maneja la ultima pelicula de Makoto Shinkay, alejándose un poco del romance que tanto se presentaba en sus ultimas películas"
predicted_intent = predict_intent(prompt, loaded_model, loaded_tokenizer, label_encoder, device)
print(f"Predicted intent: {predicted_intent}")


