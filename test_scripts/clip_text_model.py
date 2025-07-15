from transformers import CLIPTextModel, CLIPTokenizer, CLIPTextModelWithProjection
import torch

model_name = "openai/clip-vit-large-patch14"
tokenizer = CLIPTokenizer.from_pretrained(model_name)
model = CLIPTextModel.from_pretrained(model_name,cache_dir="./pretrained_models").cuda()
sentence = "groom"
inputs = tokenizer(sentence, return_tensors="pt",padding=True,truncation=True).to('cuda')
with torch.no_grad():
    outputs = model(**inputs)
    embeddings = outputs.last_hidden_state
    print(embeddings.shape)
