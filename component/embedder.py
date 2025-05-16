import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def init_embedder():
    tokenizer = AutoTokenizer.from_pretrained('./checkpoints/bert-base-uncased')
    model = AutoModel.from_pretrained('./checkpoints/nomic-embed-text-v1', trust_remote_code=True, device_map="auto")
    model.to("cuda")
    model.eval()
    return model, tokenizer

def gen_embs(tokenizer, model, sentences):
    encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')
    encoded_input = {k: v.to("cuda") for k, v in encoded_input.items()}

    with torch.no_grad():
        with torch.autocast(device_type="cuda"):
            model_output = model(**encoded_input)

    embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
    embeddings = F.normalize(embeddings, p=2, dim=1)
    # print(embeddings)
    return embeddings.flatten()

if __name__=="__main__":
    model, tokenizer = init_embedder