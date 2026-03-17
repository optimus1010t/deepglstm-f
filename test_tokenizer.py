from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")
out = tokenizer(["ACDEF"], truncation=True, max_length=1024, padding='max_length', return_tensors='pt')
print(out['input_ids'].shape)
print(out['attention_mask'].shape)
