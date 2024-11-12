from transformers import BertTokenizer, BertModel
import torch
from sklearn.metrics.pairwise import cosine_similarity

# 載入BERT模型和分詞器
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
model = BertModel.from_pretrained('bert-base-chinese')

# 預處理文本
text1 = "這是一個示例句子。"
text2 = "這是另一個示例句子。"
tokens1 = tokenizer.tokenize(tokenizer.decode(tokenizer.encode(text1)))
tokens2 = tokenizer.tokenize(tokenizer.decode(tokenizer.encode(text2)))

# 將文本轉換成BERT的輸入格式
input_ids1 = tokenizer.encode(text1, return_tensors='pt')
input_ids2 = tokenizer.encode(text2, return_tensors='pt')

# 獲取BERT模型的輸出
output1 = model(input_ids1)
output2 = model(input_ids2)

# 取CLS位置的隱藏狀態作為句子向量
sentence_vector1 = output1.last_hidden_state.mean(dim=1).detach().numpy()
sentence_vector2 = output2.last_hidden_state.mean(dim=1).detach().numpy()

# 計算餘弦相似度
similarity = cosine_similarity(sentence_vector1, sentence_vector2)[0][0]
print("相似度:", similarity)