import spacy

spacy_en = spacy.load('en') # 영어 토큰화(tokenization)
spacy_de = spacy.load('de') # 독일어 토큰화(tokenization)

tokenized = spacy_en.tokenizer("I am a graduate student.")
for i, token in enumerate(tokenized):
    print(f"인덱스 {i}: {token.text}")

