from sentence_transformers import SentenceTransformer
model = SentenceTransformer('multi-qa-mpnet-base-dot-v1')
import numpy as np

#Our sentences we like to encode
sentences = [
    '1 june',
    '1 june : a visitors &quot; day',
    '1 june : a visitors &quot; day at the ecb &apos; s eurotower',
    '1 june : a visitors &quot; day at the ecb &apos; s eurotower premises back to top',
    '''1 june : a visitors &quot; day at the ecb &apos; s eurotower premises back to top
meanwhile , the restructuring of corporate balance sheets , together with costcutting efforts , contributed to a strong rebound in corporate profit growth , while the banking sector reduced the level of outstanding nonperforming loans .
across the maturity spectrum , interest rates in the euro area are very low by historical standards , in both nominal and real terms , and thus lend ongoing support to economic activity .
in the cpi , with a tolerance margin of ± 1 percentage point .
in the case of sales , for the time necessary to reinvest in transferable securities and / or in other financial assets provided for by this directive ;
table 7 list of off @-@ shore centres for the ecb geographical breakdown for quarterly balance of payments flows and annual international investment position data iso codes eurostat + oecd offshore financial centers
marketable equity instruments market price
25 / 02 / 2009''']

#Sentences are encoded by calling model.encode()
embeddings = model.encode(sentences)

#Print the embeddings
for sentence, embedding in zip(sentences, embeddings):
    print("Sentence:", sentence)
    print("Norm:", np.linalg.norm(embedding))
    print("")