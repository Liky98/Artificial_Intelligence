"""""""""
citeseer은 논문에 대한 데이터셋
3312개의 논문
3703개의 고유 단어
(문서 빈도 10 미만 단어 모두제거)

총 6개의 클래스
- Agents
- Ai
- DB
- IR
- ML
- HCI

Content 파일에는 논문에 대한 설명.
- paper_id (고유id)
- word_attributes (어휘의 각단어가 논문에 있는지 1 or 0)
- class_label (논문 클래스 레이블

Cites 파일에는 인용그래프
- ID of cited paper (인용되는 논문의 ID
- ID of citing paper (인용이 포함된 논문의 약자)
방향은 오른쪽에서 왼쪽. paper2 -> paper1
"""""""""

import pandas as pd
import numpy as np

path_citeseer_content = '../Data/citeseer/citeseer/citeseer.content'
path_citeseer_cites = '../Data/citeseer/citeseer/citeseer.cites'
path_cora_content = '../Data/cora/cora/cora.content'
path_cora_cites = '../Data/cora/cora/cora.cites'
path_Pubmed_Dibates_DIRECTED_cites_tab = '../Data/Pubmed-Diabetes/Pubmed-Diabetes/data/Pubmed-Diabetes.DIRECTED.cites.tab'
path_Pubmed_Dibates_GRAPH_pubmed_tab = '../Data/Pubmed-Diabetes/Pubmed-Diabetes/data/Pubmed-Diabetes.GRAPH.pubmed.tab'
path_Pubmed_Dibates_NODE_paper_tab = '../Data/Pubmed-Diabetes/Pubmed-Diabetes/data/Pubmed-Diabetes.NODE.paper.tab'


#%%
citeseer_content = pd.read_csv(path_citeseer_content)
citeseer_content.shape
#%%
citeseer_content = pd.read_excel(path_cora_content+".xlsx")
citeseer_content.shape

