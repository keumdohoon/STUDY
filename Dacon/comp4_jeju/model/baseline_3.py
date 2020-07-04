import pandas as pd
import numpy as np
                    
data = pd.read_csv('./dacon/comp4_jeju/data/201901-202003.csv')
data = data.fillna('')

df = data.copy()
df = df[['REG_YYMM', 'CARD_SIDO_NM', 'STD_CLSS_NM', 'AMT']]
df = df.groupby(['REG_YYMM', 'CARD_SIDO_NM', 'STD_CLSS_NM']).sum().reset_index(drop=False)
df = df.loc[df['REG_YYMM']==202003]
df = df[['CARD_SIDO_NM', 'STD_CLSS_NM', 'AMT']]

submission = pd.read_csv('./dacon/comp4_jeju/data/submission.csv', index_col=0)
submission = submission.loc[submission['REG_YYMM']==202004]
submission = submission[['CARD_SIDO_NM', 'STD_CLSS_NM']]
submission = submission.merge(df, left_on=['CARD_SIDO_NM', 'STD_CLSS_NM'], right_on=['CARD_SIDO_NM', 'STD_CLSS_NM'], how='left')
submission = submission.fillna(0)
AMT = list(submission['AMT'])*2

submission = pd.read_csv('./dacon/comp4_jeju/data/submission.csv', index_col=0)
submission['AMT'] = AMT
submission.to_csv('./dacon/comp4_jeju/sub/submission_3.csv', encoding='utf-8-sig')
submission.head()
print('complete')