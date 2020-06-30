from aifactory.modules import activate, submit

activate('keumdohoon01@gmail.com', 'Robin0101')

task = 10
code   = r"/tf/notebooks/Keum/model/model03_ms.py"
weight = r'/tf/notebooks/Keum/save_model/model02.h5'
result = r'/tf/notebooks/Keum/sub/submission.txt'


submit(task, code, weight, result)

# mkdir /usr/local/lib/python3.6/dist-packages/aifactory
# curl http://3.34.93.88:8080/download/byNameStream/DOC_IMG/20200626155051457_rmvL/> /usr/local/lib/python3.6/dist-packages/aifactory/modules.py
# pip install tqdm requests
# exit

# 업로드 완료 (시각: 2020-06-30 12:39:47.460849). 홈페이지에 코드 제출양식 및 최종 제출 관련 공지가 등록되었으니 확인 부탁드립니다. 감사합니다.: 200