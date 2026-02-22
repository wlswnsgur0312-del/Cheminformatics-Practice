```python

import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

print("--- 1. ChEMBL 데이터 불러오는 중... ---")

# 1. CSV 파일 읽기
# low_memory=False 옵션으로 DtypeWarning 경고를 해결합니다.
try:
    df = pd.read_csv('chembl_data.csv', sep=';', low_memory=False)
except FileNotFoundError:
    print("오류: 'chembl_data.csv' 파일이 없습니다. 폴더 위치를 확인해주세요.")
    exit()

# [핵심 수정] 컬럼 이름 정리하기
# 엑셀의 "Standard Value" -> "standard_value"로 자동 변환 (공백 제거, 소문자 변환)
df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')

print(f">> 전체 데이터 개수: {len(df)}개")

# 2. 데이터 전처리 (청소하기)
# SMILES 컬럼 이름이 버전마다 달라서 찾아내는 코드
if 'canonical_smiles' in df.columns:
    smiles_col = 'canonical_smiles'
elif 'smiles' in df.columns:
    smiles_col = 'smiles'
else:
    print(f"오류: SMILES 컬럼을 찾을 수 없습니다. 현재 컬럼: {list(df.columns)}")
    exit()

# 필요한 컬럼만 선택
df = df[[smiles_col, 'standard_value']].dropna()

# 'standard_value'를 숫자로 강제 변환 (문자열이 섞여있으면 에러 나므로)
df['standard_value'] = pd.to_numeric(df['standard_value'], errors='coerce')
df = df.dropna(subset=['standard_value']) # 변환 안 되는 이상한 값 삭제

# 약효 분류 (Labeling): 1000nM 이하면 활성(1), 아니면 비활성(0)
df['Active_Label'] = df['standard_value'].apply(lambda x: 1 if x <= 1000 else 0)

print(f">> 학습에 사용할 유효 데이터: {len(df)}개 (전처리 완료)")

# 3. 특징(Feature) 추출 함수
def get_descriptors(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            return [
                Descriptors.MolWt(mol),        # 분자량
                Descriptors.MolLogP(mol),      # 지질친화도
                Descriptors.NumRotatableBonds(mol), # 회전가능 결합 수
                Descriptors.TPSA(mol)          # 극성 표면적
            ]
    except:
        return None
    return None

print("--- 2. 분자 특징 추출 중 (데이터가 많아 시간이 좀 걸립니다)... ---")
print("    (멈춘 게 아니니 잠시 기다려주세요!)")

# 데이터 전체에 적용 (progress bar가 없어서 그냥 기다려야 합니다)
df['Features'] = df[smiles_col].apply(get_descriptors)

# 계산 실패한 데이터 삭제
df = df.dropna(subset=['Features'])

# 4. AI 학습 준비
X = list(df['Features'])
y = df['Active_Label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"--- 3. AI 모델 학습 시작! (학습 데이터: {len(X_train)}개) ---")
# 나무 100그루 심기 (데이터가 많으니 100개 정도는 되어야 합니다)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 5. 결과 확인
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"\n>>> [최종 결과] AI 정확도: {accuracy*100:.2f}%")
print("축하합니다! 대규모 데이터를 이용한 QSAR 모델링에 성공하셨습니다.")
