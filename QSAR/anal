```python

import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
from sklearn.ensemble import RandomForestClassifier # AI 모델 (랜덤 포레스트)
from sklearn.model_selection import train_test_split # 데이터 나누기 (공부용/시험용)
from sklearn.metrics import accuracy_score # 채점하기

# 1. 데이터 준비 (가상의 약물 데이터 10개)
# 실제 연구에선 수천 개를 씁니다. 원리 이해를 위해 작게 만들었습니다.
data = {
    'SMILES': [
        'CC(=O)Oc1ccccc1C(=O)O', # Aspirin (잘 녹음)
        'CN1C=NC2=C1C(=O)N(C(=O)N2C)C', # Caffeine (잘 녹음)
        'CC(=O)Nc1ccc(O)cc1', # Tylenol (잘 녹음)
        'OC[C@H](O)[C@H]1OC(=O)C(O)=C1O', # Vitamin C (아주 잘 녹음)
        'CCO', # Ethanol (잘 녹음)
        'CCCCCCCCCCCCCCCC', # Hexadecane (기름 덩어리 -> 안 녹음)
        'c1ccccc1c1ccccc1', # Biphenyl (안 녹음)
        'CC(C)Cc1ccc(cc1)[C@@H](C)C(=O)O', # Ibuprofen (잘 안 녹음)
        'COc1ccc2cc(c(cc2c1)OC)C(=O)O', # Naproxen (잘 안 녹음)
        'CCCCCCCCCC(=O)O' # Fatty acid (지방산 -> 안 녹음)
    ],
    # 1: 잘 녹음 (Soluble), 0: 안 녹음 (Insoluble)
    'Solubility_Label': [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
}

df = pd.DataFrame(data)

print("--- 1. AI 학습 준비 중... ---")

# 2. 특징(Feature) 추출 함수
# AI에게 "이건 아스피린이야"라고 말하면 못 알아듣습니다.
# "이건 분자량이 180이고, LogP가 1.3인 녀석이야"라고 숫자로 알려줘야 합니다.
def get_descriptors(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        return [
            Descriptors.MolWt(mol),        # 무게
            Descriptors.MolLogP(mol),      # 기름기 (제일 중요!)
            Descriptors.NumRotatableBonds(mol), # 유연성
            Descriptors.TPSA(mol)          # 표면적
        ]
    return [0,0,0,0]

# 모든 약물의 특징을 계산해서 리스트로 만듭니다.
df['Features'] = df['SMILES'].apply(get_descriptors)

# 학습하기 좋게 데이터 다듬기
X = list(df['Features']) # 문제지 (특징들)
y = df['Solubility_Label'] # 정답지 (녹는다/안녹는다)

# 3. 데이터 나누기 (8문제는 공부용, 2문제는 수능시험용)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. AI 모델 생성 및 학습 (Random Forest)
model = RandomForestClassifier(n_estimators=10, random_state=42) #확률 고정
model.fit(X_train, y_train) # "공부 시작해!"
print(">> AI 모델 학습 완료!")

# 5. 실전 테스트
print("\n--- 2. 새로운 약물 예측해보기 ---")

# 테스트용 미지 의 약물: Cholesterol (콜레스테롤 - 기름 그 자체)
mystery_drug_smiles = "CC(C)CCCC(C)C1CCC2C3CC=C4CC(O)CCC4(C)C3CCC12C" 
mystery_features = [get_descriptors(mystery_drug_smiles)]

prediction = model.predict(mystery_features)
probability = model.predict_proba(mystery_features)

print(f"테스트 약물: Cholesterol")
print(f"특징값 (MW, LogP 등): {mystery_features[0]}")

if prediction[0] == 1:
    print(f">> AI 예측: '이 약은 물에 잘 녹습니다.' (확률: {probability[0][1]*100:.1f}%)")
else:
    print(f">> AI 예측: '이 약은 물에 잘 안 녹습니다.' (확률: {probability[0][0]*100:.1f}%)")

