# Kaggle Titanic 모델 비교 풀이 정리

## 1. 문제 정의

Kaggle Titanic 문제는 승객 정보를 바탕으로 생존 여부를 예측하는 **지도학습 기반 이진 분류 문제**다.

- `0`: 사망
- `1`: 생존

`train.csv`에는 승객 정보와 정답인 `Survived`가 함께 들어 있고, `test.csv`에는 승객 정보만 들어 있다.

현재 단계에서는 `train.csv`를 학습 데이터와 검증 데이터로 나누고, 다음 세 가지 모델의 성능을 비교했다.

1. 로지스틱 회귀
2. 결정 트리
3. 랜덤 포레스트

---

## 2. 사용 데이터

```text
train.csv
→ 승객 정보 + 정답 Survived

test.csv
→ 승객 정보만 존재
```

현재까지는 `train.csv`만 사용해 모델을 학습하고 검증했다.

---

## 3. 사용한 특성

첫 번째 베이스라인 모델에서는 다음 7개 컬럼을 사용했다.

```python
features = [
    "Pclass",
    "Sex",
    "Age",
    "SibSp",
    "Parch",
    "Fare",
    "Embarked",
]
```

| 특성 | 의미 |
|---|---|
| `Pclass` | 객실 등급 |
| `Sex` | 성별 |
| `Age` | 나이 |
| `SibSp` | 함께 탑승한 형제·자매·배우자 수 |
| `Parch` | 함께 탑승한 부모·자녀 수 |
| `Fare` | 탑승 요금 |
| `Embarked` | 탑승 항구 |

입력 데이터와 정답을 분리했다.

```python
X = train_df[features]
y = train_df["Survived"]
```

- `X`: 모델이 참고할 승객 정보
- `y`: 실제 생존 여부

머신러닝에서는 입력값을 **Feature**, 정답을 **Target 또는 Label**이라고 한다.

---

## 4. 공통 전처리

세 모델 모두 동일한 전처리를 사용했다.

동일한 전처리와 동일한 검증 데이터를 사용해야 모델 간 성능을 공정하게 비교할 수 있다.

### 4.1 숫자형 컬럼

```python
numeric_features = [
    "Age",
    "SibSp",
    "Parch",
    "Fare",
]
```

숫자형 컬럼의 결측값은 중앙값으로 채웠다.

```python
numeric_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="median"),
        ),
    ]
)
```

중앙값은 평균보다 극단적인 값의 영향을 적게 받는다.

예를 들어 일부 승객의 요금이 매우 높더라도 중앙값은 상대적으로 안정적이다.

---

### 4.2 범주형 컬럼

```python
categorical_features = [
    "Pclass",
    "Sex",
    "Embarked",
]
```

`Pclass`는 숫자로 저장되어 있지만 실제 의미는 1등급, 2등급, 3등급이라는 범주에 가깝기 때문에 범주형으로 처리했다.

범주형 결측값은 가장 자주 등장한 값으로 채우고, One-Hot Encoding을 적용했다.

```python
categorical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="most_frequent"
            ),
        ),
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore"
            ),
        ),
    ]
)
```

예를 들어 성별은 다음처럼 숫자 컬럼으로 변환된다.

```text
Sex_female  Sex_male
1           0
0           1
```

`handle_unknown="ignore"`는 학습 과정에서 보지 못한 범주가 검증 또는 테스트 데이터에 등장해도 오류를 발생시키지 않도록 한다.

---

### 4.3 컬럼별 전처리 결합

```python
preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            numeric_pipeline,
            numeric_features,
        ),
        (
            "categorical",
            categorical_pipeline,
            categorical_features,
        ),
    ]
)
```

전체 전처리 과정은 다음과 같다.

```text
숫자형 컬럼
→ 중앙값으로 결측값 처리

범주형 컬럼
→ 최빈값으로 결측값 처리
→ One-Hot Encoding
```

---

## 5. 학습 데이터와 검증 데이터 분리

```python
X_train, X_valid, y_train, y_valid = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)
```

전체 학습 데이터를 다음처럼 나눴다.

```text
전체 891명
├── 학습 데이터 약 80%: 712명
└── 검증 데이터 약 20%: 179명
```

### 옵션 설명

- `test_size=0.2`: 전체 데이터 중 20%를 검증용으로 사용
- `random_state=42`: 실행할 때마다 동일하게 데이터를 분할
- `stratify=y`: 학습 데이터와 검증 데이터의 생존 비율을 비슷하게 유지

검증 데이터는 모델 학습에 사용하지 않고, 학습이 끝난 모델의 일반화 성능을 확인하는 용도로 사용한다.

---

## 6. 모델 1: 로지스틱 회귀

```python
LogisticRegression(max_iter=1000)
```

로지스틱 회귀는 이름에 회귀가 들어가지만, 주로 이진 분류 문제에 사용한다.

모델은 각 특성에 가중치를 부여해 생존 점수를 계산한다.

```text
생존 점수 =
성별 가중치
+ 객실 등급 가중치
+ 나이 가중치
+ 요금 가중치
+ ...
```

이 점수를 시그모이드 함수를 이용해 `0~1` 사이의 생존 확률로 변환한다.

```text
생존 확률 0.7 → 생존
생존 확률 0.3 → 사망
```

### 장점

- 구조가 단순하다.
- 학습 속도가 빠르다.
- 각 특성의 영향을 비교적 해석하기 쉽다.
- 베이스라인 모델로 적합하다.

### 한계

- 복잡한 비선형 관계를 표현하는 데 제한이 있다.
- 여러 특성이 결합된 복잡한 조건을 학습하는 데 한계가 있다.

### 모델 구성

```python
logistic_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "classifier",
            LogisticRegression(max_iter=1000),
        ),
    ]
)
```

---

## 7. 모델 2: 결정 트리

```python
DecisionTreeClassifier(
    max_depth=4,
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=42,
)
```

결정 트리는 데이터를 질문으로 계속 나누면서 최종 결과를 예측한다.

개념적인 구조는 다음과 같다.

```text
성별이 여성인가?
├── 예 → 객실 등급 확인
└── 아니오 → 나이 확인
```

각 단계에서는 생존자와 사망자를 가장 잘 구분할 수 있는 조건을 선택한다.

### 주요 하이퍼파라미터

#### `max_depth=4`

트리의 최대 깊이를 4단계로 제한한다.

트리가 너무 깊어지면 학습 데이터를 지나치게 외우는 과적합이 발생할 수 있다.

#### `min_samples_split=10`

현재 노드를 추가로 나누려면 최소 10개의 데이터가 있어야 한다.

#### `min_samples_leaf=5`

최종 리프 노드에는 최소 5개의 데이터가 남아야 한다.

너무 적은 데이터만으로 예측 규칙을 만드는 것을 방지한다.

#### `random_state=42`

난수를 고정해 실행할 때마다 같은 결과가 나오도록 한다.

### 장점

- 판단 과정을 사람이 이해하기 쉽다.
- 비선형 관계를 학습할 수 있다.
- 특성 간 복합 조건을 자동으로 찾을 수 있다.

### 한계

- 트리가 너무 깊어지면 과적합되기 쉽다.
- 데이터가 조금만 달라져도 트리 구조가 크게 달라질 수 있다.

### 모델 구성

```python
decision_tree_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "classifier",
            DecisionTreeClassifier(
                max_depth=4,
                min_samples_split=10,
                min_samples_leaf=5,
                random_state=42,
            ),
        ),
    ]
)
```

---

## 8. 모델 3: 랜덤 포레스트

```python
RandomForestClassifier(
    n_estimators=300,
    max_depth=6,
    min_samples_split=10,
    min_samples_leaf=3,
    random_state=42,
    n_jobs=-1,
)
```

랜덤 포레스트는 여러 개의 결정 트리를 만든 뒤, 각 트리의 결과를 종합하는 모델이다.

```text
트리 1 → 생존
트리 2 → 사망
트리 3 → 생존
트리 4 → 생존

최종 결과 → 생존
```

여러 모델의 예측을 결합하는 방식을 **앙상블**이라고 한다.

각 트리는 서로 다른 데이터 일부와 특성 일부를 이용해 학습한다. 따라서 한 개의 결정 트리가 만든 잘못된 규칙이 전체 결과를 지배할 가능성이 줄어든다.

### 주요 하이퍼파라미터

#### `n_estimators=300`

결정 트리를 300개 생성한다.

트리 수가 많을수록 예측이 안정될 수 있지만 학습 비용도 증가한다.

#### `max_depth=6`

각 결정 트리의 최대 깊이를 6단계로 제한한다.

#### `min_samples_split=10`

노드를 추가로 나누려면 최소 10개의 데이터가 필요하다.

#### `min_samples_leaf=3`

각 리프 노드에는 최소 3개의 데이터가 남아야 한다.

#### `n_jobs=-1`

사용 가능한 CPU 코어를 모두 사용해 여러 트리를 병렬로 학습한다.

#### `random_state=42`

난수를 고정해 같은 조건에서 동일한 결과를 재현한다.

### 장점

- 결정 트리 하나보다 예측이 안정적이다.
- 복잡한 비선형 관계를 잘 학습한다.
- 특성 간 조합을 자동으로 찾을 수 있다.
- 결정 트리 하나보다 과적합에 상대적으로 강하다.

### 한계

- 하나의 결정 트리보다 내부 판단 과정을 해석하기 어렵다.
- 트리 수가 많으면 학습 및 예측 비용이 증가한다.
- 하이퍼파라미터에 따라 성능이 달라질 수 있다.

### 모델 구성

```python
random_forest_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "classifier",
            RandomForestClassifier(
                n_estimators=300,
                max_depth=6,
                min_samples_split=10,
                min_samples_leaf=3,
                random_state=42,
                n_jobs=-1,
            ),
        ),
    ]
)
```

---

## 9. 모델 학습과 검증

세 모델 모두 동일한 방식으로 학습하고 검증했다.

```python
model.fit(X_train, y_train)

train_predictions = model.predict(X_train)
valid_predictions = model.predict(X_valid)
```

- `fit()`: 학습 데이터로 모델 학습
- `predict()`: 입력 데이터의 생존 여부 예측

훈련 정확도와 검증 정확도를 각각 계산했다.

```python
train_accuracy = accuracy_score(
    y_train,
    train_predictions,
)

valid_accuracy = accuracy_score(
    y_valid,
    valid_predictions,
)
```

정확도는 다음과 같이 계산한다.

```text
정확히 예측한 데이터 수 / 전체 데이터 수
```

검증 정확도가 `0.81`이라면 검증 데이터의 약 81%를 맞혔다는 뜻이다.

---

## 10. 훈련 정확도와 검증 정확도를 함께 보는 이유

훈련 정확도만 높다고 좋은 모델은 아니다.

예를 들어 다음과 같은 결과가 나올 수 있다.

```text
훈련 정확도: 0.99
검증 정확도: 0.78
```

이 경우 모델이 학습 데이터를 거의 외웠지만 새로운 데이터에서는 성능이 낮은 **과적합** 상태일 수 있다.

반면 다음처럼 두 정확도가 비슷하다면 비교적 안정적이라고 판단할 수 있다.

```text
훈련 정확도: 0.84
검증 정확도: 0.81
```

---

## 11. 세 모델 비교 코드

```python
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

train_df = pd.read_csv(DATA_DIR / "train.csv")
test_df = pd.read_csv(DATA_DIR / "test.csv")

features = [
    "Pclass",
    "Sex",
    "Age",
    "SibSp",
    "Parch",
    "Fare",
    "Embarked",
]

X = train_df[features]
y = train_df["Survived"]

numeric_features = [
    "Age",
    "SibSp",
    "Parch",
    "Fare",
]

categorical_features = [
    "Pclass",
    "Sex",
    "Embarked",
]

numeric_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="median"),
        ),
    ]
)

categorical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="most_frequent"
            ),
        ),
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore"
            ),
        ),
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            numeric_pipeline,
            numeric_features,
        ),
        (
            "categorical",
            categorical_pipeline,
            categorical_features,
        ),
    ]
)

X_train, X_valid, y_train, y_valid = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

classifiers = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000
    ),
    "Decision Tree": DecisionTreeClassifier(
        max_depth=4,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42,
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=300,
        max_depth=6,
        min_samples_split=10,
        min_samples_leaf=3,
        random_state=42,
        n_jobs=-1,
    ),
}

results = {}

for name, classifier in classifiers.items():
    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )

    model.fit(X_train, y_train)

    train_predictions = model.predict(X_train)
    valid_predictions = model.predict(X_valid)

    train_accuracy = accuracy_score(
        y_train,
        train_predictions,
    )

    valid_accuracy = accuracy_score(
        y_valid,
        valid_predictions,
    )

    results[name] = valid_accuracy

    print(
        f"{name} "
        f"| train: {train_accuracy:.4f} "
        f"| validation: {valid_accuracy:.4f}"
    )
```

---

## 12. 현재 결과 해석

모델별 결과는 실행 환경과 설정에 따라 달라질 수 있지만 대략 다음과 같은 형태로 비교할 수 있다.

| 모델 | 검증 정확도 예시 |
|---|---:|
| 로지스틱 회귀 | 약 0.79 |
| 결정 트리 | 약 0.78 |
| 랜덤 포레스트 | 약 0.81 |

랜덤 포레스트의 검증 정확도가 `0.81`이라면 타이타닉 입문 문제의 베이스라인으로는 괜찮은 수준이다.

전체 승객을 모두 사망으로 예측해도 약 61~62% 수준의 정확도가 나오기 때문에, `0.81`은 단순 기준보다 확실히 높은 결과다.

다만 이 결과만으로 랜덤 포레스트가 항상 최적이라고 단정할 수는 없다.

성능은 다음 요소에 따라 달라질 수 있다.

- 학습 데이터와 검증 데이터의 분할
- 사용하는 Feature
- 결측값 처리 방식
- 모델 하이퍼파라미터
- 새로운 Feature 생성 여부

---

## 13. 세 모델 비교

### 로지스틱 회귀

```text
각 특성에 가중치를 부여
→ 생존 확률 계산
→ 기준 확률에 따라 분류
```

- 빠르고 단순함
- 해석이 쉬움
- 베이스라인으로 적합
- 복잡한 관계에는 한계가 있음

### 결정 트리

```text
조건 질문을 반복
→ 데이터를 여러 그룹으로 분리
→ 최종 생존 여부 결정
```

- 비선형 관계 학습 가능
- 판단 구조를 이해하기 쉬움
- 과적합 위험이 큼

### 랜덤 포레스트

```text
여러 결정 트리 생성
→ 각 트리의 결과를 종합
→ 최종 예측
```

- 결정 트리 하나보다 안정적
- 복잡한 관계 학습 가능
- 과적합에 상대적으로 강함
- 내부 구조 해석은 어려움

---

## 14. 현재까지 진행한 전체 흐름

```text
1. Titanic 문제를 이진 분류 문제로 정의
2. train.csv와 test.csv 불러오기
3. 데이터 구조와 결측값 확인
4. 입력 Feature와 정답 Target 분리
5. 숫자형과 범주형 컬럼 분리
6. 숫자형 결측값을 중앙값으로 처리
7. 범주형 결측값을 최빈값으로 처리
8. 범주형 데이터에 One-Hot Encoding 적용
9. train 데이터를 학습용과 검증용으로 분리
10. 로지스틱 회귀 모델 학습 및 검증
11. 결정 트리 모델 학습 및 검증
12. 랜덤 포레스트 모델 학습 및 검증
13. 훈련 정확도와 검증 정확도 비교
14. 동일한 조건에서 세 모델 성능 비교
```

---

## 15. 현재 상태와 다음 단계

현재는 세 가지 모델을 만들고 검증 정확도를 비교한 상태다.

아직 Kaggle 제출용 `submission.csv`는 생성하지 않았다.

다음 단계에서는 다음 작업을 진행할 수 있다.

1. 교차 검증을 통해 모델 성능을 더 안정적으로 비교
2. `FamilySize`, `IsAlone`, `Title` 등의 Feature 추가
3. 하이퍼파라미터 조정
4. 가장 적절한 모델 선택
5. 전체 `train.csv`로 최종 모델 재학습
6. `test.csv` 예측
7. Kaggle 제출 파일 생성
