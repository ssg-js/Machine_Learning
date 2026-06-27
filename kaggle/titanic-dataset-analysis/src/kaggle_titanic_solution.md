# Kaggle Titanic 생존 예측 풀이 정리

## 1. 문제 개요

Kaggle Titanic 문제는 승객 정보를 바탕으로 생존 여부를 예측하는 **지도학습 기반 이진 분류 문제**다.

- `0`: 사망
- `1`: 생존

학습 데이터인 `train.csv`에는 승객 정보와 정답인 `Survived`가 함께 들어 있다.  
테스트 데이터인 `test.csv`에는 승객 정보만 있고, 모델이 생존 여부를 예측해야 한다.

---

## 2. 프로젝트 구조

```text
titanic/
├── data/
│   ├── train.csv
│   ├── test.csv
│   └── gender_submission.csv
└── src/
    └── main.py
```

- `data/`: Kaggle에서 받은 데이터 저장
- `src/`: Python 소스 코드 저장
- `main.py`: 데이터 분석과 모델 학습 코드

---

## 3. 데이터 불러오기

```python
from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

train_df = pd.read_csv(DATA_DIR / "train.csv")
test_df = pd.read_csv(DATA_DIR / "test.csv")
```

### 코드 설명

```python
Path(__file__).resolve()
```

현재 실행 중인 `main.py`의 절대 경로를 구한다.

```python
.parent.parent
```

`src/main.py`에서 두 단계 상위 폴더로 이동해 프로젝트 루트 경로를 구한다.

```python
pd.read_csv(...)
```

CSV 파일을 pandas의 `DataFrame`으로 변환한다.

---

## 4. 데이터 구조 확인

```python
print("train 크기:", train_df.shape)
print("test 크기:", test_df.shape)

print(train_df.head())
print(train_df.columns)

train_df.info()

print(train_df.isnull().sum())
```

일반적인 데이터 크기는 다음과 같다.

```text
train: (891, 12)
test: (418, 11)
```

`train.csv`에만 정답인 `Survived` 컬럼이 있기 때문에 컬럼 수가 하나 더 많다.

### 주요 컬럼

| 컬럼 | 의미 |
|---|---|
| `PassengerId` | 승객 식별 번호 |
| `Survived` | 생존 여부 |
| `Pclass` | 객실 등급 |
| `Name` | 이름 |
| `Sex` | 성별 |
| `Age` | 나이 |
| `SibSp` | 함께 탑승한 형제·배우자 수 |
| `Parch` | 함께 탑승한 부모·자녀 수 |
| `Ticket` | 티켓 번호 |
| `Fare` | 요금 |
| `Cabin` | 객실 번호 |
| `Embarked` | 탑승 항구 |

### 결측값

주요 결측값은 다음 컬럼에서 확인된다.

- `Age`
- `Cabin`
- `Embarked`
- 테스트 데이터의 `Fare`

---

## 5. 탐색적 데이터 분석

탐색적 데이터 분석, 즉 EDA는 데이터를 모델에 넣기 전에 데이터의 특성과 패턴을 확인하는 과정이다.

### 전체 생존 비율

```python
print(train_df["Survived"].value_counts())
print(train_df["Survived"].value_counts(normalize=True))
```

`value_counts()`는 각 값의 개수를 계산한다.  
`normalize=True`를 사용하면 개수 대신 비율을 반환한다.

---

### 성별 생존율

```python
print(
    train_df.groupby("Sex")["Survived"].mean()
)
```

`Survived`는 `0`과 `1`로 구성되어 있으므로 평균이 곧 생존율이다.

분석 결과 여성의 생존율이 남성보다 높게 나타난다. 따라서 `Sex`는 생존 예측에 중요한 특성으로 볼 수 있다.

---

### 객실 등급별 생존율

```python
print(
    train_df.groupby("Pclass")["Survived"].mean()
)
```

객실 등급별 생존율을 비교하면 1등급 승객의 생존율이 높고, 3등급 승객의 생존율이 낮은 경향을 확인할 수 있다.

---

### 성별과 객실 등급을 함께 분석

```python
print(
    train_df.pivot_table(
        index="Sex",
        columns="Pclass",
        values="Survived",
        aggfunc="mean",
    )
)
```

`pivot_table()`을 이용해 성별과 객실 등급 조합별 생존율을 표 형태로 확인할 수 있다.

---

### 연령대 생성

```python
train_df["AgeGroup"] = pd.cut(
    train_df["Age"],
    bins=[0, 12, 18, 35, 60, 100],
    labels=[
        "Child",
        "Teen",
        "YoungAdult",
        "Adult",
        "Senior",
    ],
)
```

연속적인 나이 값을 구간으로 나누어 연령대별 생존율을 분석한다.

```python
print(
    train_df.groupby(
        "AgeGroup",
        observed=False,
    )["Survived"].mean()
)
```

---

### 가족 규모 생성

```python
train_df["FamilySize"] = (
    train_df["SibSp"]
    + train_df["Parch"]
    + 1
)
```

`SibSp`와 `Parch`를 더하고 본인까지 포함하기 위해 `1`을 추가한다.

```python
print(
    train_df.groupby("FamilySize")["Survived"]
    .agg(["count", "mean"])
)
```

- `count`: 해당 가족 규모의 승객 수
- `mean`: 생존율

---

### 혼자 탑승했는지 여부 생성

```python
train_df["IsAlone"] = (
    train_df["FamilySize"] == 1
).astype(int)
```

- 혼자 탑승: `1`
- 가족과 탑승: `0`

```python
print(
    train_df.groupby("IsAlone")["Survived"].mean()
)
```

---

## 6. 모델 입력값과 정답 분리

첫 번째 모델에서는 다음 컬럼을 사용한다.

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

입력값과 정답을 분리한다.

```python
X = train_df[features]
y = train_df["Survived"]
```

- `X`: 모델에 입력할 승객 정보
- `y`: 모델이 예측해야 할 생존 여부

머신러닝에서는 입력 컬럼을 **특성 또는 Feature**, 정답을 **Target 또는 Label**이라고 한다.

---

## 7. 숫자형과 범주형 컬럼 분리

```python
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
```

`Pclass`는 숫자로 저장되어 있지만, 1등급·2등급·3등급을 나타내는 범주이므로 범주형으로 처리했다.

---

## 8. 결측값 처리

### 숫자형 데이터

```python
from sklearn.impute import SimpleImputer

numeric_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="median"),
        ),
    ]
)
```

숫자형 결측값은 중앙값으로 채운다.

중앙값은 평균보다 극단적인 값의 영향을 적게 받으므로 나이와 요금 같은 데이터에 적용하기 적절하다.

### 범주형 데이터

```python
categorical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="most_frequent"
            ),
        ),
    ]
)
```

범주형 결측값은 가장 자주 등장한 값으로 채운다.

---

## 9. One-Hot Encoding

대부분의 머신러닝 모델은 문자열을 직접 계산할 수 없다.

예를 들어 성별 데이터는 다음과 같다.

```text
male
female
```

One-Hot Encoding을 적용하면 다음처럼 변환된다.

```text
Sex_female  Sex_male
1           0
0           1
```

코드는 다음과 같다.

```python
from sklearn.preprocessing import OneHotEncoder

OneHotEncoder(handle_unknown="ignore")
```

`handle_unknown="ignore"`는 학습 데이터에서 보지 못한 범주가 테스트 데이터에 등장하더라도 오류를 발생시키지 않도록 한다.

---

## 10. 전처리 파이프라인 구성

```python
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


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
```

전처리 과정은 다음과 같다.

```text
숫자형 컬럼
→ 중앙값으로 결측값 처리

범주형 컬럼
→ 최빈값으로 결측값 처리
→ One-Hot Encoding
```

`Pipeline`을 사용하면 여러 전처리 단계를 정해진 순서대로 실행할 수 있다.

---

## 11. 학습 데이터와 검증 데이터 분리

```python
from sklearn.model_selection import train_test_split


X_train, X_valid, y_train, y_valid = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)
```

전체 학습 데이터를 다음처럼 나눈다.

```text
전체 train 데이터
├── 학습 데이터 80%
└── 검증 데이터 20%
```

### 옵션 설명

- `test_size=0.2`: 전체 데이터의 20%를 검증용으로 사용
- `random_state=42`: 실행할 때마다 동일하게 분할
- `stratify=y`: 학습 데이터와 검증 데이터의 생존 비율을 비슷하게 유지

검증 데이터는 모델 학습에 사용하지 않고 모델의 성능을 확인하는 용도로 사용한다.

---

## 12. 로지스틱 회귀 모델

```python
from sklearn.linear_model import LogisticRegression


classifier = LogisticRegression(max_iter=1000)
```

로지스틱 회귀는 이름에 회귀가 들어가지만, 주로 이진 분류 문제에 사용한다.

모델은 승객의 각 특성에 가중치를 부여하고 생존 확률을 계산한다.

```text
생존 점수 =
성별 가중치
+ 객실 등급 가중치
+ 나이 가중치
+ 요금 가중치
+ ...
```

계산한 점수는 시그모이드 함수를 통해 `0~1` 사이의 확률로 변환된다.

기본적으로 생존 확률이 `0.5` 이상이면 `1`, 미만이면 `0`으로 분류한다.

---

## 13. 전처리와 모델 연결

```python
model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "classifier",
            LogisticRegression(max_iter=1000),
        ),
    ]
)
```

데이터는 다음 순서로 처리된다.

```text
원본 데이터
→ 결측값 처리
→ One-Hot Encoding
→ 로지스틱 회귀
→ 생존 여부 예측
```

전처리와 모델을 하나의 파이프라인으로 묶으면 학습 데이터와 검증 데이터에 동일한 전처리 규칙을 적용할 수 있다.

---

## 14. 모델 학습과 예측

```python
model.fit(X_train, y_train)
```

`fit()`은 학습 데이터를 이용해 입력값과 정답 사이의 관계를 학습하는 과정이다.

검증 데이터 예측:

```python
valid_predictions = model.predict(X_valid)
```

예측 결과는 다음처럼 `0` 또는 `1`로 반환된다.

```text
[0, 1, 0, 1, 1, ...]
```

생존 확률을 직접 확인하려면 다음을 사용할 수 있다.

```python
valid_probabilities = model.predict_proba(X_valid)

print(valid_probabilities[:5])
```

각 행은 다음 형식이다.

```text
[사망 확률, 생존 확률]
```

---

## 15. 정확도 평가

```python
from sklearn.metrics import accuracy_score


accuracy = accuracy_score(
    y_valid,
    valid_predictions,
)

print(f"검증 정확도: {accuracy:.4f}")
```

정확도는 다음과 같이 계산한다.

```text
정확히 예측한 데이터 수 / 전체 검증 데이터 수
```

현재 사용한 기본 특성과 로지스틱 회귀 모델에서는 일반적으로 약 `0.78~0.82` 수준의 검증 정확도를 기대할 수 있다.

---

## 16. 전체 코드

```python
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


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

model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "classifier",
            LogisticRegression(max_iter=1000),
        ),
    ]
)

model.fit(X_train, y_train)

valid_predictions = model.predict(X_valid)

accuracy = accuracy_score(
    y_valid,
    valid_predictions,
)

print("학습 데이터 크기:", X_train.shape)
print("검증 데이터 크기:", X_valid.shape)
print(f"검증 정확도: {accuracy:.4f}")
```

---

## 17. 현재 풀이 흐름 정리

```text
1. Titanic 문제를 이진 분류 문제로 정의
2. train.csv와 test.csv 불러오기
3. 데이터 크기, 컬럼, 타입, 결측값 확인
4. 성별·객실 등급·나이·가족 규모별 생존율 분석
5. 모델에 사용할 Feature와 Target 분리
6. 숫자형과 범주형 컬럼 분리
7. 결측값 처리
8. 범주형 데이터 One-Hot Encoding
9. 학습 데이터와 검증 데이터 분리
10. 로지스틱 회귀 모델 학습
11. 검증 데이터 예측
12. 정확도로 성능 평가
```

---

## 18. 다음 개선 방향

현재 모델은 기본적인 베이스라인 모델이다. 이후 다음 특성을 추가하면서 성능 변화를 비교할 수 있다.

- `FamilySize`: 함께 탑승한 가족 규모
- `IsAlone`: 혼자 탑승했는지 여부
- `Title`: 이름에서 추출한 호칭
- `HasCabin`: 객실 정보 존재 여부
- 나이 구간
- 요금 구간

또한 다음 모델과 성능을 비교할 수 있다.

- Decision Tree
- Random Forest
- Gradient Boosting
- XGBoost

새로운 특성이나 모델을 추가할 때마다 검증 정확도를 기록해 실제 성능이 개선되었는지 확인해야 한다.
