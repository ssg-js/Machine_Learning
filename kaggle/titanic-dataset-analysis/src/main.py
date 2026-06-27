from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

train_df = pd.read_csv(DATA_DIR / "train.csv")
test_df = pd.read_csv(DATA_DIR / "test.csv")

print(train_df.head())

print("\n[train 데이터 크기]")
print(train_df.shape)

print("\n[test 데이터 크기]")
print(test_df.shape)

print("\n[train 데이터 앞 5개]")
print(train_df.head())

print("\n[컬럼 목록]")
print(train_df.columns)

print("\n[데이터 타입과 결측값]")
train_df.info()

print("\n[컬럼별 결측값 개수]")
print(train_df.isnull().sum())

print("\n[생존 여부 개수]")
print(train_df["Survived"].value_counts())

print("\n[생존 여부 비율]")
print(train_df["Survived"].value_counts(normalize=True))

print("\n[성별 생존율]")
print(train_df.groupby("Sex")["Survived"].mean())

print("\n[객실 등급별 생존율]")
print(train_df.groupby("Pclass")["Survived"].mean())

print("\n[성별·객실 등급별 생존율]")
print(train_df.groupby(["Sex", "Pclass"])["Survived"].mean())

print("\n[성별·객실 등급별 생존율 표]")
print(
    train_df.pivot_table(
        index="Sex",
        columns="Pclass",
        values="Survived",
        aggfunc="mean",
    )
)

print("\n[연령대별 생존율]")
train_df["AgeGroup"] = pd.cut(
    train_df["Age"],
    bins=[0, 12, 18, 35, 60, 100],
    labels=["Child", "Teen", "YoungAdult", "Adult", "Senior"],
)
print(
    train_df.groupby(
        "AgeGroup",
        observed=False,
    )["Survived"].mean()
)

print("\n[가족 규모별 생존율]")
train_df["FamilySize"] = (
    train_df["SibSp"]
    + train_df["Parch"]
    + 1
)
print(
    train_df.groupby("FamilySize")["Survived"]
    .agg(["count", "mean"])
)

print("\n[혼자 탑승 여부별 생존율]")
train_df["IsAlone"] = (
    train_df["FamilySize"] == 1
).astype(int)
print(train_df.groupby("IsAlone")["Survived"].mean())