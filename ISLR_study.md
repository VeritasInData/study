# 통계 스터디(ISLR)

작성 일시: 2023년 10월 23일 오후 10:33

# Statistical Learning

supervised

unsupervised

# Data Examples

Wage Data

Stock Market Data

Gene Expression Data

**2.1 Statistical Learning (통계학습)**

**1) 통계학습의 목적**

X : predictors, independent variables, features, variables

Y : response, dependent variable

Y = f(X) + $\epsilon$(error term)

f: systematic information that X provides about Y

통계학습(statistical learning)은 f를 추정하기 위한 일련의 접근법.

X를 기반으로 Y를 예측하는 모형을 만들 때, 예측모형 f를 추정하는 일련의 기법.

예측과 추론 (prediction & inference)

- `prediction`: 설명변수를 사용하여 종속(반응 response)변수의 값을 예측하기 원하는 경우
    
    Y, Y_hat
    
    accuracy of Y_hat:
    
    실제 관측 결과와 예측값에는 오차가 발생.
    
    표본Y의  정확성은 다음 2가지 요소에 달렸다.
    
    1) **reducible error** (축소가능한 오차)
    
    모델을 통해 줄일 수 있는 오차. 모델 자체의 한계나 잘못된 가정으로 인한 모델 파라미터의 부정확성과 관련.
    
    2) **irreducible error** (축소불가능한 오차)
    
    모델로 완전히 제거할 수 없는 오차. 데이터 자체의 본질적인 불확실성과 관련.
    
    (cannot be predicted using X)
    
    모델로는 완전히 제거할 수 없기에 항상 존재
    
    The quantity $\epsilon$ may also contain unmeasurable variation.
    
    ⇒ reducible error를 최소화 하여 모델의 성능을 최적화하는 것이 목표.
    
- `inference`: 설명변수와 종속변수 사이의 인과관계를 파악하고자하는 경우
    
    Which predictors may have apositive relationship with Y?
    
    통계학습의 목적은 예측이 될 수도, 추론이 될 수도, 둘다로 설정될 수 있다.
    

**2) 모델의 추정방법**

- `Parametric Methods (모수적 방법)`
    
    two-step model-based approach
    
    <step 1> 
    
    make an assumption about the functional form of f (f함수의 형태에 대한 가정)
    
    <step 2> 
    
    after a model has been selected, procedure that uses the training data to fit or train the model (훈련 데이터를 통해 모델을 훈련시키고 파라미터 집합을 추정)
    
    most common approach: least squares
    
    - 장점
        - simplifies the problem of estimating f
    - 단점 (potential disadvantage)
        - not match the true
        - overfitting the data (follow the errors or noise too closely)
- `Non-parametric Methos (비모수적 방법)`
    
    f함수의 형태에 대해 명시적 가정 없이 f를 추정 (no assumption about the form of f is made)
    
    as close to the data points as possible without being too rough or wiggly.
    
    a very large number of observations is required.
    

**3) Trade-Off between `Prediction Accurracy` and `Model Interpretability`**

(예측 정확도와 모델 해석력 사이의 균형)

이 둘 관계는 trade-off 관계

- 추론이 목적일 때 덜 유연한 방법 (restrictive models) 사용
    
    when inference is goal, the linear model may be a good choice since it will be quite easy to understand the relationship between Y and X1, X2, … Xp
    
    if we are mainly interested in inference, then restrictive models are much more interpretable.
    
- 예측이 목적일 때 유연한 모델 (flexible models) 이 더 적합

**4) Supervised-learning & Unsupervised-learning**

반응 변수의 존재 유무에 따라 나뉨

- `Supervised Learning (지도학습)`
    
    특정 입력(input)에 따라 올바른 반응변수가 있는 데이터 집합이 주어질 때.
    
    알고자 하는 변수(Y)와 이를 설명할 수 있는 변수(X) 의 pair 로 기반으로 학습을 진행.
    
    일반적으로 학습용 데이터 집합(Train set) 와 검증용 데이터 집합(Test set)을 나누어 학습용 데이터를 통해 학습된 function으로 Test set의 Y를 예측하는 과정을 진행
    
    which can be used for mapping new examples.
    
    대부분의 예측(Prediction) 목적을 가진 학습을 위해 지도학습이 활용.
    

- `Unsupervised Learning (비지도학습)`
    
    반응변수가 없는 데이터 집합이 주어질 때 학습하는 법.
    
    검증용 데이터(test data)를 따로 두지 않은 머신러닝 문제의 종류.
    
    관측치 간의 상관관계를 이해하고자 할때, 서브그룹을 발견하고 분류(클러스터링)하기 위해.
    
    데이터 내에 잠재되어 있는 공통적인 특징 혹은 패턴을 확인하고 새로운 데이터에 대해 그러한 특징 혹은 패턴이 있는지에 대해 반응하는 모델링.
    

**5) Regression problem & Classification problem** (**회귀와 분류문제**)

- Regression (회귀)
    
    반응변수가 수치형(양적) 변수일 때 사용하는 통계학습 방법
    

- Classification (분류)
    
    반응변수가 범주형(질적) 변수일 때 사용하는 통계학습 방
    

2.2 Assessing model accuracy (모델 정확도 평가)

1) Measuring the Quality of Fit (적합품질의 측정)

**MSE: mean squared error (평균 제곱 오차)**

2) 편향-분산 trade-off

- 분산
    
    
- 편향

3) The classification setting

- The Bayes Classifier (베이즈 분류)
- K-Nearest Neighbors (K-최근접 이웃)

**3 Linear Regression (선형회귀)**

**3.1 Simple Linear Regression (단순선형회귀)**

straightforward approach for predicting a quantitative response Y on the basis of a single predictor variable X.,

최소제곱법(the least squares)을 이용하여 회귀직선 도출

잔차제곱합=RSS=Residual Sum of Squares

- 추정된 회귀계수의 정확성 검정