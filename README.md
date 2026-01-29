# ETRI_HumanUnderstanding_AI_Challenge
라이프로그 데이터를 활용하여 수면 품질 및 상태 예측을 위한 분류 모델을 개발

# 주제
라이프로그 데이터를 활용한 수면 품질 및 상태 예측

# 평가지표
Macro F1-Score

# 사용기술
- Python
- LightGBM/CatBoost/XGBoost
- Optuna
- SMOTE
- Ensemble  
    
# 데이터 전처리 : 		 		   
1. 시간대 설정
- 하루가 넘어가도 전날의 수면상태이므로 다음날 07시까지는 전날 수면상태로 조정      

2. 변수별 전처리
- 각 변수마다의 특성을 고려하여 mean, max, min, std, cov 등 기초통계량과 고유값, 다양성을 파생변수로 생성
- 각 변수마다 변수마다 고유값, 다양성, cov 값을 조합하여 불안전성이라는 공통적인 파생변수 생성
   
3. 종속변수별 시간대 설정
```
def cut_time(hour):
    if 22 <= hour or 0 <= hour < 6:
        return 'S1_S2'
def cut_time(hour):
    if 22 <= hour or 0 <= hour < 1:
        return 'S3'
def cut_time(hour):
    if 22 <= hour or 0 <= hour < 7:
        return 'Q1'
    elif 7 <= hour < 22:
        return 'Q2_Q3'
 ```
* 종속변수마다 특성을 고려해 각각의 시간대를 설정    

# 모델링 :    
1. 독립변수 설정
- 종속변수별에 따른 독립변수 설정     

2. train_test_split

3. 분류 모델 구축
- 종속변수의 특성에 따른 이진/다중 분류 모델 구축

5. 데이터 증강 + 하이퍼파라미터 튜닝
- LGBMClassifier, XGBoostClassifier, CatBoostClassifier 각각의 모델에 대한 최적의 파라미터 탐색     

4. 모델별 이진/다중 분류에 대한  

5. 앙상블

- 하이퍼파라미터 튜닝 모델을 결합하여 최적의 결과를 생성
- 앙상블 방법:
  * 예측이 완료된 결과값 조합하여 가장 많이 등장한 값을 최종 결과값으로 선정    

# 검증, 성능 평가 - (Macro F1-score)    
* Macro F1-score를 활용하여 종속변수별 결과값이 실제 테스트 데이터(test)와 얼마나 일치하는지 평가한 후 전체 종속변수 값인 6으로 나눠 최종 산출
