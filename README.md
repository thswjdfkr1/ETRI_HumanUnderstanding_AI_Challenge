# ETRI_HumanUnderstanding_AI_Challenge
라이프로그 데이터를 활용하여 수면 품질 및 상태 예측을 위한 분류 모델을 개발

# 주제
라이프로그 데이터를 활용한 수면 품질 및 상태 예측

# 평가지표
Macro F1-Score

# 데이터 전처리 : 		 		   
1. 시간대 설정
```
def sleep_lifelog(df):
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['sleep_date'] = np.where(
        df['timestamp'].dt.hour < 7,
        df['timestamp'].dt.date,
        df['timestamp'].dt.date + pd.Timedelta(days=1)
    )
    df['lifelog_date'] = pd.to_datetime(df['sleep_date']) - pd.Timedelta(days=1)

    return df
```
* 하루가 넘어가도 전날의 수면상태이므로 다음날 07시까지는 전날 수면상태로 조정      

2. 변수별 전처리
   각 변수마다의 특성을 고려하여 mean, max, min, std, cov 등 기초통계량과 고유값, 다양성을 파생변수로 생성
   각 변수마다 변수마다 고유값, 다양성, cov 값을 조합하여 불안전성이라는 공통적인 파생변수 생성
   
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
```
feat = {
    'Q1': 'Q1',
    'Q2': 'Q2',
    'Q3': 'Q2',
    'S1': 'S1',
    'S2': 'S1',
    'S3': 'S3'
}

features = {k: x.columns[x.columns.str.startswith(v)] for k, v in feat.items()}
``` 
* 종속변수별에 따른 독립변수 설정     

2. train_test_split
```
targets_binary = ['Q1', 'Q2', 'Q3', 'S2', 'S3']
target_multi = ['S1']

split_data = {}
for col in targets_binary + target_multi:
    y = train_df[col]
    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, stratify=y, random_state=seed)
    split_data[col] = (x_train, x_val, y_train, y_val)
```

3. 데이터 증강 + 하이퍼파라미터 튜닝
```
lgbm_best_param_dict = {}

def smote_optuna_binary(split_date):
  for col in targets_binary:
    print(f'=== target: {col} ===')
    x_train, x_val, y_train, y_val = split_data[col]


    def objective_binary(trial):
      params = {
          'learning_rate': trial.suggest_float("lr", 1e-5, 1e-1, log=True),
          'n_estimators': trial.suggest_int('n_estimators', 50, 500),
          'max_depth': trial.suggest_int('max_depth', 2, 32),
          'num_leaves': trial.suggest_int('num_leaves', 16, 64),
          'subsample': trial.suggest_float('subsample', 0.5, 1.0),
          'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
          'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10.0, log=True),
          'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True),
          'path_smooth' : trial.suggest_loguniform('path_smooth', 1e-8, 1e-3),
          'num_leaves' : trial.suggest_int('num_leaves', 30, 200),
          'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 10, 100),
          'max_bin' : trial.suggest_int('max_bin', 100, 255),
          'feature_fraction' : trial.suggest_uniform('feature_fraction', 0.5, 0.9),
          'bagging_fraction' : trial.suggest_uniform('bagging_fraction', 0.5, 0.9),
          'random_state': seed,
          'n_jobs': -1,
          'verbosity': -1
      }

      feat = features[col]
      clf_binary = LGBMClassifier(**params)
      pipeline = imbalanced_make_pipeline(SMOTE(sampling_strategy='minority', random_state=seed), clf_binary)

      cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
      score = cross_val_score(pipeline, x_train[feat], y_train, cv=cv, scoring='f1').mean()

      trial.report(score, step=0)
      if trial.should_prune():
        raise optuna.exceptions.TrialPruned()

      return score

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective_binary, n_trials=50)

    best_params = study.best_params
    best_score = study.best_value
    lgbm_best_param_dict[col] = best_params
    print(f"{col} 최적 파라미터: {best_params}")
    print(f"{col} 최고 F1 점수: {best_score:.4f}")
    print("=" * 50, "\n")
```
* LGBMClassifier, XGBoostClassifier, CatBoostClassifier 등 각각의 모델에 대한 최적의 파라미터 탐색     

5. 앙상블
   ```
   def voting(split_data,
           lgbm_best_param_dict, xgb_best_param_dict, catb_best_param_dict,
           test_x):

    preds = {}

    # 이진 분류
    for col_binary in targets_binary:
        x_train, x_val, y_train, y_val = split_data[col_binary]
        feat = features[col_binary]

        lgbm = LGBMClassifier(**lgbm_best_param_dict[col_binary],
                              random_state = seed,
                              n_jobs = -1,
                              verbosity = -1)
        xgb = XGBClassifier(**xgb_best_param_dict[col_binary],
                            random_state = seed,
                            n_jobs = -1,
                            verbosity = 0)
        catb = CatBoostClassifier(**catb_best_param_dict[col_binary],
                                  random_state = seed,
                                  n_jobs=-1,
                                  verbose=False)

        smote = SMOTE(sampling_strategy='minority', random_state=seed)
        x_train_over, y_train_over = smote.fit_resample(x_train[feat], y_train)

        voting_binary = VotingClassifier(
            estimators=[('lgbm', lgbm), ('xgb', xgb), ('catb', catb)],
            voting='hard'
        )

        voting_binary.fit(x_train_over, y_train_over)
        y_pred = voting_binary.predict(x_val[feat])

        f1 = f1_score(y_val, y_pred)
        train_acc = voting_binary.score(x_train[feat], y_train)
        val_acc = voting_binary.score(x_val[feat], y_val)

        print(f'Binary {col_binary} -> Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f} | Val F1: {f1:.4f}')
        preds[col_binary] = voting_binary.predict(test_x[feat])

    # 다중 분류
    for col_multi in target_multi:
        x_train, x_val, y_train, y_val = split_data[col_multi]
        feat = features[col_multi]

        model1 = LGBMClassifier(**lgbm_param[col_multi],
                                random_sate = seed,
                                n_jobs =-1,
                                verbosity = -1,
                                objective = 'multiclass',
                                num_class = 3)
        model2 = XGBClassifier(**xgb_param[col_multi],
                               random_sate = seed,
                               n_jobs =-1,
                               verbosity = 0,
                               objective = 'multiclass',
                               num_class = 3)
        model3 = CatBoostClassifier(**catb_param,
                                    random_sate = seed, n_jobs = -1,
                                    verbose = False,
                                    loss_function = 'MultiClass')

        smote = SMOTE(sampling_strategy='not majority', random_state=seed)
        x_train_over, y_train_over = smote.fit_resample(x_train[feat], y_train)

        voting_multi = VotingClassifier(
            estimators=[('lgbm', model1), ('xgb', model2), ('catb', model3)],
            voting='soft'
        )

        voting_multi.fit(x_train_over, y_train_over)
        y_pred = voting_multi.predict(x_val[feat])

        f1 = f1_score(y_val, y_pred, average='macro')
        train_acc = voting_multi.score(x_train[feat], y_train)
        val_acc = voting_multi.score(x_val[feat], y_val)

        print(f'Multi {col_multi} -> Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f} | Val F1: {f1:.4f}')
        preds[col_multi] = voting_multi.predict(test_x[feat])

    return preds

   preds = voting(split_data, lgbm_best_param_dict, xgb_best_param_dict, catb_best_param_dict, test_x)
   ```
* 하이퍼파라미터 튜닝 모델을 결합하여 최적의 결과를 생성
* 앙상블 방법:
  - 예측이 완료된 결과값 조합하여 가장 많이 등장한 값을 최종 결과값으로 선정    

# 검증, 성능 평가 - (Macro F1-score)    
* Macro F1-score를 활용하여 종속변수별 결과값이 실제 테스트 데이터(test)와 얼마나 일치하는지 평가한 후 전체 종속변수 값인 6으로 나눠 최종 산출
