# from lightgbm import LGBMClassifier, LGBMRegressor
# import lightgbm as lgb
# from sklearn.feature_selection import SelectFromModel
# import numpy as np
# from xgboost import XGBClassifier, XGBRegressor
# from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV
# from sklearn.datasets import load_boston
# from sklearn.metrics import accuracy_score, r2_score

# #using LGBMRegressor
# datasets = load_boston()




# ##데이터###
# x, y = load_boston(return_X_y=True)# 이렇게 적어줘도됨

# print(x.shape) #(506, 13)
# print(y.shape) #(506,)

# x_train, x_test, y_train, y_test = train_test_split(x,y,train_size = 0.8,
#                                                     shuffle = True, random_state = 66)


# ###기본 모델###
# model = LGBMRegressor(n_estimators = 300, max_depth =5, scale_pos_weight=1, colsample_bytree=1, learning_rate = 0.1, n_jobs = -1, num_leaves=5)
# print('x_train', x_train.shape)
# print('y_train', y_train.shape)



# model.fit(x_train, y_train)
# score = model.score(x_test, y_test)
# print('R2: ', score) #R2:  0.9323782715918234

# ###feature engineerg###
# thresholds = np.sort(model.feature_importances_)
# print('threshold:',thresholds)



# for thresh in thresholds:      
#     selection = SelectFromModel(model, threshold=thresh, prefit=True)
#                                             #median
#     select_x_train = selection.transform(x_train)
#     select_x_test = selection.transform(x_test) 
#     model2 = LGBMRegressor(n_estimators = 300, max_depth =5, scale_pos_weight=1, colsample_bytree=1, learning_rate = 0.1, n_jobs = -1, num_leaves=5)
#     model2.fit(select_x_train, y_train, verbose = False, eval_metric=['logloss', 'rmse'], 
#             eval_set = [(select_x_train, y_train), (select_x_test, y_test)], early_stopping_rounds = 20)

#     y_pred = model2.predict(select_x_test)
    
#     score = r2_score(y_test, y_pred)
    
#     print('Thresh=%.3f, n=%d, R2:%.2f%%' %(thresh, select_x_train.shape[1], score*100.0))

# # best_idx = res.argmax()                     #결과값 최대값의 index wjwkd
# # score = res[best_idx]                       #위 인덱스 기반으로 점수 호출
# # total_col = x_train.shape[1] - best_idx   #전체 컬럼 계산
# # models[best_idx].save_model(f'./model/lgbm_save/boston--{score}--{total_col}--.model') #인덱스 기반으로 모델 저장

# from sklearn.metrics import accuracy_score, r2_score, recall_score, f1_score, precision_score, roc_auc_score
# from sklearn.metrics import confusion_matrix

# def get_clf_eval(y_test, y_pred):
#     confusion = confusion_matrix(y_test, y_pred)
#     accuracy = accuracy_score(y_test, y_pred)
#     precision = precision_score(y_test, y_pred)
#     recall = recall_score(y_test, y_pred)
#     F1 = f1_score(y_test, y_pred)
#     AUC = roc_auc_score(y_test, y_pred)
#     print('오차행렬:\n', confusion)
#     print('\n정확도: {:.4f}'.format(accuracy))
#     print('정밀도: {:.4f}'.format(precision))
#     print('재현율: {:.4f}'.format(recall))
#     print('F1: {:.4f}'.format(F1))
#     print('AUC: {:.4f}'.format(AUC))

# get_clf_eval(y_test, y_pred)





#########################################################
