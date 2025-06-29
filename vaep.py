from socceraction.vaep import features as ft
import tqdm
import pandas as pd
import numpy as np
import socceraction.vaep.labels as lab
import xgboost as xgb
import sklearn.metrics as mt
import socceraction.vaep.formula as fm

def features_transform(spadl):
    spadl.loc[spadl.result_id.isin([2, 3]), ['result_id']] = 0 # result == 2 | result == 3 ? result = 0 : result = result
    spadl.loc[spadl.result_name.isin(['offside', 'owngoal']), ['result_name']] = 'fail' # result_name = ('offside' | 'owngoal') ? result_name = 'fail' : result_naem = result_name

    xfns = [ #definicao das features a serem extraidas do evento
        ft.actiontype_onehot,
        ft.bodypart_onehot,
        ft.result_onehot,
        ft.goalscore,
        ft.startlocation,
        ft.endlocation,
        ft.team,
        ft.time,
        ft.time_delta
    ]

    features = []
    for game in tqdm.tqdm(np.unique(spadl.game_id).tolist()): #para cada jogo 
        match_actions = spadl.loc[spadl.game_id == game].reset_index(drop=True) #acoes do jogo em escopo
        match_states = ft.gamestates(actions=match_actions) # para cada acao do jogo analiza as features acao anterior (estado do jogo)
        match_feats = pd.concat([fn(match_states) for fn in xfns], axis=1) #retira as features preteridas do estado do jogo
        features.append(match_feats) 
    features = pd.concat(features).reset_index(drop=True) #junta todos os estados em um dataFrame

    return features

def labels_transform(spadl):
    yfns = [lab.scores, lab.concedes] # probabilidade de fazer gol x probabilidade de levar gol, a partir de uma acao

    labels = []
    for game in tqdm.tqdm(np.unique(spadl.game_id).tolist()): # para cada partida
        match_actions = spadl.loc[spadl.game_id == game].reset_index(drop=True) # eventos da partida
        labels.append(pd.concat([fn(actions=match_actions) for fn in yfns], axis=1)) # prob gol x prob levar gol para cada evento da partida

    labels = pd.concat(labels).reset_index(drop=True) #junta as probabilidades em um unico dataFrame

    return labels

def train_vaep(X_train, y_train, X_test, y_test):
    models = {}
    for m in ['scores', 'concedes']: # para xG a favor e xG contra
        models[m] = xgb.XGBClassifier(random_state=0, n_estimators=50, max_depth=3) 

        #print('training ' + m + ' model')
        models[m].fit(X_train, y_train[m])

        #treino

        #p = sum(y_train[m]) / len(y_train[m]) # previsao ingenua
        #base = [p] * len(y_train[m])
        #y_train_pred = models[m].predict_proba(X_train)[:, 1]
        #train_brier = mt.brier_score_loss(y_train[m], y_train_pred) / mt.brier_score_loss(y_train[m], base)
        #print(m + ' Train NBS: ' + str(train_brier))

        #teste

        #p = sum(y_test[m]) / len(y_test[m]) # previsao ingenua
        #base = [p] * len(y_test[m])
        #y_test_pred = models[m].predict_proba(X_test)[:, 1]
        #test_brier = mt.brier_score_loss(y_test[m], y_test_pred) / mt.brier_score_loss(y_test[m], base)
        #print(m + ' Test NBS: ' + str(test_brier))

        print('----------------------------------------')

    return models

def generate_predictions(features, models): # com o modelo ja treinado, prevemos as probabilidades, no caso, do campeonato espanhol
    
    preds = {}
    for m in ['scores', 'concedes']:
        preds[m] = models[m].predict_proba(features)[:, 1]
    preds = pd.DataFrame(preds)

    return preds

def calculate_action_values(spadl, predictions): # calcula o valor da ação baseada nas predições da probabilidade de marcar um gol x sofrer um gol e agrupa em um dataFrame

    action_values = fm.value(actions=spadl, Pscores=predictions['scores'], Pconcedes=predictions['concedes']) # offensive e defensive values vem daqui
    action_values = pd.concat([
        spadl[['original_event_id', 'action_id', 'game_id', 'player_id', 'start_x', 'start_y', 'end_x', 'end_y', 'type_name', 'result_name']],
        predictions.rename(columns={'scores': 'Pscores', 'concedes': 'Pconcedes'}),
        action_values
    ], axis=1)

    return action_values

def vaep(spadl):
    ftrs = features_transform(spadl)
    lbls = labels_transform(spadl)
    models = train_vaep(X_train=ftrs, y_train=lbls, X_test=ftrs, y_test=lbls)
    preds = generate_predictions(features=ftrs, models=models)
    eventVAEP = calculate_action_values(spadl=spadl, predictions=preds)
    
    return eventVAEP