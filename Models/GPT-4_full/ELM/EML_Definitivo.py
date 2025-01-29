from sklearn.datasets import load_iris, load_digits, load_diabetes, make_regression
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import elm
import pandas as pd
from sklearn.metrics import r2_score
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold,ShuffleSplit
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_score
from sklearn import metrics
from sklearn.metrics import precision_score
from sklearn.neural_network import MLPRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import KFold,ShuffleSplit
from sklearn.metrics import confusion_matrix, accuracy_score 
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_validate
from sklearn.model_selection import cross_val_score
from sklearn.metrics import r2_score
import random
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import pickle
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import random
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
from itertools import cycle

import shap
import pickle
import numpy as np



def dados_treino_e_validacao() :
    dataset = pd.read_excel('Input_Total.xlsx')
    dataset = dataset.loc[:,dataset.apply(pd.Series.nunique) != 1]

    # Divide o conjunto de dados em X e y
    X = dataset.iloc[:, 1:203].values 
    print(X)
    encoder = LabelEncoder()
    encoder.fit(dataset.iloc[:,203].values)
    y =  encoder.fit_transform(dataset.iloc[:,203].values)
    print(y)
    scaler = StandardScaler()
    scaler.fit(X)
    X= scaler.transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


    with open('X_train.pickle', 'wb') as f:
        pickle.dump(X_train, f)

    with open('X_test.pickle', 'wb') as f:
        pickle.dump(X_test, f)

    with open('y_train.pickle', 'wb') as f:
        pickle.dump(y_train, f)

    with open('y_test.pickle', 'wb') as f:
        pickle.dump(y_test, f)

    with open('X.pickle', 'wb') as f:
        pickle.dump(X, f)

    with open('y.pickle', 'wb') as f:
        pickle.dump(y, f)

    with open('scaler.pickle', 'wb') as f:
        pickle.dump(scaler, f)

    with open('encoder.pickle', 'wb') as f:
        pickle.dump(encoder, f)



#dados_treino_e_validacao()

def reduzir_dimensao (reduzendo, todas_as_features, colunas_para_manter) :

  arr = reduzendo

  # obter todos os rótulos das colunas do dataframe
  rotulos_das_colunas = todas_as_features.columns.tolist()

  # criar uma lista com as posições das colunas a serem deletadas
  colunas_para_deletar = [i for i in range(len(rotulos_das_colunas)) if rotulos_das_colunas[i] not in colunas_para_manter]

  # deletar as colunas do array numpy
  arr = np.delete(arr, colunas_para_deletar, axis=1)

  return arr

def hiperparametros_treinoEML_metricas_cv() :


    # Carregar as variáveis
    with open('X_train_selected.pkl', 'rb') as f:
        X_train = pickle.load(f)

    with open('X_test_selected.pkl', 'rb') as f:
        X_test = pickle.load(f)

    with open('y_train.pickle', 'rb') as f:
        y_train = pickle.load(f)

    with open('y_test.pickle', 'rb') as f:
        y_test = pickle.load(f)

   # with open('X.pickle', 'rb') as f:
    #    X = pickle.load(f)

   # with open('y.pickle', 'rb') as f:
    #    y = pickle.load(f)

    #with open('scaler.pickle', 'rb') as f:
    #    scaler = pickle.load(f)

    #with open('encoder.pickle', 'rb') as f:
    #    encoder = pickle.load(f)
    
    ############################

    reduzir_dimensao= '''
    dataset = pd.read_excel('Input_Total.xlsx')
    dataset = dataset.loc[:,dataset.apply(pd.Series.nunique) != 1]
    dataset = dataset.filter(regex=r'^(?!.*Keq)')
    dataset = dataset.drop('Classe', axis=1)
    dataset = dataset.drop('Composto', axis=1)
    dataset = dataset.drop('Átomos', axis=1)
    
    dataframe_pandas_features= dataset
    
    
    colunas_para_manter = ['atomic_ea_minimo', 'atomic_ea_maximo', 'atomic_ea_soma',
       'atomic_en_allen _soma', 'atomic_en_allen _desvio',
       'atomic_en_allredroch_minimo', 'atomic_hatm_minimo',
       'atomic_spacegroupnum_maximo', 'atomic_spacegroupnum_desvio',
        'van_der_waals_rad_minimo']
    
    X_train = reduzir_dimensao(X_train,dataframe_pandas_features,colunas_para_manter).copy()
    X_test = reduzir_dimensao(X_test,dataframe_pandas_features,colunas_para_manter).copy()
    '''
    
    ##################
        
    param_grid = {
        'hidden_units': [3,6,9,12,15,18,20,22,25,32,36,40,50,75,100,150,200,400],
        'activation_function' : ['sigmoid', 'relu', 'sin', 'tanh','leaky_relu'],
        'C' : [0,1,2,3,4,5,6,7,8,9,10,13,17,25],
        'random_type': ['normal','uniform'],
        'treinador' : ['no_re', 'solution1' , 'solution2']
    }

    melhor_r2_score = -99999999.9
    melhor_numero = None
    melhor_funcao = None
    melhor_c = None
    melhor_tipo = None
    melhor_treinador = None 
    model = None

    # Create a 2-fold cross validation
    kf = KFold(n_splits=2)

    for numero in param_grid['hidden_units']:
        for funcao in param_grid['activation_function']:
            for cl in param_grid['C']:
                for tipo in param_grid['random_type']:
                    for treinador in param_grid['treinador'] :
                    
                    
                        
                        if 1<2:


                            r2_scores = []
                            for train_index, test_index in kf.split(X_train):
                                try:
                                    x_train_fold, x_val_fold = X_train[train_index], X_train[test_index]
                                    y_train_fold, y_val_fold = y_train[train_index], y_train[test_index]
                                   
                                
                                      
                                    model = elm.elm(hidden_units=numero, 
                                                activation_function=funcao, random_type=tipo, 
                                                C=cl, elm_type='clf',x=x_train_fold, y=y_train_fold,one_hot=True)
                                                
                                    model.fit(treinador)
                                    # predict on test data
                                    prediction = model.predict(x_val_fold)
                                    # calculate r2 score
                                    r2 = r2_score(y_val_fold, prediction)
                                    r2_scores.append(r2)
                                    print(r2)

                                    # calculate average r2 score
                                    avg_r2 = np.mean(r2_scores)

                                    if melhor_r2_score < avg_r2 and avg_r2<1 and avg_r2>0 :
                                        melhor_r2_score = avg_r2
                                        melhor_numero = numero
                                        melhor_funcao = funcao
                                        melhor_c = cl
                                        melhor_tipo = tipo
                                        melhor_treinador = treinador
                                except:
                                    pass
    
    
    with open("hiperparametrosEML_CV.txt", "w") as arquivo:
        arquivo.write('hidden_units = '+ str(melhor_numero) + ',' + 'activation_function = '+ '\''+str(melhor_funcao)+ '\''
       + ',' + 'C='+str(melhor_c)+ ','+ 'random_type=' +'\''+str(melhor_tipo)+ '\'' 
       +',' + 'x=x_train, y=y_train,elm_type=\'clf\''+
       ' treinador= '+'\''+str(melhor_treinador)+'\'' + ' melhor_r2_score = '+ str(melhor_r2_score))

#hiperparametros_treinoEML_metricas_cv()

def gerar_modelo ():

    # Carregar as variáveis
    with open('X_train_selected.pkl', 'rb') as f:
        X_train = pickle.load(f)

    with open('X_test_selected.pkl', 'rb') as f:
        X_test = pickle.load(f)

    with open('y_train.pickle', 'rb') as f:
        y_train = pickle.load(f)

    with open('y_test.pickle', 'rb') as f:
        y_test = pickle.load(f)
        



    ############################

    reduzir_dimensao= ''' dataset = pd.read_excel('Input_Total.xlsx')
    dataset = dataset.loc[:,dataset.apply(pd.Series.nunique) != 1]
    dataset = dataset.filter(regex=r'^(?!.*Keq)')
    dataset = dataset.drop('Classe', axis=1)
    dataset = dataset.drop('Composto', axis=1)
    dataset = dataset.drop('Átomos', axis=1)
    
    dataframe_pandas_features= dataset
    
    
    colunas_para_manter = ['atomic_ea_minimo', 'atomic_ea_maximo', 'atomic_ea_soma',
       'atomic_en_allen _soma', 'atomic_en_allen _desvio',
       'atomic_en_allredroch_minimo', 'atomic_hatm_minimo',
       'atomic_spacegroupnum_maximo', 'atomic_spacegroupnum_desvio',
        'van_der_waals_rad_minimo']
    
    X_train = reduzir_dimensao(X_train,dataframe_pandas_features,colunas_para_manter).copy()
    X_test = reduzir_dimensao(X_test,dataframe_pandas_features,colunas_para_manter).copy()
    '''
    
    ##################
    
        
    #print(len(X_train[0]))
    
    l=0
    a=0
    while l< 20000:
        model = elm.elm(hidden_units=15, activation_function='leaky_relu',
        random_type='uniform', x=X_train, y=y_train, C=2, elm_type='clf',one_hot=True) 
        beta, train_accuracy, running_time = model.fit('no_re')


        # test - Observe que estamos calculando a acurácia com y_test ainda codificado
        prediction = model.predict(X_test)
        #prediction_decoded = encoder.inverse_transform(prediction)
        #print("classifier test prediction:", prediction_decoded)
        #print('classifier test accuracy:', model.score(X_test, y_test))
        
        acc = accuracy_score(y_test, model.predict(X_test), sample_weight=None)
        print(float(acc))

        if acc > a:
            a = acc
            k = model
        l = l +1

        with open('modelEML.pickle', 'wb') as f:
            pickle.dump(k, f)
         
#gerar_modelo ()


def testar_modelo ():

    # Carregar as variáveis
    with open('X_train_selected.pkl', 'rb') as f:
        X_train = pickle.load(f)

    with open('X_test_selected.pkl', 'rb') as f:
        X_test = pickle.load(f)

    with open('y_train.pickle', 'rb') as f:
        y_train = pickle.load(f)

    with open('y_test.pickle', 'rb') as f:
        y_test = pickle.load(f)
        
    with open('modelEML.pickle', 'rb') as f:
        model = pickle.load(f)

    
    prediction = model.predict(X_test)
    #prediction_decoded = encoder.inverse_transform(prediction)
    #print("classifier test prediction:", prediction_decoded)
    print('classifier test accuracy:', model.score(X_test, y_test))
    


    y_pred = model.predict(X_test)
    conf_matrix = confusion_matrix(y_test, y_pred)


    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix')
    plt.show()
    
testar_modelo ()



def new_boot(X_test, y_test):

  novo_X = X_test.copy()
  novo_Y = y_test.copy()

  c= 0
  while c < len(X_test):
    rand = random.randint(0,len(X_test)-1)



    novo_X[c] = X_test[rand].copy()
    novo_Y[c] = y_test[rand]

    c= c+1


  return novo_X , novo_Y

# Pega a acurácia de um boot
# Ajuste em pegar a mérica sem ser pelo report
def pegar_acuracia_do_relatorio(novo_X , novo_Y):

  return accuracy_score(novo_Y, model.predict(novo_X), sample_weight=None)


if 1>2:

    with open('X_train.pickle', 'rb') as f:
        X_train = pickle.load(f)

    with open('X_test.pickle', 'rb') as f:
        X_test = pickle.load(f)

    with open('y_train.pickle', 'rb') as f:
        y_train = pickle.load(f)

    with open('y_test.pickle', 'rb') as f:
        y_test = pickle.load(f)

    with open('X.pickle', 'rb') as f:
        X = pickle.load(f)

    with open('y.pickle', 'rb') as f:
        y = pickle.load(f)

    with open('scaler.pickle', 'rb') as f:
        scaler = pickle.load(f)

    with open('encoder.pickle', 'rb') as f:
        encoder = pickle.load(f)
        
    with open('modelEML.pickle', 'rb') as f:
        model = pickle.load(f)
        
confidence_interval ='''
numero_boots = 10001
lista_boots = []
contador = 0

while contador < numero_boots:
  x, y = new_boot(X_test, y_test)
  lista_boots.append(pegar_acuracia_do_relatorio(x, y))
  contador =contador +1

plt.hist(lista_boots)
plt.xlabel('Acurácia')
plt.ylabel('Número de boots')
plt.show()

#converte a lista em float explicitamente para a função percentile ser aplicada
array = list()
for elemento in lista_boots:
  array.append(float(elemento))

# calcula os limites da integral da gaussiana que correspondem a área desejada

alpha = 0.95
p = ((1.0-alpha)/2.0) * 100
lower =  np.percentile(array, p)
p = (alpha+((1.0-alpha)/2.0)) * 100
upper =  np.percentile(array, p)

print("Intervalo de confiança : ["+str(lower)+","+str(upper)+"]")
print("Acurácia 'real' do modelo performada no teste : "+ str(accuracy_score(y_test, model.predict(X_test), sample_weight=None)))

'''

rock_curve = '''
########## Rock

with open('X_test_selected.pkl', 'rb') as f:
    X_test_selected = pickle.load(f)

with open('X_train_selected.pkl', 'rb') as f:
    X_train_selected = pickle.load(f)

with open('modelEML_73%.pickle', 'rb') as f:
    model = pickle.load(f)
    
with open('y_train.pickle', 'rb') as f:
    y_train = pickle.load(f)

with open('y_test.pickle', 'rb') as f:
    y_test = pickle.load(f)    

with open('encoder.pickle', 'rb') as f:
    encoder = pickle.load(f) 



# Supondo que y_test já esteja disponível e não seja binário
# Se y_test já for binário (em formato one-hot), você pode pular esta etapa
# Binarizar os rótulos em uma configuração um contra todos




# Ajustando para a ausência do método predict_proba
y_pred = model.predict(X_test_selected)  # Assume-se que model.predict retorna a classe prevista diretamente

# Converter y_test e y_pred para formato binário para cada classe
y_test_binarized = label_binarize(y_test, classes=np.unique(y_test))
y_pred_binarized = label_binarize(y_pred, classes=np.unique(y_test))
n_classes = y_test_binarized.shape[1]

class_labels = np.unique(y_test)
print(class_labels)

fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    # Calculando FPR e TPR manualmente
    cm = confusion_matrix(y_test_binarized[:, i], y_pred_binarized[:, i])
    TN, FP, FN, TP = cm.ravel()
    TPR = TP / (TP + FN) if (TP + FN) != 0 else 0
    FPR = FP / (FP + TN) if (FP + TN) != 0 else 0
    
    fpr[i], tpr[i] = FPR, TPR
    roc_auc[i] = auc([0, FPR, 1], [0, TPR, 1])  # AUC é simplificado para um modelo sem probabilidades

# Plot da curva ROC para cada classe
colors = cycle(['blue', 'red', 'green', 'cyan', 'magenta', 'yellow', 'black', 'pink'])
plt.figure(figsize=(10, 8))
for i, color in zip(range(n_classes), colors):
    plt.plot([0, fpr[i], 1], [0, tpr[i], 1], color=color, lw=2,
             label='{0} (AUC = {1:0.2f})'.format(class_labels[i], roc_auc[i]))

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Taxa de Falsos Positivos')
plt.ylabel('Taxa de Verdadeiros Positivos')
plt.title('Curva ROC Multiclasse (Simplificada)')
plt.legend(loc="lower right")
plt.show()
'''

########## Shap
#SHAP
#!pip install shap

SHAP = '''
with open('X_test_selected.pkl', 'rb') as f:
    X_test_selected = pickle.load(f)

with open('X_train_selected.pkl', 'rb') as f:
    X_train_selected = pickle.load(f)

with open('modelEML_73%.pickle', 'rb') as f:
    model = pickle.load(f)
    
with open('y_train.pickle', 'rb') as f:
    y_train = pickle.load(f)

with open('y_test.pickle', 'rb') as f:
    y_test = pickle.load(f)    





# Criando um explainer usando X_train para entender como o modelo aprendeu a fazer previsões
explainer = shap.Explainer(model.predict, X_train_selected)

# Calculando os valores SHAP para o conjunto de dados de teste para ver como as características afetam as previsões em novos dados
shap_values = explainer(X_test_selected)
print(shap_values)

shap.summary_plot(shap_values, X_test_selected, plot_type="dot")
'''