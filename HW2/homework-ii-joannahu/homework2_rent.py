import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import LassoCV
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_predict
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import f_regression
from sklearn.preprocessing import Imputer
from sklearn import metrics
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
import webbrowser

#global data
X_test = []
y_test = []

#Avoid too biased features, choose .8 as threshold
def remove_features(data):
    count = 0
    select_cols=[]
    for i in range(data.shape[1]):
        if data.iloc[:,i].value_counts().max()/data.shape[0]<=0.9:
            select_cols.append(i)
    data=data.iloc[:,select_cols]
    return data

def missing_value(data):
    data.sc36[data.sc36==8] = 0
    data.sc36[data.sc36==3] = 0
    data.sc37[data.sc37==8] = 0
    data.sc37[data.sc37==3] = 0
    data.sc38[data.sc38==8] = 0
    data.sc38[data.sc38==3] = 0
    data.sc181[data.sc181==7]=0
    data.sc181[data.sc181==8]=0
    data.sc186[data.sc186==8]=0
    data.sc197[data.sc197==4]=0
    data.sc197[data.sc197==8]=0
    data.sc198[data.sc198==8]=0
    data.sc187[data.sc187==8]=0
    data.sc188[data.sc188==8]=0
    data.sc571[data.sc571==5]=0
    data.sc571[data.sc571==8]=0
    data.sc189[data.sc189==8]=0
    data.sc189[data.sc189==5]=0
    data.sc190[data.sc190==8]=0
    data.sc191[data.sc191==8]=0    
    data.sc192[data.sc192==8]=0
    data.sc193[data.sc193==8]=0
    data.sc193[data.sc193==9]=3
    data.sc194[data.sc194==8]=0
    data.sc196[data.sc196==8]=0
    data.sc548[data.sc548==8]=0
    data.sc548[data.sc548==3]=0
    data.sc549[data.sc549==8]=0
    data.sc549[data.sc549==3]=0
    data.sc575[data.sc575==3]=0
    data.sc575[data.sc575==8]=0
    data.sc173[data.sc173==3]=0
    data.sc173[data.sc173==8]=0
    data.sc173[data.sc173==9]=0
    data.sc171[data.sc173==3]=0
    data.sc171[data.sc173==8]=0
    data.uf1_14[data.uf1_14==9]=0
    data.sc154[data.sc154==8]=0
    data.sc154[data.sc154==9]=0
    data.sc157[data.sc157==8]=0
    data.sc157[data.sc157==9]=0
    data.sc174[data.sc174==8]=0
    data.sc174[data.sc174==8]=0
    data.sc181[data.sc181==9]=0
    data.sc181[data.sc181==7]=0
    data.sc181[data.sc181==8]=0
    data.sc199[data.sc199==8]=0
    data.rec15[data.rec15==10]=0
    data.rec15[data.rec15==11]=0
    data.rec15[data.rec15==12]=0
    data.rec54[data.rec54==7]=0
    data.rec53[data.rec53==7]=0
    data.sc110[data.sc110==98]=0

    data.sc110[data.sc110==99]=0


        
    return data

def fill_missing(data):
        imp=Imputer(missing_values=0,strategy='most_frequent').fit(data)
        data=imp.transform(data)
        return data

#def score_rent(url, os_info):
def score_rent(filename):
    #import data
    nyc_rent=pd.read_csv(filename)
    
    
    #choose obs with rent amount applicable
    nyc_rent=nyc_rent[nyc_rent['uf17']!=99999]
    #Intuitively pick variables
    #pick features, idea: considering house condition/price/location. etc, do not 
    #consider household, owner, etc.
    X=nyc_rent[['boro', 
            'uf1_1', 'uf1_2', 'uf1_3', 'uf1_4', 'uf1_5','uf1_6', 
            'uf1_7', 'uf1_8', 'uf1_9', 'uf1_10', 'uf1_11', 
            'uf1_12','uf1_13', 'uf1_14', 'uf1_15', 'uf1_16', 'uf1_35', 
            'uf1_17','uf1_18', 'uf1_19', 'uf1_20', 'uf1_21', 'uf1_22', 
            'sc23', 'sc24',
            'sc36', 'sc37', 'sc38', 
            'uf6', 
            'uf9',  'uf48',
             'sc149', 'sc173','sc171','sc150', 'sc151',
           'sc152', 'sc153', 'sc154', 'sc155', 'sc156', 'sc157', 'sc158', 
           'sc174', 'uf13','uf14','sc164','uf15','sc166','uf16','uf64',
            'uf17', 'sc181','sc186',
            
            'sc197', 'sc198',
           'sc187', 'sc188', 'sc571', 'sc189', 'sc190', 'sc191', 'sc192','sc193','sc194'
           ,'sc196','sc548','sc549','sc199',
            'sc575', 'rec15','uf19','uf23','sc26','rec54','rec53' ,'rec21','new_csr','sc110'
       ]]
    #boro borough
    #uf1-1 - uf1-6 condition of external walls
    #uf1-7 - uf1-11 condition of windows
    #uf1-12 - uf1-16, uf1-35, condition of stairways
    #uf1-17 - uf1-22 condition of floors
    #sc23 condition of building(observation)
    #sc24 broken windows buildings
    #sc36-38, wheelchair
    #uf6 value
    #uf9 monthly maintenance fees
    #uf48 number of units in building
    #sc149, passenger elevator in building
    #sc173, sidewalk to elevator
    #sc150, 151,number of rooms,bedrooms
    #sc 152-153, plumbing
    #sc 154 toilet
    #155-157 kitchen
    #158 heating
    #174 home energy assistance
    #uf17, y
    #sc181, length of lease
    #sc186, number of heating equip breakdowns
    #sc197 air conditioning
    #198 carbon monoxide deterctor
    #187 additional source of heat
    #188 mice,rats
    #571 cockroach
    #189 exterminator service
    #190 cracks or holes in interior walls or ceiling
    #191 holes in floors
    #192,193 broken plaster or peeling paint
    #194 water leakage
    #548,549 assistance
    #575 telephone
    #uf 23, yearbuilt
    #rec21 condition of building recode
    
    X_new=remove_features(X)
    missing_value(X_new);
    
    #remove uf13,14,64, too many missing values
    X_new.columns
    X_new=X_new[['boro', 'uf1_14', 'sc36', 'sc37', 'sc38', 'uf48', 'sc149', 'sc173',
       'sc171', 'sc150', 'sc151', 'sc154', 'sc157', 'sc158', 'sc174', 
       'sc166', 'uf17', 'sc181', 'sc186', 'sc197', 'sc198',
       'sc187', 'sc188', 'sc571', 'sc189', 'sc190', 'sc191', 'sc192', 'sc193',
       'sc194', 'sc196', 'sc548', 'sc549', 'sc199', 'sc575', 'rec15', 'uf19',
       'uf23', 'sc26', 'rec54', 'rec53', 'new_csr','sc110']]
    
    #remove observations that has more than 10 missing values 
    df = X_new
    df = df[(df == 0).astype(int).sum(axis=1) <=10]
    
    X_new = pd.DataFrame(fill_missing(df),columns=X_new.columns)
    
    y = X_new['uf17']
    X_new=X_new.drop('uf17', 1)
    
    category_list=['boro', 'uf1_14', 'sc36', 'sc37', 'sc38', 'uf48', 'sc149', 'sc173',
       'sc171',  'sc154', 'sc157', 'sc158', 'sc174','sc166', 'sc174',  'sc181', 'sc186', 'sc197', 'sc198',
       'sc187', 'sc188', 'sc571', 'sc189', 'sc190', 'sc191', 'sc192', 'sc193',
       'sc194', 'sc196', 'sc548', 'sc549', 'sc199', 'sc575', 'rec15', 'uf19',
       'uf23', 'sc26', 'rec54', 'rec53', 'new_csr','sc110']
    for i in category_list:
        X_new[i]=X_new[i].astype('category')
        
    #global dum_cat
    global X_test
    global y_test
    X_train, X_test, y_train, y_test = train_test_split(X_new, y, random_state=50)
    dum_cat = ['boro','new_csr','sc36','sc37','sc38','uf48','sc151','sc158','sc181','sc197','sc198'
          ,'sc575','rec15','rec53','sc110']
    #g_dum_cat = dum_cat

    x_train_dummy = pd.DataFrame()
    for ele in (dum_cat):
        x_train_dummy = pd.concat([x_train_dummy,pd.get_dummies(X_train[ele], prefix=ele)],axis = 1)
    
    '''  
    # prepare a range of alpha values to test
    alphas = np.array([1,2,3,4,5,6,7,8,9,10])
    # create and fit a ridge regression model, testing each alpha
    model = Lasso()
    grid = GridSearchCV(estimator=model, param_grid=dict(alpha=alphas))
    grid.fit(X_train, y_train)
    print(grid)
    # summarize the results of the grid search
    print(grid.best_score_)
    print(grid.best_estimator_.alpha)
    '''  
    lasso = make_pipeline(StandardScaler(), Lasso(alpha=2))
    scores=cross_val_score(lasso, x_train_dummy, y_train, cv=10)

    x_test_dummy = pd.DataFrame()
    for ele in (dum_cat):
        x_test_dummy = pd.concat([x_test_dummy,pd.get_dummies(X_test[ele], prefix=ele)],axis = 1)

    #lasso = make_pipeline(StandardScaler(), Lasso(alpha=10))
    predicted = cross_val_predict(lasso, x_test_dummy,y_test, cv=10)
    #print predicted
    #print y_test
    return np.mean(scores)
    
#s = score_rent("homework2_data.csv")
s = score_rent("https://ndownloader.figshare.com/files/7586326")
#print s

def predict_rent():
    dum_cat = ['boro','new_csr','sc36','sc37','sc38','uf48','sc151','sc158','sc181','sc197','sc198'
          ,'sc575','rec15','rec53','sc110']
    x_test_dummy = pd.DataFrame()
    for ele in (dum_cat):
        x_test_dummy = pd.concat([x_test_dummy,pd.get_dummies(X_test[ele], prefix=ele)],axis = 1)

    lasso = make_pipeline(StandardScaler(), Lasso(alpha=2))
    predicted = cross_val_predict(lasso, x_test_dummy,y_test)
    return np.array(y_test),predicted

predict_rent()




