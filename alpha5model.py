import pandas as pd
from functionlist import metrics, model, model_df, dataexport, which, modelplots

code, modelling, dependent, features = which("alpha5")

df = pd.read_excel("./jp_buyback.xlsx")
df = metrics(df,code)

model_df, X, y, groups = model_df(df,features,modelling,dependent)
model , pred = model(X,y, groups)

modelplots(modelling,model,features,X,y,pred,model_df, oof_pred=model.oof_predictions_)
dataexport(model,features,X,y,model_df,modelling,dependent, groups=groups, oof_pred=model.oof_predictions_)
