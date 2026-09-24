from functionlist import metrics, dataimport, report, which, checker_input
import sys
import pandas as pd

df = pd.read_excel("./jp_buyback.xlsx")

selected_stock = None
if len(sys.argv) > 2 and sys.argv[1] == "--ticker":
    selected_stock = sys.argv[2]

code, modelling, dependent, hist_features = which("alpha5")
model, features, historical = dataimport("alpha5","alpha1")

trade = checker_input(selected_stock,df)
trade = metrics(trade,"a")

report(model,features, historical, trade, "alpha5","alpha1")