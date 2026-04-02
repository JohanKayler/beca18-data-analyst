import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

df_seleccionados = pd.read_csv("data/postulantes_seleccionados.csv")
df_no_seleccionados = pd.read_csv("data/postulantes_no_seleccionados.csv")

df = pd.concat([df_seleccionados,df_no_seleccionados],  ignore_index=True)

#Limpiamos las columnas importantes
df_limpio = df[["MODALIDAD_BECA","DNI", "IES","CARRERA","CONCEPTO_A","CONCEPTO_B","PUNTAJE_FINAL","CONDICION"]]

#Trabajaremos con solo la modalidad ordinaria
df_limpio = df_limpio[df["MODALIDAD_BECA"]=="BECA 18 ORDINARIA"]

#Pesos relativos
df_limpio["PESO_A"]= df_limpio["CONCEPTO_A"]/df_limpio["PUNTAJE_FINAL"]
df_limpio["PESO_B"]= df_limpio["CONCEPTO_B"]/df_limpio["PUNTAJE_FINAL"]

#Ratio
df_limpio["RATIO_AB"] = df_limpio["CONCEPTO_A"]/(df_limpio["CONCEPTO_B"]+1)

le = LabelEncoder()
df_limpio["CONDICION_ENCODING"] = le.fit_transform(df_limpio["CONDICION"])


#PRUEBA
def clasificar_tipo(ies):
    ies = str(ies).upper()
    
    if "UNIVERSIDAD" in ies:
        return "UNIVERSIDAD"
    elif "" in ies or "TECSUP" in ies or "CIBERTEC" in ies:
        return "INSTITUTO"
    else:
        return "OTRO"

df_limpio["TIPO_INSTITUCION"] = df_limpio["IES"].apply(clasificar_tipo)

def clasificar_gestion(ies):
    ies = str(ies).upper()
    
    if "NACIONAL" in ies:
        return "PUBLICA"
    else:
        return "PRIVADA"

df_limpio["GESTION"] = df_limpio["IES"].apply(clasificar_gestion)

df_limpio[df_limpio["CONDICION"]== "SELECCIONADO"]["IES"].value_counts()

tabla_seleccionados = df_limpio.groupby("IES")["CONDICION"].apply(lambda x: (x == "SELECCIONADO").sum())
print(tabla_seleccionados)

# print(df_limpio.groupby["IES"].sort_values())

