from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def construir_pipeline(X_train):
    atributos_numericos = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
    atributos_categoricos = X_train.select_dtypes(include=['object']).columns.tolist()

    pipeline_numerico = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    pipeline_categorico = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessador = ColumnTransformer([
        ("num", pipeline_numerico, atributos_numericos),
        ("cat", pipeline_categorico, atributos_categoricos)
    ])

    return preprocessador
