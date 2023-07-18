import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, LogisticRegression
from sklearn.naive_bayes import BernoulliNB, MultinomialNB, GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
# Título de la aplicación
st.title("Aplicación de Entrenamiento de Modelos")

# Cargar el conjunto de datos
uploaded_file = st.file_uploader("Cargar archivo CSV", type="csv")
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # Obtener la lista de columnas del conjunto de datos
    columns = df.columns.tolist()

    # Seleccionar las columnas a utilizar
    default_columns = columns  # Todas las columnas seleccionadas por defecto
    selected_columns = st.multiselect("Seleccionar columnas a utilizar", columns, default=default_columns)

    # Filtrar el conjunto de datos con las columnas seleccionadas
    df_selected = df[selected_columns]

    # Verificar si hay valores nulos en el conjunto de datos
    if df_selected.isnull().values.any():
        st.warning("¡Atención! El conjunto de datos contiene valores nulos.")
    else:
        st.success("¡El conjunto de datos está libre de valores nulos!")

    # Configurar si es un problema de regresión o clasificación
    problem_type = st.radio("Seleccionar tipo de problema", ("Regresión", "Clasificación"))

    # Seleccionar el vector objetivo
    target_col = st.selectbox("Seleccionar vector objetivo", selected_columns)

    # Dividir los datos en conjuntos de entrenamiento y prueba
    X = df_selected.drop(target_col, axis=1)
    y = df_selected[target_col]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Construir el escalador
    scaler = StandardScaler()

    # Sidebar para seleccionar hiperparámetros
    st.sidebar.title("Ajuste de Hiperparámetros")

    # Regresión Lineal
    if problem_type == "Regresión":
        model = st.selectbox("Seleccionar modelo", ("LinearRegression", "Ridge", "Lasso", "ElasticNet"))

        if model == "LinearRegression":
            reg = LinearRegression()
        elif model == "Ridge":
            alpha = st.sidebar.slider("Valor de alpha", 0.0, 10.0, 0.5)
            reg = Ridge(alpha=alpha)
        elif model == "Lasso":
            alpha = st.sidebar.slider("Valor de alpha", 0.0, 10., 0.5)
            reg = Lasso(alpha=alpha)
        elif model == "ElasticNet":
            alpha = st.sidebar.slider("Valor de alpha", 0.0, 10.0, 0.5)
            l1_ratio = st.sidebar.slider("Valor de l1_ratio", 0.0, 10.0, 0.5)
            reg = ElasticNet(alpha=alpha, l1_ratio=l1_ratio)

        # Aplicar escalado a las características
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Obtener los valores de hiperparámetros seleccionados por el usuario
        params = {}
        for param_name in st.sidebar.text_input("Hiperparámetros (clave=valor)").split(','):
            if '=' in param_name:
                key, value = param_name.split('=')
                params[key.strip()] = eval(value.strip())

        # Entrenar el modelo utilizando GridSearchCV
        reg_grid = GridSearchCV(reg, params, cv=5)
        reg_grid.fit(X_train_scaled, y_train)

        # Obtener el mejor modelo y sus resultados
        best_reg = reg_grid.best_estimator_
        best_score = reg_grid.best_score_
        best_params = reg_grid.best_params_

        # Calcular y mostrar el puntaje R2
        score = best_reg.score(X_test_scaled, y_test)
        st.write("Mejor modelo:", best_reg)
        st.write("Mejor puntaje R2:", best_score)
        st.write("Mejores hiperparámetros:", best_params)
        st.write("Puntaje R2 del mejor modelo en conjunto de prueba:", score)

    # Clasificación
    else:
        model = st.selectbox("Seleccionar modelo", ("LogisticRegression", "BernoulliNB", "MultinomialNB", "GaussianNB",
                                                    "LinearDiscriminantAnalysis", "QuadraticDiscriminantAnalysis", "SVC"))

        if model == "LogisticRegression":
            reg = LogisticRegression()
        elif model == "BernoulliNB":
            reg = BernoulliNB()
        elif model == "MultinomialNB":
            reg = MultinomialNB()
        elif model == "GaussianNB":
            reg = GaussianNB()
        elif model == "LinearDiscriminantAnalysis":
            reg = LinearDiscriminantAnalysis()
        elif model == "QuadraticDiscriminantAnalysis":
            reg = QuadraticDiscriminantAnalysis()
        elif model == "SVC":
            reg = SVC()

        # Obtener los valores de hiperparámetros seleccionados por el usuario
        params = {}
        for param_name in st.sidebar.text_input("Hiperparámetros (clave=valor)").split(';'):
            if '=' in param_name:
                key, value = param_name.split('=')
                params[key.strip()] = eval(value.strip())
        print(params)
        # Entrenar el modelo utilizando GridSearchCV
        reg_grid = GridSearchCV(reg, params, cv=5)
        reg_grid.fit(X_train, y_train)

        # Obtener el mejor modelo y sus resultados
        best_reg = reg_grid.best_estimator_
        best_score = reg_grid.best_score_
        best_params = reg_grid.best_params_

        # Calcular y mostrar la precisión
        accuracy = best_reg.score(X_test, y_test)
        st.write("Mejor modelo:", best_reg)
        st.write("Mejor precisión:", best_score)
        st.write("Mejores hiperparámetros:", best_params)
        st.write("Precisión del mejor modelo en conjunto de prueba:", accuracy)

        # Calcular y mostrar la matriz de confusión
        y_pred = reg_grid.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)

        fig, ax = plt.subplots()
        sns.heatmap(cm, ax=ax, cmap='Blues', annot=True)
        ax.set_xlabel("Predicted labels")
        ax.set_ylabel("True labels")
        ax.set_title("Confusion Matrix")
        st.pyplot(fig)

        # Mostrar el classification report
        report = classification_report(y_test, y_pred)
        st.markdown("Classification Report:")
        st.text(report)
