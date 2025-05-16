from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

app = Flask(__name__)
app.secret_key = "clave_secreta"

# === CARGAR MODELOS ===
modelo_logistica = joblib.load("modelos/modelo_logistica.pkl")
modelo_mlp = joblib.load("modelos/modelo_mlp.pkl")
modelo_svm = joblib.load("modelos/modelo_svm.pkl")
modelo_fcm = joblib.load("modelos/modelo_fcm.pkl")

# === CARGAR ESCALADORES === 
escalador_fcm = joblib.load("modelos/escalador_fcm.pkl")
escalador_logistica = joblib.load("modelos/escalador_logistica.pkl")
escalador_mlp = joblib.load("modelos/escalador_mlp.pkl")
escalador_svm = joblib.load("modelos/escalador_svm.pkl")

# === VARIABLES ===
columnas_completas = [f"C{i}" for i in range(1, 31)]  # C1 a C30
columnas_sin_c20_c21 = [c for c in columnas_completas if c not in ["C20", "C21"]]

# === FUNCIÓN FCM ===
def predecir_fcm(X, pesos):
    # X es un numpy ndarray con 28 columnas (sin C20 y C21)
    activacion = np.dot(X, pesos)  # (n_samples x 28) · (28,)
    return (activacion > 0.5).astype(int)

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/form_individual')
def form_individual():
    return render_template("form_individual.html")

@app.route('/form_lote')
def form_lote():
    return render_template("form_lote.html")

@app.route('/predecir', methods=['POST'])
def predecir():
    try:
        datos = [float(request.form[f"C{i}"]) for i in range(1, 31)]
        modelo_seleccionado = request.form["modelo"]

        df_completo = pd.DataFrame([datos], columns=columnas_completas)

        if modelo_seleccionado == "logistica":
            df = df_completo[columnas_sin_c20_c21]
            X_np = df.to_numpy()
            pred = modelo_logistica.predict(escalador_logistica.transform(X_np))[0]
            resultado = ("Regresión Logística", pred)

        
        elif modelo_seleccionado == "fcm":
                df_temp = df_completo[columnas_completas].copy()
                X_np = df_temp.to_numpy()
                X_scaled = escalador_fcm.transform(X_np)
                pred = modelo_fcm.predict(X_scaled)[0]  # predicción individual (el primer resultado)
                resultado = ("Mapa Cognitivo Difuso (FCM)", pred)

        else:
            df = df_completo[columnas_completas]
            X_np = df.to_numpy()
            if modelo_seleccionado == "mlp":
                pred = modelo_mlp.predict(escalador_mlp.transform(X_np))[0]
                resultado = ("Red Neuronal (MLP)", pred)
            elif modelo_seleccionado == "svm":
                pred = modelo_svm.predict(escalador_svm.transform(X_np))[0]
                resultado = ("Máquina SVM", pred)
            else:
                return "Modelo seleccionado no válido."

        return render_template("resultado_individual.html", resultado=resultado[1], modelo_usado=resultado[0])

    except Exception as e:
        return f"Error en predicción individual: {str(e)}"

@app.route('/lote', methods=['POST'])
def lote():
    try:
        archivo = request.files['archivo']
        modelo_seleccionado = request.form["modelo"]

        nombre_archivo = archivo.filename
        if not nombre_archivo.endswith((".xlsx", ".xls", ".csv")):
            return "Error: Solo se permiten archivos .xlsx, .xls o .csv"

        if nombre_archivo.endswith(".csv"):
            df = pd.read_csv(archivo)
        else:
            df = pd.read_excel(archivo)

        y = df["C31"]

        if modelo_seleccionado == "logistica":
            X = df[columnas_sin_c20_c21]
            X_np = X.to_numpy()
            pred = modelo_logistica.predict(escalador_logistica.transform(X_np))

        elif modelo_seleccionado == "fcm":
            df_temp = df[columnas_completas].copy()
            X_np = df_temp.to_numpy()
            X_scaled = escalador_fcm.transform(X_np)
            pred = modelo_fcm.predict(X_scaled)
            resultado = ("Mapa Cognitivo Difuso (FCM)", pred)



        else:
            X = df[columnas_completas]
            X_np = X.to_numpy()
            if modelo_seleccionado == "mlp":
                pred = modelo_mlp.predict(escalador_mlp.transform(X_np))
            elif modelo_seleccionado == "svm":
                pred = modelo_svm.predict(escalador_svm.transform(X_np))
            else:
                return "Modelo seleccionado no válido."

        df_resultado = df.copy()
        df_resultado["Predicción"] = np.where(pred == 1, "FGR", "Normal")

        cm = confusion_matrix(y, pred)
        acc = accuracy_score(y, pred)
        tn, fp, fn, tp = cm.ravel()

        plt.figure()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Purples', xticklabels=['Normal', 'FGR'], yticklabels=['Normal', 'FGR'])
        plt.title("Matriz de Confusión")
        path_cm = f"static/cm_{modelo_seleccionado}.png"
        plt.savefig(path_cm, bbox_inches="tight")
        plt.close()

        interpretacion = {
            "TN": tn,
            "FP": fp,
            "FN": fn,
            "TP": tp,
            "descripcion": f"- TN (Normal bien clasificados): {tn}\n- FP (Normal clasificados como FGR): {fp}\n- FN (FGR clasificados como Normal): {fn}\n- TP (FGR correctamente detectados): {tp}"
        }

        modelo_nombre_map = {
            "logistica": "Regresión Logística",
            "mlp": "Red Neuronal (MLP)",
            "svm": "Máquina SVM",
            "fcm": "Mapa Cognitivo Difuso (FCM)"
        }

        return render_template(
            "resultado_lote.html",
            exactitud=acc,
            df_resultado=df_resultado.to_dict(orient='records'),
            matriz_img=path_cm,
            interpretacion=interpretacion,
            modelo_usado=modelo_nombre_map[modelo_seleccionado]
        )

    except Exception as e:
        return f"Error en predicción por lote: {str(e)}"

if __name__ == '__main__':
    app.run(debug=True)
