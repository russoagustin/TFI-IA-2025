from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
import numpy as np

def umbrales(y_true,y_pred_probs):
  mejores_umbrales = [] #Declaro una lista vacía

  for i, clase in enumerate(LABEL_COLUMS): #Recorro 5 clases (i de 0 a 4).
    mejor_f1 = 0
    mejor_umbral = 0.5  # Por defecto 0.5

    # Generamos 100 umbrales entre 0.0 y 1.0, paso ~0.01
    lista_umbrales = np.linspace(0.0, 1.0, 100)

    for umbral_actual in lista_umbrales:  #Bucle interno
        #Obtengo: "todas las filas para la columna i" → todas las probabilidades para el diagnóstico i.
        probabilidades_clase = y_pred_probs[:, i]   #  : -> significa todas las filas.

        # Convertimos las probabilidades de cada fila a predicciones 0 o 1 según el umbral actual
        predicciones_binarias = []
        for prob in probabilidades_clase:
            if prob >= umbral_actual:
                predicciones_binarias.append(1)
            else:
                predicciones_binarias.append(0)

        # calcula el F1 comparando los arrays correctos y los de predicción elemento a elemento.
        f1 = f1_score(y_true[:, i], predicciones_binarias)
        # Guard si encuentor el mejor F1
        if f1 > mejor_f1:
            mejor_f1 = f1
            mejor_umbral = umbral_actual
    #Guardo el mejor umbral en la lista de 5 elementos
    mejores_umbrales.append(mejor_umbral)
    print(f"Para el diagnostico '{clase}' el mejor umbral es {mejor_umbral:.2f} con F1={mejor_f1:.4f}")
  return mejores_umbrales

def mostrar_metricas(y_true, y_pred_probs, predicciones):
  print("métricas")

  for i , task in enumerate(LABEL_COLUMS):
    print(f"\n----{task}----")
    try:
      auc = roc_auc_score(y_true[:,i], y_pred_probs[:,i])
      print(f"ROC AUC: {auc}")
    except ValueError:
      print("ROC AUC no se puede calcular")

    precision = precision_score(y_true[:,i], predicciones[:,i], zero_division = 0)
    print(f"Precisión: {precision}")

    recall = recall_score(y_true[:,i], predicciones[:,i], zero_division=0)
    print(f"Recall: {recall}")

    f1 = f1_score(y_true[:,i], predicciones[:,i], zero_division=0)
    print(f"Puntuación F1: {f1:.4f}")