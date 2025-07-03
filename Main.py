from Model import crear_modelo
from Data import obtener_dataset, crear_dataFrame
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
from Config import LABEL_COLUMS
import tensorflow as tf

def main():
    model = crear_modelo()
    train_df = crear_dataFrame('train.csv',0.05)
    valid_df = crear_dataFrame('valid.csv',0.5)

    train_dataset = obtener_dataset(train_df)
    valid_dataset = obtener_dataset(valid_df)

    model.compile(optimizer = 'adam',
                   loss = 'binary_crossentropy', 
                   metrics=[tf.keras.metrics.AUC(name='auc')])
    
    model.fit(train_dataset, epochs= 2, validation_data= valid_dataset)

    y_pred_probs = model.predict(valid_dataset)
    y_pred = (y_pred_probs > 0.15).astype(int)
    y_true = valid_df[LABEL_COLUMS].values

    print("métricas")

    for i , task in enumerate(LABEL_COLUMS):
        print(f"\n----{task}----")
        try:
            auc = roc_auc_score(y_true[:,i], y_pred_probs[:,i])
            print(f"ROC AUC: {auc}")
        except ValueError:
            print("ROC AUC no se puede calcular")

        precision = precision_score(y_true[:,i], y_pred[:,i], zero_division = 0)
        print(f"Precisión: {precision}")

        recall = recall_score(y_true[:,i], y_pred[:,i], zero_division=0)
        print(f"Recall: {recall}")

        f1 = f1_score(y_true[:,i], y_pred[:,i], zero_division=0)
        print(f"Puntuación F1: {f1:.4f}")



main()