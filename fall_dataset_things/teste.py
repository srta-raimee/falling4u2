import os
from ultralytics import YOLO
from PIL import Image

# kfold stratified 10 folds - fold 2 model
# best_modelo_kfold = 'runs/detect/kfold_kfold_dataset_stratified_10_10folds_fold_2/weights/best.pt'
best_modelo_kfold = 'runs/detect/kfold_kfold_dataset_stratified_10_10folds_fold_2/weights/best.pt'
best_modelo_holdout = 'runs/detect/split_8020_stratified_run1/weights/best.pt'

MODELO_1 = YOLO(best_modelo_kfold)
MODELO_2 = YOLO(best_modelo_holdout)

pasta_de_entrada = 'arquivos_teste'
pasta_de_saida = 'resultados_predicao_kfold'

os.makedirs(pasta_de_saida, exist_ok=True)

for nome_arquivo in os.listdir(pasta_de_entrada):
    if nome_arquivo.lower().endswith(('.png', '.jpg', '.jpeg', '.JPG')):
        
        caminho_da_imagem = os.path.join(pasta_de_entrada, nome_arquivo)
        print(f"Processando a imagem: {caminho_da_imagem}...")

        results = MODELO_1.predict(caminho_da_imagem)

        for r in results:
            im_array = r.plot()  # plota as caixas na imagem
            im = Image.fromarray(im_array[..., ::-1])  # Converte para formato de imagem
            
            base, ext = os.path.splitext(nome_arquivo)
            caminho_salvo = os.path.join(pasta_de_saida, f"{base}_predito{ext}")
            
            #im.show() 
            im.save(caminho_salvo) # Salva a imagem com a predição
            print(f"--> Resultado salvo em: {caminho_salvo}")

print("\nProcesso concluído!")