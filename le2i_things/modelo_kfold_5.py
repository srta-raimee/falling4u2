import os
import shutil
import csv
import time
import numpy as np
from ultralytics import YOLO
import matplotlib
matplotlib.use('Agg')
import pandas as pd

def format_time(seconds):
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{int(hours):02d}h {int(minutes):02d}m {seconds:.2f}s"

def run_kfold_training(kfold_base_dir, output_csv, k_folds, epochs, batch_size, model_name, patience):
    CSV_RESULTS_FILE = output_csv
    CSV_FOLDS_LOG_FILE = f'log_detalhado_{os.path.basename(kfold_base_dir)}.csv'
    K_FOLDS = k_folds
    EPOCHS = epochs
    BATCH_SIZE = batch_size
    NOME_MODELO_INICIAL = model_name
    PATIENCE = patience
    EXPERIMENT_PREFIX = f'run_{os.path.basename(kfold_base_dir)}'

    if not os.path.isdir(kfold_base_dir):
        print(f"ERRO: O diretório do K-Fold '{kfold_base_dir}' não foi encontrado. Rode o script de preparação primeiro.")
        return

    completed_folds = []
    if os.path.exists(CSV_FOLDS_LOG_FILE):
        log_df = pd.read_csv(CSV_FOLDS_LOG_FILE)
        completed_runs = log_df[(log_df['experimento_base'] == EXPERIMENT_PREFIX) & (log_df['k_total'] == K_FOLDS)]
        if not completed_runs.empty:
            completed_folds = completed_runs['fold_num'].tolist()

    all_fold_metrics = []
    total_start_time = time.perf_counter()

    for fold_num in range(1, K_FOLDS + 1):
        if fold_num in completed_folds:
            print(f"\nPULANDO FOLD {fold_num}/{K_FOLDS} - JÁ COMPLETADO ANTERIORMENTE.")
            fold_metrics_saved = pd.read_csv(CSV_FOLDS_LOG_FILE)
            metric_row = fold_metrics_saved[(fold_metrics_saved['experimento_base'] == EXPERIMENT_PREFIX) & (fold_metrics_saved['fold_num'] == fold_num)].iloc[0]
            all_fold_metrics.append({
                'mAP50_95': metric_row['mAP50_95'], 'mAP50': metric_row['mAP50'],
                'precision': metric_row['precisao'], 'recall': metric_row['recall']
            })
            continue

        print("\n" + "="*60)
        print(f"INICIANDO TREINAMENTO DO FOLD {fold_num}/{K_FOLDS}")
        print("="*60)
        
        fold_dir = os.path.join(kfold_base_dir, f"fold_{fold_num}")
        
        yaml_path = os.path.join(fold_dir, 'data.yaml')
        with open(yaml_path, 'w') as f:
            f.write(f"path: {os.path.abspath(fold_dir)}\n")
            f.write("train: images/train\n")
            f.write("val: images/val\n")
            f.write("nc: 2\n")
            f.write("names: ['fall', 'not_fall_activity']\n")
        
        model = YOLO(NOME_MODELO_INICIAL)
        
        fold_experiment_name = f"{EXPERIMENT_PREFIX}_fold_{fold_num}"
        model.train(
            data=yaml_path, 
            epochs=EPOCHS, 
            batch=BATCH_SIZE, 
            device=0, 
            name=fold_experiment_name,
            patience=PATIENCE
        )
        
        best_model = YOLO(f'runs/detect/{fold_experiment_name}/weights/best.pt')
        results = best_model.val()

        fold_metrics = {
            'mAP50_95': results.box.map, 'mAP50': results.box.map50,
            'precision': results.box.mp, 'recall': results.box.mr
        }
        all_fold_metrics.append(fold_metrics)
        print(f"Métricas do Fold {fold_num}: {fold_metrics}")

        header_fold = ['experimento_base', 'k_total', 'fold_num', 'mAP50_95', 'mAP50', 'precisao', 'recall']
        data_row_fold = [
            EXPERIMENT_PREFIX, K_FOLDS, fold_num,
            f"{fold_metrics['mAP50_95']:.4f}", f"{fold_metrics['mAP50']:.4f}",
            f"{fold_metrics['precision']:.4f}", f"{fold_metrics['recall']:.4f}"
        ]
        file_exists_fold = os.path.isfile(CSV_FOLDS_LOG_FILE)
        with open(CSV_FOLDS_LOG_FILE, 'a', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            if not file_exists_fold:
                writer.writerow(header_fold)
            writer.writerow(data_row_fold)
        print(f"Progresso do Fold {fold_num} salvo em '{CSV_FOLDS_LOG_FILE}'")
        
    total_end_time = time.perf_counter()
    total_duration_str = format_time(time.perf_counter() - total_start_time)
    
    if len(all_fold_metrics) < K_FOLDS:
        print("\nAVISO: O experimento não foi concluído. A média final não será calculada.")
        return

    avg_metrics = {
        'avg_mAP50_95': np.mean([m['mAP50_95'] for m in all_fold_metrics]),
        'std_mAP50_95': np.std([m['mAP50_95'] for m in all_fold_metrics]),
        'avg_mAP50': np.mean([m['mAP50'] for m in all_fold_metrics]),
        'std_mAP50': np.std([m['mAP50'] for m in all_fold_metrics]),
        'avg_precision': np.mean([m['precision'] for m in all_fold_metrics]),
        'std_precision': np.std([m['precision'] for m in all_fold_metrics]),
        'avg_recall': np.mean([m['recall'] for m in all_fold_metrics]),
        'std_recall': np.std([m['recall'] for m in all_fold_metrics]),
    }

    print("\n" + "="*60)
    print(f"RESULTADO FINAL DO K-FOLD ({K_FOLDS} FOLDS)")
    print("="*60)
    print(f"Tempo Total de Execução: {total_duration_str}")
    print(f"Métricas (Média ± Desvio Padrão):")
    print(f"mAP50-95: {avg_metrics['avg_mAP50_95']:.4f} ± {avg_metrics['std_mAP50_95']:.4f}")
    print(f"mAP50:    {avg_metrics['avg_mAP50']:.4f} ± {avg_metrics['std_mAP50']:.4f}")
    print(f"Precisão: {avg_metrics['avg_precision']:.4f} ± {avg_metrics['std_precision']:.4f}")
    print(f"Recall:   {avg_metrics['avg_recall']:.4f} ± {avg_metrics['std_recall']:.4f}")

    header = ['experimento_base', 'k_folds', 'tempo_total', 'avg_mAP50_95', 'std_mAP50_95', 'avg_mAP50', 'std_mAP50', 'avg_precision', 'std_precision', 'avg_recall', 'std_recall']
    data_row = [
        EXPERIMENT_PREFIX, K_FOLDS, total_duration_str,
        f"{avg_metrics['avg_mAP50_95']:.4f}", f"{avg_metrics['std_mAP50_95']:.4f}",
        f"{avg_metrics['avg_mAP50']:.4f}", f"{avg_metrics['std_mAP50']:.4f}",
        f"{avg_metrics['avg_precision']:.4f}", f"{avg_metrics['std_precision']:.4f}",
        f"{avg_metrics['avg_recall']:.4f}", f"{avg_metrics['std_recall']:.4f}"
    ]

    file_exists = os.path.isfile(CSV_RESULTS_FILE)
    with open(CSV_RESULTS_FILE, 'a', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        if not file_exists:
            writer.writerow(header)
        writer.writerow(data_row)
    
    print(f"\nResultados médios salvos em '{CSV_RESULTS_FILE}'")


if __name__ == '__main__':
    KFOLD_DATA_DIR = 'LE2I_kfold_stratified_5'
    OUTPUT_CSV_FILE = 'resultados_gerais.csv'
    NUM_FOLDS = 5
    NUM_EPOCHS = 200
    PATIENCE = 15
    BATCH_SIZE = 8
    MODELO = "yolo11n.pt"

    run_kfold_training(
        kfold_base_dir=KFOLD_DATA_DIR,
        output_csv=OUTPUT_CSV_FILE,
        k_folds=NUM_FOLDS,
        epochs=NUM_EPOCHS,
        patience=PATIENCE,
        batch_size=BATCH_SIZE,
        model_name=MODELO
    )