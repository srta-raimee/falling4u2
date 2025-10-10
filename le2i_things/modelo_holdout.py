import os
import csv
import time
from ultralytics import YOLO

def format_time(seconds):
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{int(hours):02d}h {int(minutes):02d}m {seconds:.2f}s"

def run_single_experiment(yaml_path, experiment_name, output_csv, epochs=200, patience=10, model_name="yolo11n.pt"):
    start_time = time.perf_counter()
    model = YOLO(model_name)

    model.train(data=yaml_path, epochs=epochs, batch=8, device=0, name=experiment_name, patience=patience)
    
    training_duration = format_time(time.perf_counter() - start_time)

    best_model = YOLO(f'runs/detect/{experiment_name}/weights/best.pt')
    results = best_model.val()
    
    metrics = { 'mAP50_95': results.box.map, 'mAP50': results.box.map50, 'precision': results.box.mp, 'recall': results.box.mr }

    print("\n--- Relatório Final ---")
    print(f"Experimento: {experiment_name}")
    print(f"Tempo Total: {training_duration}")
    print(f"Métricas Finais: {metrics}")

    header = ['experimento', 'dataset', 'modelo', 'tempo_treino', 'mAP50_95', 'mAP50', 'precisao', 'recall']
    data_row = [
        experiment_name, yaml_path, model_name, training_duration,
        f"{metrics['mAP50_95']:.4f}", f"{metrics['mAP50']:.4f}", f"{metrics['precision']:.4f}", f"{metrics['recall']:.4f}"
    ]
    
    file_exists = os.path.isfile(output_csv)
    with open(output_csv, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(header)
        writer.writerow(data_row)
        
    print(f"Resultados salvos em '{output_csv}'")

if __name__ == '__main__':
    run_single_experiment(
        yaml_path='yaml/le2i_holdout_8020.yaml',
        experiment_name='LE2I_Holdout_8020_Stratified',
        output_csv='resultados_holdout_8020_stratified.csv'
    )