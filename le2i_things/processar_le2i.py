import os
import shutil
import cv2
from tqdm import tqdm

def processar_dataset_le2i_final():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.join(script_dir, 'le2i')
    output_dir = os.path.join(script_dir, 'LE2I_YOLO_FORMAT')
    
    fall_scenarios = ['Coffee_room_01', 'Coffee_room_02', 'Home_01', 'Home_02']
    nofall_scenarios = ['Lecture_room', 'Office']

    output_img_dir = os.path.join(output_dir, 'images')
    output_label_dir = os.path.join(output_dir, 'labels')

    if os.path.exists(output_dir):
        print(f"Limpando o diretório de saída '{output_dir}'...")
        shutil.rmtree(output_dir)

    os.makedirs(output_img_dir, exist_ok=True)
    os.makedirs(output_label_dir, exist_ok=True)
    
    all_scenarios = fall_scenarios + nofall_scenarios
    
    print("\n--- ETAPA 1: Extraindo todos os frames de todos os vídeos ---")
    for scenario in all_scenarios:
        video_source_subfolder = 'Videos' if scenario in fall_scenarios else ''
        video_dir = os.path.join(base_dir, scenario, video_source_subfolder)
        
        if not os.path.isdir(video_dir) and scenario not in fall_scenarios:
            video_dir = os.path.join(base_dir, scenario)
        
        if not os.path.isdir(video_dir):
            print(f"AVISO: Diretório de vídeo não encontrado para '{scenario}', pulando.")
            continue

        for video_filename in tqdm(os.listdir(video_dir), desc=f"Extraindo de '{scenario}'"):
            if not video_filename.lower().endswith(('.avi', '.mp4')):
                continue

            video_path = os.path.join(video_dir, video_filename)
            video_name = os.path.splitext(video_filename)[0]
            
            cap = cv2.VideoCapture(video_path)
            frame_idx = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame_name = f"{video_name}_frame_{frame_idx:05d}"
                cv2.imwrite(os.path.join(output_img_dir, f"{frame_name}.jpg"), frame)
                frame_idx += 1
            cap.release()

    print("\n--- ETAPA 2: Convertendo anotações para o formato YOLO ---")
    for scenario in fall_scenarios:
        video_dir = os.path.join(base_dir, scenario, 'Videos')
        ann_dir = os.path.join(base_dir, scenario, 'Annotations_Cleaned')
        
        if not os.path.isdir(ann_dir):
            print(f"AVISO: Pasta 'Annotations_Cleaned' não encontrada para '{scenario}'. Rode o script de limpeza primeiro.")
            continue

        for ann_filename in tqdm(os.listdir(ann_dir), desc=f"Convertendo de '{scenario}'"):
            if not ann_filename.endswith('.txt'):
                continue

            video_name = os.path.splitext(ann_filename)[0]
            video_path = os.path.join(video_dir, f"{video_name}.avi")
            if not os.path.exists(video_path):
                video_path = os.path.join(video_dir, f"{video_name}.mp4")
                if not os.path.exists(video_path):
                    continue
            
            cap = cv2.VideoCapture(video_path)
            vid_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            vid_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
            if vid_w == 0 or vid_h == 0:
                continue

            with open(os.path.join(ann_dir, ann_filename), 'r') as f:
                lines = [line.strip() for line in f.readlines() if line.strip()]
                if not lines:
                    continue
                
                if ',' in lines[0]:
                    coord_lines = lines
                    has_fall_info = False
                else:
                    start_fall_frame = int(lines[0])
                    end_fall_frame = int(lines[1])
                    coord_lines = lines[2:]
                    has_fall_info = True

                for line in coord_lines:
                    parts = line.split(',')
                    if len(parts) != 6:
                        continue
                    
                    frame_num, pose_id, x_min, y_min, x_max, y_max = map(int, parts)
                    if x_max == 0 and y_max == 0:
                        continue

                    class_id = 0 if has_fall_info and start_fall_frame <= frame_num <= end_fall_frame else 1

                    box_w = x_max - x_min
                    box_h = y_max - y_min
                    x_center = x_min + box_w / 2
                    y_center = y_min + box_h / 2

                    x_center_norm = x_center / vid_w
                    y_center_norm = y_center / vid_h
                    w_norm = box_w / vid_w
                    h_norm = box_h / vid_h
                    
                    yolo_line = f"{class_id} {x_center_norm:.6f} {y_center_norm:.6f} {w_norm:.6f} {h_norm:.6f}\n"
                    
                    label_filename = f"{video_name}_frame_{frame_num:05d}.txt"
                    with open(os.path.join(output_label_dir, label_filename), 'w') as out_f:
                        out_f.write(yolo_line)

    print(f"\nProcesso concluído! Dataset em formato YOLO salvo em '{output_dir}'.")

if __name__ == '__main__':
    processar_dataset_le2i_final()