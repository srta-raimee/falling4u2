import os
import shutil
from tqdm import tqdm

def limpar_anotacoes_le2i():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.join(script_dir, 'le2i')
    
    fall_scenarios = ['Coffee_room_01', 'Coffee_room_02', 'Home_01', 'Home_02']
    
    for scenario in fall_scenarios:
        ann_dir_opt1 = os.path.join(base_dir, scenario, 'Annotation_files')
        ann_dir_opt2 = os.path.join(base_dir, scenario, 'Annotations_files')
        
        ann_dir = ann_dir_opt1 if os.path.isdir(ann_dir_opt1) else ann_dir_opt2
        if not os.path.isdir(ann_dir):
            continue

        output_ann_dir = os.path.join(base_dir, scenario, 'Annotations_Cleaned')
        if os.path.exists(output_ann_dir):
            shutil.rmtree(output_ann_dir)
        os.makedirs(output_ann_dir)

        print(f"\nLimpando e padronizando anotações de '{scenario}'...")
        for filename in tqdm(os.listdir(ann_dir)):
            if not filename.endswith('.txt'):
                continue
            
            with open(os.path.join(ann_dir, filename), 'r') as f:
                lines = [line.strip() for line in f.readlines() if line.strip()]

            header_lines = []
            coord_lines = []
            
            for line in lines:
                if ',' in line:
                    coord_lines.append(line)
                elif line.isdigit() and int(line) > 0:
                    header_lines.append(line)

            new_content = ""
            if len(header_lines) == 2:
                new_content += f"{header_lines[0]}\n{header_lines[1]}\n"
            
            new_content += "\n".join(coord_lines)
            
            with open(os.path.join(output_ann_dir, filename), 'w') as out_f:
                out_f.write(new_content)
                
    print("\nFaxina concluída! Anotações padronizadas salvas nas pastas 'Annotations_Cleaned'.")

if __name__ == '__main__':
    limpar_anotacoes_le2i()