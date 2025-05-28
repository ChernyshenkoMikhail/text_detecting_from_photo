from ultralytics import YOLO
from collections import Counter
from pathlib import Path

def detect_object (model: YOLO, image_path: str) -> None:
    """
    Выполняет детекцию объектов на изображении и сохраняет результат в указанный файл.
    Также выводит в теримнал количество найденных объектов по категориям.
    
    :param model: загруженная модель YOLO
    :param image_path: путь к изображению, на котором нужно произвести детекцию

    """
    print ("[INFO] загружается информация об объекте...")
    results = model (image_path, verbose=False)[0]

    if results.names and results.boxes is not None:
        # Получаем список идентификаторов классов объектов
        labels = results.boxes.cls.tolist ()
        # Преобразуем ID в название классов
        label_names = [results.names [int (cls)] for cls in labels]
        # Считаем количество каждого уникального объекта
        counts = Counter (label_names)
        
        print ("[INFO] Обнаруженные объекты: ")
        
        for label, count in  counts.items():
            print (f'[+] {label}: {count}')
            
    else:
        print (f'[!] Объекты не обнаружены!')
        
    save_path = results.save (filename=f'RESULT_{Path (image_path).stem}.png')
    print (f'[+] Результат сохранён в файл: {save_path}')

def main ():
    model = YOLO ('yolov8n.pt')
    
    detect_object (model, '...ссылка на репозиторий, где сохранён файл с фото')

if __name__  == '__main__':
    main ()