import os
import shutil
import requests
from zipfile import ZipFile
from io import BytesIO

def download_and_prepare_dataset(target_dir='./dataset'):
    # 1. Crea la cartella dataset se non esiste
    os.makedirs(target_dir, exist_ok=True)
    
    dataset_url = 'http://cs231n.stanford.edu/tiny-imagenet-200.zip' # [cite: 1, 10]
    
    # 2. Scarica lo zip
    print(f"Scaricando Tiny ImageNet da {dataset_url}...")
    response = requests.get(dataset_url) # [cite: 10]
    
    if response.status_code == 200: # [cite: 10]
        # 3. Estrai lo zip direttamente nella cartella target
        print("Download completato! Estrazione in corso...")
        with ZipFile(BytesIO(response.content)) as zip_file: # [cite: 10]
            zip_file.extractall(target_dir) # [cite: 10]
        print("Estrazione completata!")
    else:
        print("Errore nel download del dataset.")
        return

    # 4. Riorganizza la cartella di validazione
    # Tiny ImageNet ha le immagini di validazione in un'unica cartella. 
    # Dobbiamo dividerle in sottocartelle per classe usando il file val_annotations.txt per far funzionare ImageFolder.
    print("Riorganizzando la cartella di validazione...")
    
    val_dir = os.path.join(target_dir, 'tiny-imagenet-200', 'val')
    annotations_file = os.path.join(val_dir, 'val_annotations.txt') # [cite: 1]
    
    if os.path.exists(annotations_file):
        with open(annotations_file) as f: # [cite: 1]
            for line in f: # [cite: 1]
                fn, cls, *_ = line.split('\t') # [cite: 1]
                
                # Crea la cartella della classe se non esiste
                class_dir = os.path.join(val_dir, cls)
                os.makedirs(class_dir, exist_ok=True) # [cite: 1]
                
                # Sposta l'immagine nella cartella della sua classe
                src_path = os.path.join(val_dir, 'images', fn)
                dst_path = os.path.join(class_dir, fn)
                
                if os.path.exists(src_path):
                    shutil.copyfile(src_path, dst_path) # [cite: 1]
        
        # Elimina la vecchia cartella 'images' che ora è vuota
        shutil.rmtree(os.path.join(val_dir, 'images')) # [cite: 1]
        print("Dataset pronto all'uso!")
    else:
        print("File di annotazione non trovato. Assicurati che l'estrazione sia andata a buon fine.")

if __name__ == '__main__':
    download_and_prepare_dataset()