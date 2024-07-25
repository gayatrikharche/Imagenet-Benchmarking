import os
import zipfile
import random

def shuffle_and_split(file_path, output_dir, chunk_size):
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    with zipfile.ZipFile(file_path, 'r') as zip_ref:
        file_names = zip_ref.namelist()
        random.shuffle(file_names)

        chunk = []
        total_size = 0
        chunk_index = 0
        
        for file_name in file_names:
            info = zip_ref.getinfo(file_name)
            total_size += info.file_size
            chunk.append(file_name)
            
            if total_size >= chunk_size:
                with zipfile.ZipFile(f'{output_dir}/ILSVRC.zip.part{chunk_index}', 'w') as chunk_zip:
                    for name in chunk:
                        chunk_zip.writestr(name, zip_ref.read(name))
                chunk_index += 1
                chunk = []
                total_size = 0

        # Write any remaining files
        if chunk:
            with zipfile.ZipFile(f'{output_dir}/ILSVRC.zip.part{chunk_index}', 'w') as chunk_zip:
                for name in chunk:
                    chunk_zip.writestr(name, zip_ref.read(name))

if __name__ == "__main__":
    shuffle_and_split('ILSVRC.zip', 'chunks', 4990 * 1024 * 1024)  # 5GB chunk size
