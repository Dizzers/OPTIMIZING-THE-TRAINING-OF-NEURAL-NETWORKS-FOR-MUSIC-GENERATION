import os
from pathlib import Path
import mido
from mido import MidiFile


def count_notes_in_midi(midi_path):
    """Подсчитывает количество нот в MIDI-файле"""
    try:
        midi = MidiFile(midi_path)
        note_count = 0

        for track in midi.tracks:
            for msg in track:
                
                if msg.type == 'note_on' and msg.velocity > 0:
                    note_count += 1

        return note_count
    except Exception as e:
        print(f"Ошибка при обработке файла {midi_path}: {e}")
        return 0


def analyze_midi_directory(directory_path):
    """Анализирует директорию с MIDI-файлами"""
    try:
       
        directory = Path(directory_path).expanduser().resolve()

        print(f"Ищем MIDI-файлы в: {directory}")
        print("-" * 80)

        if not directory.exists():
            print(f"ОШИБКА: Директория не существует: {directory}")
            print("Проверьте правильность пути и убедитесь, что диск подключен.")
            return

        
        midi_files = list(directory.rglob("*.mid")) + list(directory.rglob("*.midi"))

        if not midi_files:
            print("MIDI-файлы не найдены! Проверьте расширения файлов (.mid или .midi)")
            print("Содержимое директории:")
            for item in directory.iterdir():
                print(f"  {item.name}/" if item.is_dir() else f"  {item.name}")
            return

        print(f"Найдено MIDI-файлов: {len(midi_files)}")
        print("-" * 80)

        
        folders_dict = {}
        for midi_file in midi_files:
            folder = str(midi_file.parent.relative_to(directory))
            if folder == ".":
                folder = "Корневая папка"

            if folder not in folders_dict:
                folders_dict[folder] = []
            folders_dict[folder].append(midi_file)

        total_notes = 0
        file_info = []

        
        for folder, files in folders_dict.items():
            print(f"\nПапка: {folder}")
            print(f"Файлов в папке: {len(files)}")

            folder_notes = 0
            for i, midi_file in enumerate(files, 1):
                notes = count_notes_in_midi(midi_file)
                total_notes += notes
                folder_notes += notes
                file_info.append((str(midi_file.relative_to(directory)), notes))

                
                if i <= 3:
                    print(f"  {i}. {midi_file.name}: {notes} нот")
                elif i == 4 and len(files) > 3:
                    print(f"  ... и еще {len(files) - 3} файлов")

            print(f"  Всего нот в папке: {folder_notes}")

        print("-" * 80)
        print(f"ИТОГО:")
        print(f"Количество MIDI-файлов: {len(midi_files)}")
        print(f"Общее количество нот: {total_notes}")

        
        if len(midi_files) > 0:
            avg_notes = total_notes / len(midi_files)
            print(f"Среднее количество нот на файл: {avg_notes:.1f}")

            
            if file_info:
                max_file = max(file_info, key=lambda x: x[1])
                min_file = min(file_info, key=lambda x: x[1])

                print(f"\nФайл с наибольшим количеством нот:")
                print(f"  {max_file[0]}: {max_file[1]} нот")
                print(f"\nФайл с наименьшим количеством нот:")
                print(f"  {min_file[0]}: {min_file[1]} нот")

                
                print(f"\nСтатистика по папкам:")
                folder_stats = {}
                for file_path, notes in file_info:
                    path_obj = Path(file_path)
                    folder = str(path_obj.parent)
                    if folder == ".":
                        folder = "Корневая папка"

                    if folder not in folder_stats:
                        folder_stats[folder] = {'files': 0, 'notes': 0}
                    folder_stats[folder]['files'] += 1
                    folder_stats[folder]['notes'] += notes

                for folder, stats in folder_stats.items():
                    avg = stats['notes'] / stats['files'] if stats['files'] > 0 else 0
                    print(
                        f"  {folder}: {stats['files']} файлов, {stats['notes']} нот (в среднем {avg:.1f} нот на файл)")

    except Exception as e:
        print(f"Произошла ошибка: {e}")
        import traceback
        traceback.print_exc()


def get_directory_stats(directory_path):
    """Простая функция для быстрого получения статистики"""
    try:
        directory = Path(directory_path).expanduser().resolve()

        if not directory.exists():
            return None

        
        midi_files = list(directory.rglob("*.mid")) + list(directory.rglob("*.midi"))

        if not midi_files:
            return 0, 0, 0

        total_notes = 0
        for midi_file in midi_files:
            notes = count_notes_in_midi(midi_file)
            total_notes += notes

        return len(midi_files), total_notes, total_notes / len(midi_files) if midi_files else 0

    except Exception as e:
        print(f"Ошибка при получении статистики: {e}")
        return None


if __name__ == "__main__":
    
    paths_to_try = []

    
    user_input = input("Введите путь к директории с MIDI-файлами (нажмите Enter для текущей директории): ").strip()

    if user_input:
        paths_to_try.append(user_input)

   
    paths_to_try.append("/Volumes/T7/Университет/PythonProject/TISPIS/archive")

    paths_to_try.append(".")

    paths_to_try.append("~/Desktop")

    for path in paths_to_try:
        print(f"\nПробуем путь: {path}")
        stats = get_directory_stats(path)

        if stats is None:
            print(f"  Директория не существует или недоступна")
        elif stats[0] == 0:
            print(f"  MIDI-файлы не найдены")
        else:
            print(f"  Найдено: {stats[0]} MIDI-файлов, {stats[1]} нот")
            break


    analyze_midi_directory(user_input if user_input else ".")

    analyze_midi_directory("/Volumes/T7/Университет/PythonProject/TISPIS/archive")
