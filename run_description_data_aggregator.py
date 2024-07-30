import os
import datasets

def aggregate(path,save_pth):
    all_files = os.listdir(path)

    # Filter files with prefix "xyz"
    json_files = [file for file in all_files if file.endswith(".json")]
    combined_dataset = datasets.Dataset.from_dict({})
    for path in json_files:
        data = datasets.Dataset.from_json(path+path)
        combined_dataset = datasets.concatenate_datasets([combined_dataset, data])
    
    combined_dataset.to_json(save_pth + "sft_finetune_dataset_full.json")
    return 


if __name__ == "__main__":
    folder_path = "C:/Users/gneeraj/Desktop/Projects/cypher/sft_dataset_files/"
    save_path = "C:/Users/gneeraj/Desktop/Projects/cypher/Datasets/"
    aggregate(folder_path, save_path)