import torch
import os

envs = ['open_field', 'four_rooms', 't_maze', 'random_barriers']
print("Environment | RMSE")
print("-" * 20)

for env in envs:
    filename = f"trained_decoder_{env}.pth"
    if os.path.exists(filename):
        try:
            # Load using CPU to avoid CUDA errors if just checking metadata
            # Set weights_only=False to allow loading numpy scalars in metadata
            data = torch.load(filename, map_location=torch.device('cpu'), weights_only=False)
            # Check where metadata is stored. Based on train_decoder.py, 
            # it saves a dict with 'metadata' key inside the main state dict,
            # or sometimes the metadata is at the top level depending on how it was saved.
            # Let's check the structure in train_decoder.py save_decoder function:
            # state = {
            #    'model_state_dict': model.state_dict(),
            #    ...
            #    'metadata': metadata or {}
            # }
            
            if 'metadata' in data:
                metadata = data['metadata']
                rmse = metadata.get('test_rmse', 'N/A')
                if isinstance(rmse, float):
                    print(f"{env} | {rmse:.4f}")
                else:
                    print(f"{env} | {rmse}")
            else:
                print(f"{env} | Metadata not found")
        except Exception as e:
            print(f"{env} | Error: {e}")
    else:
        print(f"{env} | File not found")