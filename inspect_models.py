import torch
import os

def inspect_model_file(file_path):
    print(f"\n=== Examining model: {os.path.basename(file_path)} ===")
    
    # Load the model file
    try:
        # Try to load using different methods as we don't know how it was saved
        try:
            # Method 1: If saved as state_dict
            model_data = torch.load(file_path, map_location=torch.device('cpu'))
            if isinstance(model_data, dict):
                print("Model loaded as a state dictionary")
                if 'state_dict' in model_data:
                    print("Contains 'state_dict' key")
                    model_data = model_data['state_dict']
                
                # Print the keys and shapes
                print("\nLayers/Parameters:")
                for key, value in model_data.items():
                    if isinstance(value, torch.Tensor):
                        print(f"  {key}: shape {list(value.shape)}, dtype {value.dtype}")
                    else:
                        print(f"  {key}: {type(value)}")
                
            # If it's a whole model object
            else:
                print(f"Model loaded as a complete object of type: {type(model_data)}")
                
                # Try to get model summary
                try:
                    print("\nModel structure:")
                    print(model_data)
                except:
                    print("Could not print model structure")
                    
        except Exception as e:
            print(f"Error examining model using standard loading: {str(e)}")
            
            # Try alternate loading method for models saved with torch.save(model)
            try:
                model_data = torch.jit.load(file_path)
                print(f"Model loaded as TorchScript module: {type(model_data)}")
                print("\nModel structure:")
                print(model_data)
            except Exception as e2:
                print(f"Error with alternate loading method: {str(e2)}")
                
    except Exception as e:
        print(f"Failed to load model file: {str(e)}")

# Directory containing the model files
model_dir = "./models"

# List of model files to inspect
model_files = [
    os.path.join(model_dir, "encoder.pt"),
    os.path.join(model_dir, "decoder.pt"),
    os.path.join(model_dir, "surrogate_model.pt")
]

# Check MPS availability
print(f"PyTorch version: {torch.__version__}")
print(f"MPS (Metal Performance Shaders) available: {torch.backends.mps.is_available()}")

# Process each model file
for model_file in model_files:
    if os.path.exists(model_file):
        inspect_model_file(model_file)
    else:
        print(f"\n=== Model file not found: {model_file} ===")

print("\nInspection complete!")
