import torch
from src.models.small.small_model import SmallModel

def main():
    model = SmallModel()
    torch.save({"model": model.state_dict()}, "small_model_init.pt")
    print("Saved small_model_init.pt")

if __name__ == "__main__":
    main()
