import os
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
import torchvision.utils as vutils

from tools.arguments import parse_args
from data.datasets import PineappleDataset

# Import your models
from models.vae import VAE
from models.vqvae import VQVAE
from models.dual_vae import DUALVAE

def prepare_test_data(args):
    """Initializes the test dataset and dataloader."""
    testset = PineappleDataset(
        path=args.dataset_path,
        split='test', 
        test_txt=args.path_test_ids, 
        augment=False, 
        seed=args.seed
    )
    testloader = DataLoader(
        testset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=2
    )
    return testloader

def load_model(args, device):
    """Instantiates the correct model and loads the checkpoint."""
    if args.model == "vae":
        model = VAE()
    elif args.model == "vqvae":
        model = VQVAE(
            commitment_cost=args.commitment_cost,
            embedding_dim=args.codebook_dim,
            num_embeddings=args.num_embeddings
        )
    elif args.model == "dualvae":
        model = DUALVAE(
            commitment_cost=args.commitment_cost,
            embedding_dim=args.codebook_dim,
            num_embeddings=args.num_embeddings
        )
    else:
        raise ValueError(f"Unknown model type: {args.model}")

    model = model.to(device)
    
    # Load the trained weights
    if not os.path.exists(args.checkpoint_path_test):
        raise FileNotFoundError(f"Checkpoint not found at {args.checkpoint_path_test}")
    
    model.load_state_dict(torch.load(args.checkpoint_path_test, map_location=device))
    model.eval()
    
    return model

def generate_inferences(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    
    # 1. Setup Data & Model
    print(f"Loading test data from {args.path_test_ids}...")
    testloader = prepare_test_data(args)
    
    # Extract the underlying dataset object so we can access its 'images' list
    testset = testloader.dataset
    
    print(f"Loading {args.model} model from {args.checkpoint_path_test}...")
    model = load_model(args, device)

    # 2. Create Output Directory
    os.makedirs(args.output_dir_test, exist_ok=True)
    print(f"Inferences will be saved to: {args.output_dir_test}")

    # 3. Inference Loop
    with torch.no_grad():
        for batch in tqdm(testloader, desc="Generating Inferences"):
            images = batch["image"].to(device)
            indices = batch["idx"] # Grab the original indices from the batch
            
            # Forward pass depends on what the model returns
            if args.model == "vae":
                recon, _, _ = model(images)
            elif args.model == "vqvae":
                recon, _, _, _ = model(images)
            elif args.model == "dualvae":
                recon, _, _ = model(images, ablation_mode=1) # Use the same ablation_mode you trained with if applicable

            # 4. Save the reconstructed images
            # Clamp to [0, 1] just in case to prevent visual artifacts
            recon = recon.clamp(0, 1)
            
            for i in range(images.size(0)):
                # 1. Get the dataset index for this specific image in the batch
                dataset_idx = indices[i].item()
                
                # 2. Look up the original file path in the dataset's 'images' list
                original_path = testset.images[dataset_idx]
                
                # 3. Extract just the filename with its original extension
                filename = os.path.basename(original_path)
                
                # 4. Create the final save path
                save_path = os.path.join(args.output_dir_test, filename)
                
                # Save single image
                vutils.save_image(recon[i], save_path)

    print("Inference complete!")

if __name__ == "__main__":
    args = parse_args()
    
    generate_inferences(args)