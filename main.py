from tools.arguments import parse_args

if __name__ == "__main__":
    args = parse_args()
    if args.model == "vae_perceptual":
        from experiments.train_vae_perceptual import train_vae as train
    elif args.model == "vqvae":
        from experiments.train_vqvae import train_vqvae as train
    elif args.model == "dualvae":
        from experiments.train_dualvae import train_dualvae as train
    elif args.model == "swd_dualvae":
        from experiments.train_swd_dualvae import train_swd_dualvae as train
    elif args.model == "latent_flow":
        # Class-conditional flow matching in a FROZEN autoencoder's latent space
        # (configs/flow_dualvae.yaml, configs/flow_vae.yaml).
        from experiments.train_latent_flow import train_latent_flow as train
    else:
        from experiments.train_vae import train_vae as train

    train(args)