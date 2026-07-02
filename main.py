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
    else:
        from experiments.train_vae import train_vae as train

    train(args)