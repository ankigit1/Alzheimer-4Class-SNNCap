from train import train_model
from validate import validate_model
from inference import single_image_inference

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Alzheimer 4-class classification with SNNCap")
    parser.add_argument('--mode', type=str, choices=['train', 'validate', 'inference'], default='train')
    parser.add_argument('--img_path', type=str, help="Path to single image for inference")
    args = parser.parse_args()

    if args.mode == 'train':
        train_model()
    elif args.mode == 'validate':
        validate_model()
    elif args.mode == 'inference':
        if args.img_path is None:
            print("Please provide --img_path for inference")
        else:
            single_image_inference(args.img_path)