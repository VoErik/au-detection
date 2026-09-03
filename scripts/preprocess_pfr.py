import argparse
from pathlib import Path
from src.config import AUDetectionConfig
from src.datasets.painfacereader import load_units
from src.preprocess import preprocess, build_detector


from src.utils import Logger

parser = argparse.ArgumentParser()
parser.add_argument(
    "--scale", 
    type=float, 
    default=0.5, 
    help=""
)
parser.add_argument(
    "--n_probe_frames",
    type=int,
    default=32,
    help="Face detection is run every n_probe frames. Decrease for moving subjects."
)
parser.add_argument(
    "--n_workers",
    type=int,
    default=8,
    help="How many workers to use. Increase for speedup."
)
parser.add_argument(
    "--overwrite", 
    action="store_true", 
    help="Set to overwrite existing crops."
)

if __name__ == "__main__":
    args = parser.parse_args()
    log = Logger(name=Path(__file__).stem).get_logger()
    
    cfg = AUDetectionConfig()
    log.info(f"Configuration loaded. Target image size: {cfg.img_size}, Crop margin: {cfg.crop_margin}")

    log.info("Loading Action Units...")
    units, au_names = load_units(cfg.painfacereader)
    log.info(f"Successfully loaded {len(units)} units and {len(au_names)} AU names.")

    log.info(
        f"Starting preprocessing pipeline with settings - "
        f"Scale: {args.scale}, Workers: {args.n_workers}, N-Probe: {args.n_probe_frames}, Overwrite: {args.overwrite}"
    )
    log.info("This may take a while, time to grab a coffee ☕.")
    
    preprocess(
        units=units, 
        crops_dir=cfg.painfacereader.crops_dir, 
        out_size=cfg.img_size, 
        margin=cfg.crop_margin, 
        detector=build_detector(cfg.device),
        detect_scale=args.scale,
        n_workers=args.n_workers,
        overwrite=args.overwrite,
        n_probe=args.n_probe_frames
    )
    log.info("Preprocessing finished.")