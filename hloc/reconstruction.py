import argparse
import multiprocessing
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional
from pdb import set_trace as bb
import pycolmap
import os


from . import logger
from .triangulation import (
    OutputCapture,
    estimation_and_geometric_verification,
    import_features,
    import_matches,
    parse_option_args,
)
from .utils.database import COLMAPDatabase


def create_empty_db(database_path: Path):
    if database_path.exists():
        logger.warning("The database already exists, deleting it.")
        database_path.unlink()
    logger.info("Creating an empty database...")
    db = COLMAPDatabase.connect(database_path)
    db.create_tables()
    db.commit()
    db.close()


def import_images(
    image_dir: Path,
    database_path: Path,
    camera_mode: pycolmap.CameraMode,
    image_list: Optional[List[str]] = None,
    options: Optional[Dict[str, Any]] = None,
):
    logger.info("Importing images into the database...")
    if options is None:
        options = {}
    images = list(image_dir.iterdir())
    if len(images) == 0:
        raise IOError(f"No images found in {image_dir}.")

    with pycolmap.ostream():
        pycolmap.import_images(
            database_path,
            image_dir,
            camera_mode,
            image_list=image_list or [],
            options=options,
        )


def get_image_ids(database_path: Path) -> Dict[str, int]:
    db = COLMAPDatabase.connect(database_path)
    images = {}
    for name, image_id in db.execute("SELECT name, image_id FROM images;"):
        images[name] = image_id
    db.close()
    return images


def run_reconstruction(
    sfm_dir: Path,
    database_path: Path,
    image_dir: Path,
    verbose: bool = False,
    options: Optional[Any] = None,
) -> pycolmap.Reconstruction:
    models_path = sfm_dir / "models"
    models_path.mkdir(exist_ok=True, parents=True)
    logger.info("Running 3D reconstruction...")
    
    if options is None:
        options = {}

    dict_options = {"num_threads": min(multiprocessing.cpu_count(), 16), **options}
    if pycolmap.__version__.startswith("3."):
        default_options = pycolmap.IncrementalPipelineOptions()
        # From dict input to pycolmap's incrementalmapperoptions
        for key, value in dict_options.items():
            if hasattr(default_options, key):  # if attributes exist
                setattr(default_options, key, value)

        for key, value in dict_options['colmap_mapper_cfgs'].items():
            if str(key).startswith("tri_"):
                key = str(key)[4:]
            if hasattr(default_options.triangulation, key):
                setattr(default_options.triangulation, key, value)

            if hasattr(default_options.mapper, key):
                setattr(default_options.mapper, key, value)


        full_options = default_options

        with OutputCapture(verbose):
            with pycolmap.ostream():
                reconstructions = pycolmap.incremental_mapping(
                    database_path, image_dir, models_path, options=full_options
                )

    elif pycolmap.__version__.startswith("0.6"):
        if options is None:
            options = {}
        with OutputCapture(verbose):
            with pycolmap.ostream():
                reconstructions = pycolmap.incremental_mapping(
                database_path, image_dir, models_path, options=options
            )

    else:
        ValueError("Not implemented on this PYCOLMAP version.")



    if len(reconstructions) == 0:
        logger.error("Could not reconstruct any model!")
        return None
    logger.info(f"Reconstructed {len(reconstructions)} model(s).")




    largest_num_images = 0
    for index, rec in reconstructions.items():
        num_images = rec.num_reg_images()
        logger.info(
            f"Model index #{index} has" f"{num_images} images."
        )
    
    for index in range(len(reconstructions)):
        for filename in ["images.bin", "cameras.bin", "points3D.bin"]:
            if (sfm_dir / filename).exists():
                (sfm_dir / filename).unlink()
            os.makedirs(sfm_dir / str(index), exist_ok=True)
            shutil.move(str(models_path / str(index) / filename), str(sfm_dir / str(index)))
    shutil.rmtree(str(models_path))

    return reconstructions


def main(
    sfm_dir: Path,
    image_dir: Path,
    pairs: Path,
    features: Path,
    matches: Path,
    camera_mode: pycolmap.CameraMode = pycolmap.CameraMode.AUTO,
    verbose: bool = False,
    skip_geometric_verification: bool = False,
    min_match_score: Optional[float] = None,
    image_list: Optional[List[str]] = None,
    image_options: Optional[Dict[str, Any]] = None,
    mapper_options: Optional[Any] = None,
) -> pycolmap.Reconstruction:
    assert features.exists(), features
    assert pairs.exists(), pairs
    assert matches.exists(), matches

    sfm_dir.mkdir(parents=True, exist_ok=True)
    database = sfm_dir / "database.db"

    create_empty_db(database)
    import_images(image_dir, database, camera_mode, image_list, image_options)
    image_ids = get_image_ids(database)
    import_features(image_ids, database, features)
    import_matches(
        image_ids,
        database,
        pairs,
        matches,
        min_match_score,
        skip_geometric_verification,
    )
    if not skip_geometric_verification:
        max_error = 4.0        
        try:
            if 'geometry_verify_thr' in mapper_options.keys():
                max_error = mapper_options['geometry_verify_thr']
        except:
            max_error=4.0

        estimation_and_geometric_verification(database, pairs, verbose, max_error=max_error)
    reconstruction_set = run_reconstruction(
        sfm_dir, database, image_dir, verbose, mapper_options
    )

    for i in range(len(reconstruction_set)):
        if reconstruction_set[i] is not None:
            logger.info(
                f"Reconstruction statistics:\n{reconstruction_set[i].summary()}"
                + f"\n\tnum_input_images = {len(image_ids)}"
            )
    return reconstruction_set


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sfm_dir", type=Path, required=True)
    parser.add_argument("--image_dir", type=Path, required=True)

    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--matches", type=Path, required=True)

    parser.add_argument(
        "--camera_mode",
        type=str,
        default="AUTO",
        choices=list(pycolmap.CameraMode.__members__.keys()),
    )
    parser.add_argument("--skip_geometric_verification", action="store_true")
    parser.add_argument("--min_match_score", type=float)
    parser.add_argument("--verbose", action="store_true")

    parser.add_argument(
        "--image_options",
        nargs="+",
        default=[],
        help="List of key=value from {}".format(pycolmap.ImageReaderOptions().todict()),
    )
    parser.add_argument(
        "--mapper_options",
        nargs="+",
        default=[],
        help="List of key=value from {}".format(
            pycolmap.IncrementalMapperOptions().todict()
        ),
    )
    args = parser.parse_args().__dict__

    image_options = parse_option_args(
        args.pop("image_options"), pycolmap.ImageReaderOptions()
    )
    mapper_options = parse_option_args(
        args.pop("mapper_options"), pycolmap.IncrementalMapperOptions()
    )

    main(**args, image_options=image_options, mapper_options=mapper_options)
