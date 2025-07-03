import argparse
import contextlib
import io
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional
import os

import numpy as np
import pycolmap
from tqdm import tqdm

from . import logger
from .utils.database import COLMAPDatabase
from .utils.geometry import compute_epipolar_errors
from .utils.io import get_keypoints, get_matches
from .utils.parsers import parse_retrieval


class OutputCapture:
    def __init__(self, verbose: bool):
        self.verbose = verbose

    def __enter__(self):
        if not self.verbose:
            self.capture = contextlib.redirect_stdout(io.StringIO())
            self.out = self.capture.__enter__()

    def __exit__(self, exc_type, *args):
        if not self.verbose:
            self.capture.__exit__(exc_type, *args)
            if exc_type is not None:
                logger.error("Failed with output:\n%s", self.out.getvalue())
        sys.stdout.flush()


def create_db_from_model(
    reconstruction: pycolmap.Reconstruction, database_path: Path
) -> Dict[str, int]:
    if database_path.exists():
        logger.warning("The database already exists, deleting it.")
        database_path.unlink()

    db = COLMAPDatabase.connect(database_path)
    db.create_tables()

    for i, camera in reconstruction.cameras.items():
        db.add_camera(
            camera.model.value,
            camera.width,
            camera.height,
            camera.params,
            camera_id=i,
            prior_focal_length=True,
        )

    for i, image in reconstruction.images.items():
        db.add_image(image.name, image.camera_id, image_id=i)

    db.commit()
    db.close()
    return {image.name: i for i, image in reconstruction.images.items()}


def import_features(
    image_ids: Dict[str, int], database_path: Path, features_path: Path
):
    logger.info("Importing features into the database...")
    db = COLMAPDatabase.connect(database_path)

    for image_name, image_id in tqdm(image_ids.items()):
        keypoints = get_keypoints(features_path, image_name)
        keypoints += 0.5  # COLMAP origin
        db.add_keypoints(image_id, keypoints)
    db.commit()
    db.close()


def import_matches(
    image_ids: Dict[str, int],
    database_path: Path,
    pairs_path: Path,
    matches_path: Path,
    min_match_score: Optional[float] = None,
    skip_geometric_verification: bool = False,
):
    logger.info("Importing matches into the database...")

    with open(str(pairs_path), "r") as f:
        pairs = [p.split() for p in f.readlines()]

    db = COLMAPDatabase.connect(database_path)

    matched = set()
    for name0, name1 in tqdm(pairs):
        id0, id1 = image_ids[name0], image_ids[name1]
        if len({(id0, id1), (id1, id0)} & matched) > 0:
            continue
        matches, scores = get_matches(matches_path, name0, name1)
        if min_match_score:
            matches = matches[scores > min_match_score]
        db.add_matches(id0, id1, matches)
        matched |= {(id0, id1), (id1, id0)}
        if skip_geometric_verification:
            db.add_two_view_geometry(id0, id1, matches)

    db.commit()
    db.close()


def estimation_and_geometric_verification(
    database_path: Path, pairs_path: Path, verbose: bool = False, max_error=4.0
):
    logger.info("Performing geometric verification of the matches...")
    with OutputCapture(True):
        with pycolmap.ostream():   
            pycolmap.verify_matches(  database_path,pairs_path,     options=dict(ransac=dict(max_error=max_error, max_num_trials=20000, min_inlier_ratio=0.1)), )

def geometric_verification(
    image_ids: Dict[str, int],
    reference: pycolmap.Reconstruction,
    database_path: Path,
    features_path: Path,
    pairs_path: Path,
    matches_path: Path,
    max_error: float = 4.0,
):
    logger.info("Performing geometric verification of the matches...")

    pairs = parse_retrieval(pairs_path)
    db = COLMAPDatabase.connect(database_path)

    inlier_ratios = []
    matched = set()
    for name0 in tqdm(pairs):
        id0 = image_ids[name0]
        image0 = reference.images[id0]
        cam0 = reference.cameras[image0.camera_id]
        kps0, noise0 = get_keypoints(features_path, name0, return_uncertainty=True)
        noise0 = 1.0 if noise0 is None else noise0
        if len(kps0) > 0:
            kps0 = np.stack(cam0.cam_from_img(kps0))
        else:
            kps0 = np.zeros((0, 2))

        for name1 in pairs[name0]:
            id1 = image_ids[name1]
            image1 = reference.images[id1]
            cam1 = reference.cameras[image1.camera_id]
            kps1, noise1 = get_keypoints(features_path, name1, return_uncertainty=True)
            noise1 = 1.0 if noise1 is None else noise1
            if len(kps1) > 0:
                kps1 = np.stack(cam1.cam_from_img(kps1))
            else:
                kps1 = np.zeros((0, 2))

            matches = get_matches(matches_path, name0, name1)[0]

            if len({(id0, id1), (id1, id0)} & matched) > 0:
                continue
            matched |= {(id0, id1), (id1, id0)}

            if matches.shape[0] == 0:
                db.add_two_view_geometry(id0, id1, matches)
                continue

            cam1_from_cam0 = image1.cam_from_world * image0.cam_from_world.inverse()
            errors0, errors1 = compute_epipolar_errors(
                cam1_from_cam0, kps0[matches[:, 0]], kps1[matches[:, 1]]
            )
            valid_matches = np.logical_and(
                errors0 <= cam0.cam_from_img_threshold(noise0 * max_error),
                errors1 <= cam1.cam_from_img_threshold(noise1 * max_error),
            )
            # TODO: We could also add E to the database, but we need
            # to reverse the transformations if id0 > id1 in utils/database.py.
            db.add_two_view_geometry(id0, id1, matches[valid_matches, :])
            inlier_ratios.append(np.mean(valid_matches))
    logger.info(
        "mean/med/min/max valid matches %.2f/%.2f/%.2f/%.2f%%.",
        np.mean(inlier_ratios) * 100,
        np.median(inlier_ratios) * 100,
        np.min(inlier_ratios) * 100,
        np.max(inlier_ratios) * 100,
    )

    db.commit()
    db.close()


def run_triangulation(
    model_path: Path,
    database_path: Path,
    image_dir: Path,
    reference_model: pycolmap.Reconstruction,
    verbose: bool = False,
    options: Optional[Any] = None,
) -> pycolmap.Reconstruction:

    model_path.mkdir(parents=True, exist_ok=True)
    logger.info("Running 3D triangulation...")


    if pycolmap.__version__.startswith("3."):
        default_options = pycolmap.IncrementalPipelineOptions()
        """
        {'ImageReader_camera_mode': 'auto', 'ImageReader_single_camera': False, 'min_model_size': 6, 'no_refine_intrinsics': True, 'n_threads': 16, 
        'use_pba': False, 'geometry_verify_thr': 12.0, 
        'reregistration': {'abs_pose_max_error': 12, 'abs_pose_min_num_inliers': 30, 'abs_pose_min_inlier_ratio': 0.25, 'filter_max_reproj_error': 5}, 
        'colmap_mapper_cfgs': {'init_max_error': 12, 'abs_pose_max_error': 12, 'filter_max_reproj_error': 12, 'tri_merge_max_reproj_error': 12, 
        'tri_complete_max_reproj_error': 12, 'tri_ignore_two_view_tracks': 1}}
        """        
        # From dict input to pycolmap's incrementalmapperoptions
        for key, value in options.items():
            if hasattr(default_options, key):  # 속성이 존재하는 경우만
                setattr(default_options, key, value)

        for key, value in options['colmap_mapper_cfgs'].items():
            if str(key).startswith("tri_"):
                key = str(key)[4:]
            if hasattr(default_options.triangulation, key):
                setattr(default_options.triangulation, key, value)

            if hasattr(default_options.mapper, key):
                setattr(default_options.mapper, key, value)


        options = default_options

        with OutputCapture(verbose):
            with pycolmap.ostream():
                reconstruction = pycolmap.triangulate_points(
                    reference_model, database_path, image_dir, model_path, options=options
                )


    elif pycolmap.__version__.startswith("0.6"):
        if options is None:
            options = {}
        with OutputCapture(verbose):
            with pycolmap.ostream():
                reconstruction = pycolmap.triangulate_points(
                    reference_model, database_path, image_dir, model_path, options=options
                )
    else:
        ValueError("Not implemented on this PYCOLMAP version.")


    return reconstruction


def main(
    sfm_dir: Path,
    reference_model: Path,
    image_dir: Path,
    pairs: Path,
    features: Path,
    matches: Path,
    skip_geometric_verification: bool = False,
    estimate_two_view_geometries: bool = False,
    min_match_score: Optional[float] = None,
    verbose: bool = False,
    mapper_options: Optional[Any] = None,
) -> pycolmap.Reconstruction:
    assert reference_model.exists(), reference_model
    assert features.exists(), features
    assert pairs.exists(), pairs
    assert matches.exists(), matches

    sfm_dir.mkdir(parents=True, exist_ok=True)
    database = sfm_dir / "database.db"

    # making '0' in sfm_dir for consistency
    sfm_dir = sfm_dir / "0"
    sfm_dir.mkdir(parents=True, exist_ok=True)

    reference = pycolmap.Reconstruction(reference_model)

    image_ids = create_db_from_model(reference, database)
    import_features(image_ids, database, features)
    import_matches(
        image_ids,
        database,
        pairs,
        matches,
        min_match_score,
        skip_geometric_verification,
    )

    from pdb import set_trace as bb
    if not skip_geometric_verification:
        max_error = 4.0
        
        try:
            if 'geometry_verify_thr' in mapper_options.keys():
                max_error = mapper_options['geometry_verify_thr']
        except:
            max_error=4.0
        """
        if estimate_two_view_geometries:
            estimation_and_geometric_verification(database, pairs, verbose)
        else:
        """
        geometric_verification(
            image_ids, reference, database, features, pairs, matches, max_error=max_error
        )

    reconstruction = run_triangulation(
        sfm_dir, database, image_dir, reference, verbose, mapper_options
    )
    logger.info(
        "Finished the triangulation with statistics:\n%s", reconstruction.summary()
    )
    return reconstruction


def parse_option_args(args: List[str], default_options) -> Dict[str, Any]:
    options = {}
    for arg in args:
        idx = arg.find("=")
        if idx == -1:
            raise ValueError("Options format: key1=value1 key2=value2 etc.")
        key, value = arg[:idx], arg[idx + 1 :]
        if not hasattr(default_options, key):
            raise ValueError(
                f'Unknown option "{key}", allowed options and default values'
                f" for {default_options.summary()}"
            )
        value = eval(value)
        target_type = type(getattr(default_options, key))
        if not isinstance(value, target_type):
            raise ValueError(
                f'Incorrect type for option "{key}":' f" {type(value)} vs {target_type}"
            )
        options[key] = value
    return options


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sfm_dir", type=Path, required=True)
    parser.add_argument("--reference_sfm_model", type=Path, required=True)
    parser.add_argument("--image_dir", type=Path, required=True)

    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--matches", type=Path, required=True)

    parser.add_argument("--skip_geometric_verification", action="store_true")
    parser.add_argument("--min_match_score", type=float)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args().__dict__

    mapper_options = parse_option_args(
        args.pop("mapper_options"), pycolmap.IncrementalMapperOptions()
    )
    main(**args, mapper_options=mapper_options)
