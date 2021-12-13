#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

import functools
import os
import logging
import shutil
import traceback
import warnings
from pathlib import Path
from typing import Sequence, Tuple, Union

import numpy as np
import PIL
from monai.data import Dataset
from monai.data.image_reader import WSIReader
from tqdm import tqdm

from InnerEye.ML.Histopathology.preprocessing import tiling
from InnerEye.ML.Histopathology.datasets.panda_dataset import PandaDataset, LoadPandaROId


CSV_COLUMNS = ['slide_id', 'tile_id', 'image', 'mask', 'tile_x', 'tile_y', 'occupancy',
               'data_provider', 'slide_isup_grade', 'slide_gleason_score']
TMP_SUFFIX = "_tmp"

logging.basicConfig(format='%(asctime)s %(message)s', filemode='w')
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


def select_tile(mask_tile: np.ndarray, occupancy_threshold: float) \
        -> Union[Tuple[bool, float], Tuple[np.ndarray, np.ndarray]]:
    if occupancy_threshold < 0. or occupancy_threshold > 1.:
        raise ValueError("Tile occupancy threshold must be between 0 and 1")
    foreground_mask = mask_tile > 0
    occupancy = foreground_mask.mean(axis=(-2, -1))
    return (occupancy > occupancy_threshold).squeeze(), occupancy.squeeze()


def get_tile_descriptor(tile_location: Sequence[int]) -> str:
    return f"{tile_location[0]:05d}x_{tile_location[1]:05d}y"


def get_tile_id(slide_id: str, tile_location: Sequence[int]) -> str:
    return f"{slide_id}.{get_tile_descriptor(tile_location)}"


def save_image(array_chw: np.ndarray, path: Path) -> PIL.Image:
    path.parent.mkdir(parents=True, exist_ok=True)
    array_hwc = np.moveaxis(array_chw, 0, -1).astype(np.uint8).squeeze()
    pil_image = PIL.Image.fromarray(array_hwc)
    pil_image.convert('RGB').save(path)
    return pil_image


def generate_tiles(sample: dict, tile_size: int, occupancy_threshold: float) \
        -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    image_tiles, tile_locations = tiling.tile_array_2d(sample['image'], tile_size=tile_size,
                                                       constant_values=255)
    mask_tiles, _ = tiling.tile_array_2d(sample['mask'], tile_size=tile_size, constant_values=0)

    selected: np.ndarray
    occupancies: np.ndarray
    selected, occupancies = select_tile(mask_tiles, occupancy_threshold)
    n_discarded = (~selected).sum()
    logging.info(f"Percentage tiles discarded: {round(selected.sum() / n_discarded * 100, 2)}")

    image_tiles = image_tiles[selected]
    mask_tiles = mask_tiles[selected]
    tile_locations = tile_locations[selected]
    occupancies = occupancies[selected]

    abs_tile_locations = (sample['scale'] * tile_locations + sample['location']).astype(int)

    return image_tiles, mask_tiles, abs_tile_locations, occupancies, n_discarded


# TODO refactor this to separate metadata identification from saving. We might want the metadata
# even if the saving fails
def save_tile(sample: dict, image_tile: np.ndarray, mask_tile: np.ndarray,
              tile_location: Sequence[int], output_dir: Path) -> dict:
    slide_id = sample['image_id']
    descriptor = get_tile_descriptor(tile_location)
    image_tile_filename = f"train_images/{descriptor}.png"
    mask_tile_filename = f"train_label_masks/{descriptor}_mask.png"

    save_image(image_tile, output_dir / image_tile_filename)
    save_image(mask_tile, output_dir / mask_tile_filename)

    tile_metadata = {
        'slide_id': slide_id,
        'tile_id': get_tile_id(slide_id, tile_location),
        'image': image_tile_filename,
        'mask': mask_tile_filename,
        'tile_x': tile_location[0],
        'tile_y': tile_location[1],
        'data_provider': sample['data_provider'],
        'slide_isup_grade': sample['isup_grade'],
        'slide_gleason_score': sample['gleason_score'],
    }

    return tile_metadata


def process_slide(sample: dict, level: int, margin: int, tile_size: int, occupancy_threshold: int,
                  output_dir: Path, tile_progress: bool = False) -> None:
    slide_id = sample['image_id']
    slide_dir: Path = output_dir / (slide_id + "/")
    logging.info(f">>> Slide dir {slide_dir}")
    if slide_dir.exists():  # already processed slide - skip
        logging.info(f">>> Skipping {slide_dir} - already processed")
        return
    else:
        try:
            slide_dir.mkdir(parents=True)

            dataset_csv_path = slide_dir / "dataset.csv"
            dataset_csv_file = dataset_csv_path.open('w')
            dataset_csv_file.write(','.join(CSV_COLUMNS) + '\n')  # write CSV header

            tiles_failure = 0
            failed_tiles_csv_path = slide_dir / "failed_tiles.csv"
            failed_tiles_file = failed_tiles_csv_path.open('w')
            failed_tiles_file.write('tile_id' + '\n')

            logging.info(f"Loading slide {slide_id} ...")
            loader = LoadPandaROId(WSIReader(), level=level, margin=margin)
            sample = loader(sample)  # load 'image' and 'mask' from disk

            logging.info(f"Tiling slide {slide_id} ...")
            image_tiles, mask_tiles, tile_locations, occupancies, _ = \
                generate_tiles(sample, tile_size, occupancy_threshold)
            n_tiles = image_tiles.shape[0]

            for i in tqdm(range(n_tiles), f"Tiles ({slide_id[:6]}…)", unit="img", disable=not tile_progress):
                try:
                    tile_metadata = save_tile(sample, image_tiles[i], mask_tiles[i], tile_locations[i],
                                              slide_dir)
                    tile_metadata['occupancy'] = occupancies[i]
                    tile_metadata['image'] = os.path.join(slide_dir.name, tile_metadata['image'])
                    tile_metadata['mask'] = os.path.join(slide_dir.name, tile_metadata['mask'])
                    dataset_row = ','.join(str(tile_metadata[column]) for column in CSV_COLUMNS)
                    dataset_csv_file.write(dataset_row + '\n')
                except Exception as e:
                    tiles_failure += 1
                    descriptor = get_tile_descriptor(tile_locations[i]) + '\n'
                    failed_tiles_file.write(descriptor)
                    traceback.print_exc()
                    warnings.warn(f"An error occurred while saving tile "
                                  f"{get_tile_id(slide_id, tile_locations[i])}: {e}")

            dataset_csv_file.close()
            failed_tiles_file.close()
            if tiles_failure > 0:
                # TODO what we want to do with slides that have some failed tiles?
                logging.warning(f"{slide_id} is incomplete. {tiles_failure} tiles failed.")
        except Exception as e:
            traceback.print_exc()
            warnings.warn(f"An error occurred while processing slide {slide_id}: {e}")


def merge_dataset_csv_files(dataset_dir: Path) -> Path:
    full_csv = dataset_dir / "dataset.csv"
    # TODO change how we retrieve these filenames, probably because mounted, the operation is slow
    #  and it seems to find many more files
    # print("List of files")
    # print([str(file) + '\n' for file in dataset_dir.glob("*/dataset.csv")])
    with full_csv.open('w') as full_csv_file:
        # full_csv_file.write(','.join(CSV_COLUMNS) + '\n')  # write CSV header
        first_file = True
        for slide_csv in tqdm(dataset_dir.glob("*/dataset.csv"), desc="Merging dataset.csv", unit='file'):
            logging.info(f"Merging slide {slide_csv}")
            content = slide_csv.read_text()
            if not first_file:
                content = content[content.index('\n') + 1:]  # discard header row for all but the first file
            full_csv_file.write(content)
            first_file = False
    return full_csv


def main(panda_dir: Union[str, Path], root_output_dir: Union[str, Path], level: int, tile_size: int,
         margin: int, occupancy_threshold: float, parallel: bool = False, overwrite: bool = False) -> None:

    # Ignoring some types here because mypy is getting confused with the MONAI Dataset class
    # to select a subsample use keyword n_slides
    dataset = Dataset(PandaDataset(panda_dir))  # type: ignore

    output_dir = Path(root_output_dir) / f"panda_tiles_level{level}_{tile_size}"
    logging.info(f"Creating dataset of level-{level} {tile_size}x{tile_size} PANDA tiles at: {output_dir}")

    if overwrite and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=not overwrite)

    func = functools.partial(process_slide, level=level, margin=margin, tile_size=tile_size,
                             occupancy_threshold=occupancy_threshold, output_dir=output_dir,
                             tile_progress=not parallel)

    if parallel:
        import multiprocessing

        pool = multiprocessing.Pool()
        map_func = pool.imap_unordered  # type: ignore
    else:
        map_func = map  # type: ignore

    list(tqdm(map_func(func, dataset), desc="Slides", unit="img", total=len(dataset)))  # type: ignore

    if parallel:
        pool.close()

    logging.info("Merging slide files in a single file")
    merge_dataset_csv_files(output_dir)


if __name__ == '__main__':
    main(panda_dir="/tmp/datasets/PANDA",
         root_output_dir="/datadrive",
         level=1,
         tile_size=224,
         margin=64,
         occupancy_threshold=0.05,
         parallel=True,
         overwrite=False)
