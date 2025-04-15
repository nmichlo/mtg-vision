"""
Populate Qdrant with card information from Scryfall.

TODO: should merge this with qdrant_populate.py...
TODO: card information might not be correctly produced by ScryfallDataset, I don't think
      we are handling card faces properly...
      e.g. foil version vs normal version of card might have different ID
      e.g. cards with multiple faces might be overridden
"""

import itertools

from tqdm import tqdm

from mtgdata import ScryfallImageType
from mtgvision.encoder_datasets import SyntheticBgFgMtgImages
from mtgvision.qdrant import VectorStoreQdrant


def _iter_batches(iter, batch_size: int = 64):
    """Yield successive n-sized batches from an iterable."""
    it = iter
    while True:
        batch = list(itertools.islice(it, batch_size))
        if not batch:
            return
        yield batch


def _cli():
    # qdrant
    vstore = VectorStoreQdrant()

    # Configuration
    data = SyntheticBgFgMtgImages(
        img_type=ScryfallImageType.small,
        predownload=False,
    )
    ds = data.make_scryfall_data(force_update=False, predownload=False).ds

    # update cards
    # TODO: should batch...
    for raw in tqdm(ds.yield_raw_info(), total=len(ds)):
        id_ = raw["id"]
        try:
            vstore.update_payload(id_, payload=raw)
        except Exception as e:
            print(f"Failed to update payload for ID {id_}, error: {e}, payload: {raw}")
            continue


if __name__ == "__main__":
    _cli()
