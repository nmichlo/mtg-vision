import itertools
import multiprocessing
from pathlib import Path
from typing import Iterable, Iterator, Sequence, TypeVar
from tqdm import tqdm


from doorway.x import ProxyDownloader
from mtgdata.scryfall import ScryfallCardFace
from mtgvision.encoder_datasets import SyntheticBgFgMtgImages
from mtgvision.encoder_export import CoreMlEncoder
from mtgvision.qdrant import QdrantPoint, VectorStoreQdrant
from mtgvision.util.image import imread_float


T = TypeVar("T")


def batched(iterable: Iterable[T], n: int) -> Iterator[Sequence[T]]:
    """Yield successive n-sized batches from an iterable."""
    it = iter(iterable)
    while True:
        batch = list(itertools.islice(it, n))
        if not batch:
            return
        yield batch


class CardProcessor(multiprocessing.Process):
    def __init__(
        self,
        model_path: Path,
        job_queue: multiprocessing.Queue,
        result_queue: multiprocessing.Queue,
    ):
        """Initialize with picklable arguments."""
        super().__init__(daemon=True)
        self.model_path = model_path
        self.job_queue = job_queue
        self.result_queue = result_queue
        self._is_init = False
        self._x_size_hw = None
        self._encoder = None
        self._vstore = None
        self._proxy = None

    def _initialize(self):
        """Lazily initialize non-picklable resources in the worker process."""
        if not self._is_init:
            self._is_init = True
            self._encoder = CoreMlEncoder(
                self.model_path.with_suffix(".encoder.mlpackage")
            )
            h, w, c = self._encoder.input_hwc
            self._x_size_hw = (h, w)
            self._vstore = VectorStoreQdrant()
            self._proxy = ProxyDownloader()
            print(f"Worker {self.pid} initialized")

    def run(self):
        """Main loop for the worker process."""
        self._initialize()
        while True:
            batch = self.job_queue.get()
            if batch is None:
                break
            result = self.process_batch(batch)
            self.result_queue.put(result)

    def process_batch(self, batch: Sequence[ScryfallCardFace]) -> int:
        """Process a batch of cards and upload missing ones to Qdrant."""
        # Extract card IDs and check existing entries in Qdrant
        existing_points = self._vstore.retrieve(str(card.id) for card in batch)
        existing_ids = {point.id for point in existing_points}
        missing_cards = [card for card in batch if str(card.id) not in existing_ids]
        # Process each missing card and prepare points for upload
        points_to_upload = []
        for card in missing_cards:
            points_to_upload.append(self._get_card_point(card))
        if points_to_upload:
            self._vstore.save_points(points_to_upload)
        return len(points_to_upload)

    def _get_card_point(self, card: ScryfallCardFace) -> QdrantPoint:
        """Generate a Point object for a single card."""
        path = card.download(proxy=self._proxy)
        x = imread_float(path)
        x = SyntheticBgFgMtgImages.make_cropped(x, size_hw=self._x_size_hw)
        z = self._encoder.predict(x).tolist()
        return QdrantPoint(id=str(card.id), vector=z, payload=None)


def _cli():
    from mtgvision.encoder_export import MODEL_PATH

    # Configuration
    dataset = SyntheticBgFgMtgImages(img_type="small", predownload=False)
    num_workers = 4  # Adjust based on CPU cores
    batch_size = 32  # Adjust based on memory/performance
    model_path = MODEL_PATH

    # Create job and result queues
    job_queue = multiprocessing.Queue()
    result_queue = multiprocessing.Queue()

    # Start worker processes
    processes = []
    for _ in range(num_workers):
        worker = CardProcessor(model_path, job_queue, result_queue)
        worker.start()
        processes.append(worker)

    # Submit all batches to the job queue
    batch_itr = batched(dataset.card_iter(), batch_size)
    total_batches = 0
    for batch in tqdm(batch_itr, desc="Submitting batches"):
        job_queue.put(batch)
        total_batches += 1

    # Signal workers to stop by putting None for each
    for _ in range(num_workers):
        job_queue.put(None)

    # Collect results and track progress
    total_points_uploaded = 0
    with tqdm(total=len(dataset), desc="Processing batches") as pbar:
        for _ in range(total_batches):
            result = result_queue.get()
            total_points_uploaded += result
            pbar.update(batch_size)

    # Ensure all workers have finished
    for proc in processes:
        proc.join()

    print(f"Total points uploaded: {total_points_uploaded}")


if __name__ == "__main__":
    _cli()
