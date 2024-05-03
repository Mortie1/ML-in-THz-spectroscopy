from typing import Any, Iterable
from csv import writer

from io import TextIOWrapper

from multiprocessing import Lock


class BatchWriter:
    def __init__(self, file: TextIOWrapper, batch_size: int = 128) -> None:
        self.file = file
        self.batch_size = batch_size
        self._batch = []
        self.writer = writer(file)

    def flush(self) -> None:
        self.writer.writerows(self._batch)
        self._batch = []

    def append(self, item: Any) -> None:
        self._batch.append(item)
        if len(self._batch) >= self.batch_size:
            self.flush()


class SpectraBatchWriter(BatchWriter):

    def __init__(
        self,
        file: TextIOWrapper,
        max_molecules_in_mixture: int,
        batch_size: int = 32,
    ) -> None:
        super().__init__(file, batch_size)
        self.max_molecules_in_mixture = max_molecules_in_mixture

    def append(
        self,
        spectra: Iterable,
        molecules: Iterable,
        ratios: Iterable,
        temperature: float,
        pressure: float,
        air_ratio: float,
    ) -> None:
        molecules = molecules + ["null"] * (
            self.max_molecules_in_mixture - len(molecules)
        )
        ratios = ratios + ["null"] * (self.max_molecules_in_mixture - len(ratios))
        item = spectra + molecules + ratios + [temperature, pressure, air_ratio]
        return super().append(item)
