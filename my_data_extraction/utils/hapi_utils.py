from hapi import fetch
from hapi2_abscoef_discrete import absorptionCoefficient_Voigt as fast_voigt
from typing import List, Tuple, Optional
from itertools import chain, combinations
from multiprocessing import Lock
from logging import Logger
from os import getpid
from time import time

import numpy as np


from .file_writer import SpectraBatchWriter


def fetch_molecules(
    molecules: List[str], molecules_ids: List[int], wn_min: int, wn_max: int
) -> None:
    if len(molecules) != len(molecules_ids):
        raise ValueError("length of molecules must be equal to len of molecules ids")

    for molecule, id in zip(molecules, molecules_ids):
        fetch(molecule, id, 1, wn_min, wn_max)


def get_spectra(
    mol_ids: List[int],
    mol_ratios: List[float],
    source_tables: List[str],
    pressure: float = 1.0,  # in atmospheres
    temperature: float = 296.0,  # in Kelvins
    air_ratio: float = 0.0,  # ratio of air compared to mixture. air=1.0 - only air in mixture, air=0.0 - no air in mixture
    step: float = 1.0,  # in cm^-1
    wn_range: Optional[Tuple[float, float]] = None,
) -> List[float]:
    components = list(
        zip(
            mol_ids,
            [1] * len(mol_ids),
            mol_ratios,
        )
    )  # need to add isotope number to molecule ids + add ratio of that molecule
    _, coef = fast_voigt(
        Components=components,
        SourceTables=source_tables,
        OmegaStep=step,
        OmegaRange=wn_range,
        HITRAN_units=False,
        Environment={"p": pressure, "T": temperature},
        Diluent={"self": 1.0 - air_ratio, "air": air_ratio},
        # Verbose=verbose,
    )
    return coef


def synthesize_data(
    ids_combos: List[List[int]],
    mol_combos: List[List[str]],
    pressure_range: List[float],
    temperature_range: List[float],
    max_elements_in_mixture: int,
    air_ratio: float,
    file_name: str,
    logger: Logger,
    logger_lock: Lock,
    log_every: int = 50,
) -> None:

    assert len(ids_combos) == len(mol_combos)
    with open(file_name, "w") as f:
        batch_writer = SpectraBatchWriter(
            f, max_molecules_in_mixture=max_elements_in_mixture
        )
        cur_rows_count = 0
        total_rows_count = (
            len(mol_combos) * len(pressure_range) * len(temperature_range)
        )
        total_timer = time()
        cur_timer = time()

        with logger_lock:
            logger.info(
                f"PID: {getpid()}   |   task started    |   total_rows_to_synthesize: {total_rows_count} rows"
            )

        for ids, molecules in zip(ids_combos, mol_combos):
            for pressure in pressure_range:
                for temperature in temperature_range:
                    ratios = np.random.rand(len(ids))
                    ratios /= np.sum(ratios)  # normalize to add-up to 1
                    ratios = ratios.tolist()
                    spectra = get_spectra(
                        ids,
                        ratios,
                        molecules,
                        pressure=pressure,
                        temperature=temperature,
                        air_ratio=air_ratio,
                        wn_range=(10, 110),
                        step=0.1,
                    )
                    batch_writer.append(
                        list(spectra),
                        list(
                            molecules
                        ),  # as it is tuple, we need to cast it to list to be able to modify it
                        ratios,
                        temperature,
                        pressure,
                        air_ratio,
                    )
                    cur_rows_count += 1

                    if cur_rows_count % log_every == 0:
                        with logger_lock:
                            logger.info(
                                f"PID: {getpid()}   |   progress: {cur_rows_count / total_rows_count * 100 :.1f}%   |   cur_speed: {log_every / (time() - cur_timer) :.3f} rows/sec    |   total_speed: {cur_rows_count / (time() - total_timer) :.3f} rows/sec"
                            )
                        cur_timer = time()
        with logger_lock:
            logger.info(
                f"PID: {getpid()}   |   task completed!    |   total_speed: {cur_rows_count / (time() - total_timer) :.1f} rows/sec"
            )
        batch_writer.flush()


def get_molecules_combinations(
    molecules: List[str], ids: List[int], max_element_num: int = 2
) -> Tuple[List[List[str]], List[List[int]]]:
    mol_combos = list(
        chain.from_iterable(
            combinations(molecules, r) for r in range(1, max_element_num + 1)
        )
    )
    ids_combos = list(
        chain.from_iterable(combinations(ids, r) for r in range(1, max_element_num + 1))
    )
    return mol_combos, ids_combos
