from hapi import db_begin
import numpy as np
import multiprocessing
import time
import logging

from utils.hapi_utils import (
    get_molecules_combinations,
    fetch_molecules,
    synthesize_data,
)

logger = logging.getLogger(__name__)
logging.basicConfig(
    filename="processes.log",
    encoding="utf-8",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s]:  %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
)


WN_RANGE = (10, 110)
MOLECULES = [
    "H2O",
    "O2",
    "CH4",
    "NH3",
    "H2S",
    "OH",
    "N2O",
    "CO",
    "N2",
    "CH3OH",
    "ClO",
    "H2O2",
    "SO",
    "PH3",
    "HCN",
    "HOCl",
    "H2CO",
    "HI",
    "HBr",
    "CS",
    "HC3N",
    "NO+",
    "CH3Cl",
    "HO2",
    "CH3CN",
]
MOLECULE_IDS = [
    1,
    7,
    6,
    11,
    31,
    13,
    4,
    5,
    22,
    39,
    18,
    25,
    50,
    28,
    23,
    21,
    20,
    17,
    16,
    46,
    44,
    36,
    24,
    33,
    41,
]
MAX_ELEMENTS_IN_MIXTURE = 3

PRESSURE_RANGE = np.linspace(0.1, 1.0, 10)
TEMP_RANGE = np.linspace(253, 323, 8)
AIR_RATIO_RANGE = [0.6]  # [0.3, 0.6, 0.9]  # np.linspace(0, 0.9, 5)

db_begin("data")


# fetch_molecules(MOLECULES, MOLECULE_IDS, *WN_RANGE)

mol_combos, ids_combos = get_molecules_combinations(
    MOLECULES, MOLECULE_IDS, MAX_ELEMENTS_IN_MIXTURE
)


total_data_parsed = (
    len(ids_combos) * len(AIR_RATIO_RANGE) * len(PRESSURE_RANGE) * len(TEMP_RANGE)
)

cnt = 0
elem = "H2O"
for comb in mol_combos:
    if elem in comb:
        cnt += 1
print(f"{elem} containing combos: {cnt / len(mol_combos) * 100 :.1f}%")
print(f"total data expected: {total_data_parsed}")

if __name__ == "__main__":
    logging.info(f"\n\n\n=====================    START    =====================")
    logging.info(f"len of combinations of molecules: {len(mol_combos)}")
    logging.info(f"len of air ratios: {len(AIR_RATIO_RANGE)}")
    logging.info(f"len of pressures: {len(PRESSURE_RANGE)}")
    logging.info(f"len of temperatures: {len(TEMP_RANGE)}\n")

    logging.info(f"total data expected: {total_data_parsed}")

    start = time.time()
    n_cpus = multiprocessing.cpu_count() - 2
    logging.info(f"using {n_cpus} cpus")

    with multiprocessing.Manager() as manager:
        logger_lock = manager.Lock()
        with multiprocessing.Pool(n_cpus) as pool:

            def kill_pool(err_msg):
                print(err_msg)
                pool.terminate()
                pool.close()

            results = [
                pool.apply_async(
                    synthesize_data,
                    (
                        ids_combos,
                        mol_combos,
                        [pressure],
                        [temperature],
                        MAX_ELEMENTS_IN_MIXTURE,
                        air_ratio,
                        f"./results/new-more-data/pressure-{pressure:.1f}-air_ratio-{air_ratio:.1f}-temp-{int(temperature)}.csv",
                        logger,
                        logger_lock,
                        5,
                    ),
                    error_callback=kill_pool,
                )
                for pressure in PRESSURE_RANGE
                for temperature in TEMP_RANGE
                for air_ratio in AIR_RATIO_RANGE
            ]

            logging.info(f"total tasks: {len(results)}")

            values = [res.get() for res in results]

    total_time = time.time() - start
    logging.info(
        f"total time required for parsing {total_data_parsed} rows: {total_time}. Speed: {total_data_parsed/total_time:.2f} rows/s"
    )
