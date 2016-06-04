import numpy as np


def test23(algo, num_sims, checkPoints, trueBestArms):
    num_points = len(checkPoints)
    horizons = np.zeros(num_sims)
    checkpoints = np.zeros((num_sims, num_points), dtype=int)
    checkerrors = np.zeros((num_sims, num_points), dtype=bool)

    for sim in range(num_sims):
        algo.initialize()
        algo.set_checkpoints(checkPoints, trueBestArms)
        algo.run_with_check()
        horizons[sim] = algo.N
        checkpoints[sim] = algo.checkpoints
        checkerrors[sim] = algo.checkerrors + [algo.checkerrors[-1]]*(num_points-len(algo.checkerrors))

    return horizons, checkpoints, checkerrors
