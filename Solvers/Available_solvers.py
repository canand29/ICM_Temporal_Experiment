# Licensing Information:  You are free to use or extend this codebase for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) inform Guni Sharon at 
# guni@tamu.edu regarding your usage (relevant statistics is reported to NSF).
# The development of this assignment was supported by NSF (IIS-2238979).
# Contributors:
# The core code base was developed by Guni Sharon (guni@tamu.edu).

solvers = [
    "dqn",
    "dqn_icm_temporal",
    "dqn_icm"
]


def get_solver_class(name):
    if name == solvers[0]:
        from Solvers.DQN import DQN

        return DQN
    elif name == solvers[1]:
        from Solvers.DQN_ICM_Temporal import DQN_ICM_Temporal

        return DQN_ICM_Temporal

    elif name == solvers[2]:
        from Solvers.DQN_ICM import DQN_ICM

        return DQN_ICM

    else:
        assert False, "unknown solver name {}. solver must be from {}".format(
            name, str(solvers)
        )
