from comet_ml import Experiment

def comet():
    experiment = Experiment(
            api_key = "nqdXsFWZqh06eZy2P1ZRE88RD",
            project_name = "pix2pix",
            workspace = "taikisugiura",
            auto_output_logging = "simple",
        )
    return experiment

def valGLossComet(experiment, meanLossG, epoch):
    experiment.log_metric("valLossG",
                            meanLossG,
                            step = epoch,
                            )


def valDLossComet(experiment, meanLossD, epoch):
    experiment.log_metric("ValLossD",
                            meanLossD,
                            step = epoch,
                            )

def FIDComet(experiment, fretchet_dist, epoch):
    experiment.log_metric("FIDscore",
                            fretchet_dist,
                            step = epoch,
                            )


def gLossComet(experiment, lossG, epoch):
    experiment.log_metric("lossG",
                            lossG,
                            step = epoch,
                            )


def dLossComet(experiment, lossD, epoch):
    experiment.log_metric("lossD",
                            lossD,
                            step = epoch,
                            )
