from comet_ml import Experiment

def comet():
    experiment = Experiment(
            api_key = "nqdXsFWZqh06eZy2P1ZRE88RD",
            project_name = "pix2pix",
            workspace = "taikisugiura",
            auto_output_logging = "simple",
        )
    return experiment

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

def testGLossComet(experiment, meanLossG, step):
    experiment.log_metric("lossG",
                            meanLossG,
                            step = step,
                            )


def testDLossComet(experiment, meanLossD, step):
    experiment.log_metric("lossD",
                            meanLossD,
                            step = step,
                            )