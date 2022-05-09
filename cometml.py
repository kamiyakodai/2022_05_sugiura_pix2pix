from comet_ml import Experiment

def comet():
    experiment = Experiment(
            api_key = "nqdXsFWZqh06eZy2P1ZRE88RD",
            project_name = "pix2pix",
            workspace = "taikisugiura",
            auto_output_logging = "simple",
        )
    return experiment

def log_comet(experiment, fretchet_dist, epoch):
    experiment.log_metric("FIDscore",
                            fretchet_dist,
                            step = epoch,
                            )