from experiments import Experiment

experiment = Experiment("experiment.yaml")
experiment.run_experiments()

experiment.generate_plots()
experiment.generate_reports()
experiment.save()