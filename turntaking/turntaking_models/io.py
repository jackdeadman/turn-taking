from turntaking.utils import load_yaml, merge_dicts
from turntaking.extra_typing import PathString
from turntaking.turntaking_models.markov import TurntakingFactory, prune_model


def load_model_from_dict(dictionary):
    defaults = {
        "num_speakers": 4,
        # semi_markov | independent_semi_markov | competing | markov | independent_markov
        "model_type": "semi_markov",
        "pruned": True,
        "smoothing": None,
        'state_type': 'wald'
    }

    values = merge_dicts(defaults, dictionary)

    if values['model_type'] == 'semi_markov':
        model = TurntakingFactory.full_semi_markov(
            num_spks=values['num_speakers'],
            state_type=values['state_type'],
            smoothing=values['smoothing']
        )
    elif values['model_type'] == 'independent_semi_markov':
        model = TurntakingFactory.independent_semi_markov(
            num_spks=values['num_speakers'],
            state_type=values['state_type'],
            smoothing=values['smoothing'])
    elif values['model_type'] == 'competing':
        model = TurntakingFactory.competing(
            num_spks=values['num_speakers'],
            smoothing=values['smoothing'])
    elif values['model_type'] == 'competing_semi_markov':
        model = TurntakingFactory.competing_semi_markov(
            num_spks=values['num_speakers'],
            state_type=values['state_type'],
            smoothing=values['smoothing'])
    elif values['model_type'] == 'markov':
        model = TurntakingFactory.full(
            num_spks=values['num_speakers'],
            smoothing=values['smoothing']
        )
    elif values['model_type'] == 'independent_markov':
        model = TurntakingFactory.independent(
            num_spks=values['num_speakers'],
            smoothing=values['smoothing']
        )
    else:
        raise ValueError('Not a valid model type: ' + values['model_type'])

    if values['pruned']:
        prune_model(model)

    return model


def load_model_from_yaml(path_to_file: PathString):
    path = str(path_to_file)
    return load_model_from_dict(load_yaml(path))


def main():
    import sys
    model = load_model_from_yaml(sys.argv[1])
    print(model, model.smoothing)

if __name__ == '__main__':
    main()
