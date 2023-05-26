

def run_model(input_path, model_name):
    from livermask.utils.run import run_analysis
    run_analysis(cpu=True, extension='.nii', path=input_path, output='prediction', verbose=True, vessels=False, name=model_name, mp_enabled=False)
