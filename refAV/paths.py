from pathlib import Path

# change to path where the Argoverse2 Sensor dataset is downloaded
AV2_DATA_DIR = Path('data/av2')
TRACKER_DOWNLOAD_DIR = Path('tracker_downloads')
SM_DOWNLOAD_DIR = Path('av2_sm_downloads')

# # path to cached atomic function outputs, likely does not exist for you
CACHE_PATH = Path('/home/crdavids/Trinity-Sync/av2-api/output/misc')

#input directories, do not change
EXPERIMENTS = Path('run/experiments.yml')
REFAV_CONTEXT = Path('refAV/llm_prompting/refAV_context.txt')
AV2_CATEGORIES = Path('refAV/llm_prompting/av2_categories.txt')
PREDICTION_EXAMPLES = Path('refAV/llm_prompting/prediction_examples.txt')

work_dir='./'



exp_name='exp102'
#output directories, do not change
SM_DATA_DIR = Path(work_dir)/Path('output/sm_dataset')
SM_PRED_DIR = Path(work_dir)/Path('output/sm_predictions')
LLM_PRED_DIR = Path(work_dir)/Path('output')/Path(exp_name)/Path('llm_code_predictions')
TRACKER_PRED_DIR = Path(work_dir)/Path('output/tracker_predictions')
EVALUATION_ERRORS_DIR = Path(work_dir)/Path('output')/Path(exp_name)/Path('evaluation_errors.log')
