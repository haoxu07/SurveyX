# offline run (entire pipeline)
python tasks/offline_run.py --title "Controllable Text Generation for Large Language Models: A Survey" --key_words "controlled text generation, text generation, large language model, LLM" --ref_path "outputs/ref1/jsons"


# Workflow (step by step, only for debugging)
task_id="xxx"
# python tasks/workflow/01_fetch_data.py  --task_id $task_id
# python tasks/workflow/02_clean_data.py  --task_id $task_id
python tasks/workflow/03_gen_outlines.py  --task_id $task_id
python tasks/workflow/04_gen_content.py  --task_id $task_id
python tasks/workflow/05_post_refine.py  --task_id $task_id
python tasks/workflow/06_gen_latex.py  --task_id $task_id
