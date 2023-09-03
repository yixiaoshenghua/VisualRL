# Reproduce CURL
python run.py --agent CURL --env walker-walk --agent_type model_free --exp_name reproduce --pre_transform_image_size 100 --image_size 84 --action_repeat 2 --update_steps 1 --buffer_size 100000 --batch_size 512 --save_tb --gpu 0 --seed 1
# Reproduce DreamerV1
python run.py --agent Dreamerv1 --env walker-walk --agent_type model_based --exp_name reproduce  --pre_transform_image_size 64 --image_size 64 --action_repeat 2 --update_steps 100 --save_tb --gpu 0
# Reproduce SAC-AE
python run.py --agent SAC_AE --env walker-walk --agent_type model_free --exp_name reproduce --pre_transform_image_size 84 --image_size 84 --action_repeat 2 --update_steps 1 --save_tb --gpu 0 --seed 1
# Reproduce DrQ
python run.py --agent DrQ --env walker-walk --agent_type model_free --exp_name reproduce --pre_transform_image_size 84 --image_size 84 --action_repeat 2 --update_steps 1 --buffer_size 100000 --batch_size 512 --save_tb --gpu 0 --seed 1
