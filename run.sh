# [Note]: Change the path to your own data
scene_path="../data/man"
tag="man_leather_jacket"
exp_root_dir="./outputs/$tag"
test_save_path="./outputs/$tag/rgb_cache"
shape_init="$scene_path/full_body.ply"
gpu_id=0

# [Note]: Change the prompt for your editing task
seg_prompt="jacket"
edit_prompt="classic brown biker leather jacket"  # edit_prompt="olive-green hooded waterproof jacket"  # edit_prompt="denim jacket with a faded wash"
local_prompt="A DSLR photo of a $edit_prompt"
global_prompt="A DSLR photo of a man wearing a $edit_prompt, full body"
sample_type="upper"  # editing garment type: "upper", "lower" or "full"



# 1. SDF instantiation
# TODO

# 2. TetGS initialization & segmenting interested area
cd Edit_core
# 2.1 Geometry initialization
exp_name="stage0-geometry-init"
CASE_init=${exp_root_dir}/${exp_name}
python train_spatial.py \
    --config configs/geometry-init.yaml \
    --train \
    --gpu ${gpu_id} \
    tag=${tag} \
    name=${exp_name} \
    exp_root_dir=${exp_root_dir} \
    system.geometry.shape_init=${shape_init}
# 2.2 Export Geometry
python train_spatial.py \
    --config ${CASE_init}/configs/parsed.yaml \
    --export \
    --gpu ${gpu_id} \
    system.shape_init=false \
    system.prev_checkpoint=${CASE_init}/ckpts/initial_checkpoint.ckpt \
    resume=${CASE_init}/ckpts/initial_checkpoint.ckpt \
    system.exporter_part_type=mesh-exporter-init 
# 2.3 Texture initialization & segmentation
python train_init_texture.py \
    -s ${scene_path} \
    -m ${CASE_init}/save/init_mesh.npy \
    -o ${exp_root_dir} \
    --white_background True \
    --refinement_iterations 4000 \
    --seg_prompt ${seg_prompt} \
    --seg_mesh_path ${CASE_init}/save/init_mesh_coarse.ply \
    --gpu ${gpu_id}

# 3. Spatial editing
# 3.1 Geometry editing
exp_name="stage1-geometry-edit"
CASE_edit=${exp_root_dir}/${exp_name}
python train_spatial.py \
    --config configs/geometry-edit.yaml \
    --train \
    --gpu ${gpu_id} \
    tag=${tag} \
    name=${exp_name} \
    exp_root_dir=${exp_root_dir} \
    resume=${CASE_init}/ckpts/initial_checkpoint.ckpt \
    system.prev_checkpoint=${CASE_init}/ckpts/initial_checkpoint.ckpt \
    system.prompt_processor_local.prompt="${local_prompt}, black background, normal map" \
    system.prompt_processor_global.prompt="${global_prompt}, black background, normal map" \
    system.geometry.shape_init=${shape_init} \
    system.mask_npy_path=${exp_root_dir}/editing_region_info.npy \
    data.local_type=${sample_type}
# 3.2 Export Geometry
python train_spatial.py \
    --config ${CASE_edit}/configs/parsed.yaml \
    --export \
    --gpu ${gpu_id} \
    system.prev_checkpoint=${CASE_init}/ckpts/initial_checkpoint.ckpt \
    resume=${CASE_edit}/ckpts/last.ckpt \
    system.mask_npy_path=${exp_root_dir}/editing_region_info.npy \
    system.exporter_part_type=mesh-exporter-part

# 4. Texture editing
# [NOTE]: seed=-1 means random seed, try to change the seed for better results
# [NOTE]: upscale_to_2048=False means the refined output resolution is 1024x1024. Setting it to 2048x2048 results in more detailed textures but longer training time.
python train_edit_texture.py \
    -s ${scene_path} \
    -m ${CASE_edit}/save/edit_mesh.npy \
    -o ${exp_root_dir} \
    --white_background True \
    --refinement_iterations 2000 \
    --prompt "${global_prompt}" \
    --sample_type "${sample_type}" \
    --seed -1 \
    --upscale_to_2048 False \
    --gpu ${gpu_id}