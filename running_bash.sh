# 首先通过main训练
model_name=MAnet
encoder_name=resnet34
epoch=6
load_model=/home/rton/pyproj/ieee_building_extract/new_test/save_model/old/MAnet_resnet34_8_pretrain.pkl
pre_trained_save_path=/home/rton/pyproj/ieee_building_extract/new_test/save_model/${model_name}_${encoder_name}_${epoch}_pretrain.pkl
finetune_savepath=/home/rton/pyproj/ieee_building_extract/new_test/save_model/${model_name}_${encoder_name}_finetune.pkl

# 预训练
python -u main.py \
    --model_name $model_name \
    --encoder_name $encoder_name \
    --load_model $load_model \
    --epoch $epoch \
    --model_savepath $pre_trained_save_path

# 预训练结果测试
python -u rs_big_pred.py \
    --model_name $model_name \
    --encoder_name $encoder_name \
    --model_path $pre_trained_save_path \
    --flag pretrain

# 微调
python -u finetune_model.py \
    --model_name $model_name \
    --encoder_name $encoder_name \
    --finetune_epoch 5 \
    --pre_trained_modelpath /home/rton/pyproj/ieee_building_extract/new_test/save_model/MAnet_resnet34_8_pretrain.pkl \
    --model_savepath $finetune_savepath

# 预训练结果测试
python -u rs_big_pred.py \
    --model_name $model_name \
    --encoder_name $encoder_name \
    --model_path $finetune_savepath \
    --flag finetune
