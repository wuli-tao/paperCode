Log_Name='Train_Adaption'
Resume_Model='None'
OutputPath='/home/zhongtao/code/CrossDomainFER/my_method/checkpoints'
GPU_ID=2
Backbone='ResNet18'
Network='AConv+DHSA+global(x)'
sourceDataset='SFEW'
targetDataset='JAFFE'
batch_size=32
useMultiDatasets='False'
epochs=50
lr=0.0002
momentum=0.9
weight_decay=0.01
isTest='False'
isSave='True'
showFeature='False'
workers=4
seed=2022

CUDA_VISIBLE_DEVICES=${GPU_ID} python3 Train_Adaption.py \
    --Log_Name ${Log_Name} \
    --OutputPath ${OutputPath} \
    --Resume_Model ${Resume_Model} \
    --GPU_ID ${GPU_ID} \
    --Backbone ${Backbone} \
    --Network ${Network} \
    --sourceDataset ${sourceDataset} \
    --targetDataset ${targetDataset} \
    --batch_size ${batch_size}\
    --useMultiDatasets ${useMultiDatasets} \
    --epochs ${epochs} \
    --lr ${lr} \
    --momentum ${momentum} \
    --weight_decay ${weight_decay} \
    --isTest ${isTest} \
    --isSave ${isSave} \
    --showFeature ${showFeature} \
    --workers ${workers} \
    --seed ${seed}


#Backbone='ResNet18'
#Network='AConv+DHSA'
sourceDataset='SFEW'
targetDataset='FER2013Plus'

CUDA_VISIBLE_DEVICES=${GPU_ID} python3 Train_Adaption.py \
    --Log_Name ${Log_Name} \
    --OutputPath ${OutputPath} \
    --Resume_Model ${Resume_Model} \
    --GPU_ID ${GPU_ID} \
    --Backbone ${Backbone} \
    --Network ${Network} \
    --sourceDataset ${sourceDataset} \
    --targetDataset ${targetDataset} \
    --batch_size ${batch_size}\
    --useMultiDatasets ${useMultiDatasets} \
    --epochs ${epochs} \
    --lr ${lr} \
    --momentum ${momentum} \
    --weight_decay ${weight_decay} \
    --isTest ${isTest} \
    --isSave ${isSave} \
    --showFeature ${showFeature} \
    --workers ${workers} \
    --seed ${seed}

#Backbone='unified'
#Network='AConv+DHSA+global'
sourceDataset='SFEW'
targetDataset='RAFDB'

CUDA_VISIBLE_DEVICES=${GPU_ID} python3 Train_Adaption.py \
    --Log_Name ${Log_Name} \
    --OutputPath ${OutputPath} \
    --Resume_Model ${Resume_Model} \
    --GPU_ID ${GPU_ID} \
    --Backbone ${Backbone} \
    --Network ${Network} \
    --sourceDataset ${sourceDataset} \
    --targetDataset ${targetDataset} \
    --batch_size ${batch_size}\
    --useMultiDatasets ${useMultiDatasets} \
    --epochs ${epochs} \
    --lr ${lr} \
    --momentum ${momentum} \
    --weight_decay ${weight_decay} \
    --isTest ${isTest} \
    --isSave ${isSave} \
    --showFeature ${showFeature} \
    --workers ${workers} \
    --seed ${seed}

#Backbone='ResNet18'
#Network='AConv+DHSA+global(x)'
sourceDataset='SFEW'
targetDataset='ExpW'

CUDA_VISIBLE_DEVICES=${GPU_ID} python3 Train_Adaption.py \
    --Log_Name ${Log_Name} \
    --OutputPath ${OutputPath} \
    --Resume_Model ${Resume_Model} \
    --GPU_ID ${GPU_ID} \
    --Backbone ${Backbone} \
    --Network ${Network} \
    --sourceDataset ${sourceDataset} \
    --targetDataset ${targetDataset} \
    --batch_size ${batch_size}\
    --useMultiDatasets ${useMultiDatasets} \
    --epochs ${epochs} \
    --lr ${lr} \
    --momentum ${momentum} \
    --weight_decay ${weight_decay} \
    --isTest ${isTest} \
    --isSave ${isSave} \
    --showFeature ${showFeature} \
    --workers ${workers} \
    --seed ${seed}

#Backbone='ResNet18'
#Network='Baseline'
sourceDataset='SFEW'
targetDataset='AffectNet'

CUDA_VISIBLE_DEVICES=${GPU_ID} python3 Train_Adaption.py \
    --Log_Name ${Log_Name} \
    --OutputPath ${OutputPath} \
    --Resume_Model ${Resume_Model} \
    --GPU_ID ${GPU_ID} \
    --Backbone ${Backbone} \
    --Network ${Network} \
    --sourceDataset ${sourceDataset} \
    --targetDataset ${targetDataset} \
    --batch_size ${batch_size}\
    --useMultiDatasets ${useMultiDatasets} \
    --epochs ${epochs} \
    --lr ${lr} \
    --momentum ${momentum} \
    --weight_decay ${weight_decay} \
    --isTest ${isTest} \
    --isSave ${isSave} \
    --showFeature ${showFeature} \
    --workers ${workers} \
    --seed ${seed}