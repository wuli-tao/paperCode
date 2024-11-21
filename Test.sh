Log_Name='test'
Resume_Model='/home/zhongtao/code/CrossDomainFER/my_method/checkpoints/RAFDB_JAFFE_Baseline.pkl'
OutputPath='/home/zhongtao/code/CrossDomainFER/my_method/checkpoints'
GPU_ID=1
Backbone='ResNet18'
Network='Baseline'
sourceDataset='SFEW'
targetDataset='JAFFE'
batch_size=32
useMultiDatasets='False'
epochs=50
lr=0.0002
showFeature='False'
workers=4
seed=2022

CUDA_VISIBLE_DEVICES=${GPU_ID} python3 Test.py \
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
    --showFeature ${showFeature} \
    --workers ${workers} \
    --seed ${seed}


Resume_Model='/home/zhongtao/code/CrossDomainFER/my_method/checkpoints/RAFDB_JAFFE_AConv+DHSA.pkl'
Network='AConv+DHSA'
#sourceDataset='SFEW'
#targetDataset='JAFFE'

CUDA_VISIBLE_DEVICES=${GPU_ID} python3 Test.py \
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
    --showFeature ${showFeature} \
    --workers ${workers} \
    --seed ${seed}

Resume_Model='/home/zhongtao/code/CrossDomainFER/my_method/checkpoints/RAFDB_JAFFE_AConv+DHSA+global.pkl'
Network='AConv+DHSA+global'
#sourceDataset='SFEW'
#targetDataset='SFEW'

CUDA_VISIBLE_DEVICES=${GPU_ID} python3 Test.py \
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
    --showFeature ${showFeature} \
    --workers ${workers} \
    --seed ${seed}

Resume_Model='/home/zhongtao/code/CrossDomainFER/my_method/checkpoints/RAFDB_JAFFE_AConv+DHSA+global(x).pkl'
Network='AConv+DHSA+global(x)'
#sourceDataset='SFEW'
#targetDataset='ExpW'

CUDA_VISIBLE_DEVICES=${GPU_ID} python3 Test.py \
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
    --showFeature ${showFeature} \
    --workers ${workers} \
    --seed ${seed}
