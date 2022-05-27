python inference.py --variant resnet50 --checkpoint checkpoints/rvm_resnet50.pth --device cuda --input-source data/val2022/Hisense/img --output-composition data/output --output-type png_sequence --num-workers 3

python inference.py --variant resnet50 --checkpoint checkpoints/rvm_resnet50.pth --device cuda --input-source data/val2022/Hisense/img --output-alpha data/output --output-type png_sequence --num-workers 3

python inference.py --eval --input-source data/val2022/Hisense/label --output-alpha data/output

python inference.py --variant resnet50 --checkpoint checkpoints/rvm_resnet50.pth --device cuda --input-source data/val2022/VideoMatte240K/img --output-composition data/output --output-type png_sequence --num-workers 3

python inference.py --variant resnet50 --checkpoint checkpoints/rvm_resnet50.pth --device cuda --input-source data/val2022/VideoMatte240K/img --output-alpha data/output --output-type png_sequence --num-workers 3

python inference.py --eval --input-source data/val2022/VideoMatte240K/label --output-alpha data/output

python inference_speed_test.py --model-variant mobilenetv3 --resolution 3840 2160 --downsample-ratio 0.125
python inference_speed_test.py --model-variant mobilenetv3 --resolution 1920 1080 --downsample-ratio 0.25

python inference.py --variant resnet50 --checkpoint checkpoints/rvm_resnet50.pth --device cuda --input-source data/inter_data --interact-segm


python inference.py --variant resnet50 --checkpoint checkpoints/rvm_resnet50.pth --device cuda --input-source data/spiderman.mp4 --output-composition data/output.mp4 --output-type video --num-workers 3