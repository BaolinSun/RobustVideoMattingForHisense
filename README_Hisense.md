python inference.py --variant resnet50 --checkpoint checkpoints/rvm_resnet50.pth --device cuda --input-source data/val2022/Hisense/img --output-composition data/output --output-type png_sequence --num-workers 3

python inference.py --variant resnet50 --checkpoint checkpoints/rvm_resnet50.pth --device cuda --input-source data/val2022/Hisense/img --output-alpha data/output --output-type png_sequence --num-workers 3

python inference.py --eval --input-source data/val2022/Hisense/label --output-alpha data/output

python inference.py --variant resnet50 --checkpoint checkpoints/rvm_resnet50.pth --device cuda --input-source data/val2022/VideoMatte240K/img --output-composition data/output --output-type png_sequence --num-workers 3

python inference.py --variant resnet50 --checkpoint checkpoints/rvm_resnet50.pth --device cuda --input-source data/val2022/VideoMatte240K/img --output-alpha data/output --output-type png_sequence --num-workers 3

python inference.py --eval --input-source data/val2022/VideoMatte240K/label --output-alpha data/output