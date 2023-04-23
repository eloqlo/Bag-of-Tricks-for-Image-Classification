# trainer.py 실행
python trainer.py --model resnet50 --batch_size 64 --lr 0.0001 --epochs 100 --scheduler no --pretrained no 

# run.sh 실행
./run.sh

# tensorboard 실행
tensorboard --logdir=./runs