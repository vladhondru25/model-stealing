# python3 base_experiment.py --generator cifar_100_90_classes_gan \
#                            --optim adam \
#                            --proxy_dataset cifar10 \
#                            --sample_optimization class \
#                            --samples optimized \
#                            --size 32 \
#                            --student half_alexnet \
#                            --teacher alexnet \
#                            --true_dataset cifar10 \
#                            --calls_limit 51200
# python3 base_experiment.py --generator cifar_100_90_classes_gan \
#                            --optim adam \
#                            --proxy_dataset food101 \
#                            --sample_optimization class \
#                            --samples optimized \
#                            --size 224 \
#                            --student half_alexnet_food \
#                            --teacher alexnet_food \
#                            --true_dataset food101 \
#                            --calls_limit 51200
python3 base_experiment.py --generator cifar_100_90_classes_gan \
                           --optim adam \
                           --proxy_dataset food101 \
                           --sample_optimization class \
                           --samples optimized \
                           --size 224 \
                           --student resnet_food \
                           --teacher resnet50 \
                           --true_dataset food101 \
                           --calls_limit 51200