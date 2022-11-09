python3 base_experiment.py --generator cifar_100_90_classes_gan \
                           --optim adam \
                           --proxy_dataset cifar10 \
                           --sample_optimization class \
                           --samples optimized \
                           --size 32 \
                           --student half_alexnet \
                           --teacher alexnet \
                           --true_dataset cifar10 \
                           --calls_limit 51200