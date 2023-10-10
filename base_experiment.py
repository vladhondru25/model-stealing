import argparse
import setup
import trainer

if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(description = 'Experiment setup')

    arg_parser.add_argument('--epochs', type = str, default = '200')
    arg_parser.add_argument('--generator', type = str, default = 'cifar_100_90_classes_gan')
    arg_parser.add_argument('--optim', type = str, default = 'adam')
    arg_parser.add_argument('--proxy_dataset', type = str, default = 'cifar10')
    arg_parser.add_argument('--sample_optimization', type = str, default = 'class')
    arg_parser.add_argument('--samples', type = str, default = 'optimized')
    arg_parser.add_argument('--size', type = int, default = 32)
    arg_parser.add_argument('--student', type = str, default = 'half_lenet')
    arg_parser.add_argument('--teacher', type = str, default = 'lenet')
    arg_parser.add_argument('--true_dataset', type = str, default = 'split_fmnist')
    arg_parser.add_argument('--calls_limit', type = int, default = 51200)

    env = arg_parser.parse_args()

    for _ in range(5):
        # for power in range(13):
        for power in range(0,11):
            calls_limit = 101 * (2**power)

            teacher, teacher_dataset, student = setup.prepare_teacher_student(env)

            # trainer.evaluate(teacher, teacher_dataset)
            generator = setup.prepare_generator(env)

            student_dataset = setup.prepare_student_dataset(
                env, teacher, teacher_dataset, student, generator
            )

            if env.optim == 'sgd':
                trainer.train_or_restore_predictor(
                    student, student_dataset, loss_type = 'binary',
                    n_epochs = int(env.epochs)
                )
            else:
                trainer.train_or_restore_predictor_adam(
                    student, student_dataset, loss_type = 'binary',
                    n_epochs = int(env.epochs),
                    calls_limit=calls_limit
                )
            trainer.evaluate(student, teacher_dataset, calls_limit)
            print()
        print()

    """ Original code
    teacher, teacher_dataset, student = setup.prepare_teacher_student(env)
    trainer.evaluate(teacher, teacher_dataset)
    generator = setup.prepare_generator(env)

    student_dataset = setup.prepare_student_dataset(
        env, teacher, teacher_dataset, student, generator
    )

    if env.optim == 'sgd':
        trainer.train_or_restore_predictor(
            student, student_dataset, loss_type = 'binary',
            n_epochs = int(env.epochs)
        )
    else:
        trainer.train_or_restore_predictor_adam(
            student, student_dataset, loss_type = 'binary',
            n_epochs = int(env.epochs), calls_limit = env.calls_limit
        )
        
    proxy_dataset_train, proxy_dataset_valid = setup.prepare_proxy_dataset()
    trainer.train_proxy_dataset(model=student, dataset=(proxy_dataset_train, proxy_dataset_valid), n_epochs=int(env.epochs))
        
    trainer.evaluate(student, teacher_dataset)
    """


