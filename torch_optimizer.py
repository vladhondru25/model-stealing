import random
import numpy as np
import torch
import torchvision
# import setup


def loss(softmax, image, label):
    softmax = np.exp(softmax) / np.sum(np.exp(softmax))
    return np.sum(np.power(
        softmax -
        np.eye(10)[label],
        2
    ))

def optimize_to_grayscale(classifier, generator, batch_size = 64, encoding_size = 128):
    batch = []

    n_iter = batch_size
    for i in range(n_iter):
        c = 0.
        x = 0
        specimens = np.random.uniform(-3.3, 3.3, size = (30, encoding_size))
        label = np.random.randint(10, size = (1, 1))
        while c < .9 and x < 10:
            x += 1
            encodings = specimens
            with torch.no_grad():
                images = generator(torch.tensor(
                    specimens
                ).float().cuda())
                multipliers = [.2126, .7152, .0722]
                multipliers = np.expand_dims(multipliers, 0)
                multipliers = np.expand_dims(multipliers, -1)
                multipliers = np.expand_dims(multipliers, -1)
                multipliers = np.tile(multipliers, [1, 1, 32, 32])
                multipliers = torch.Tensor(multipliers).to('cuda')
                images = images * multipliers
                images = images.sum((1,), keepdim = True)

                softmaxes = classifier(images).detach().cpu()
            losses = [loss(np.array(s), i, label) for s, i in zip(softmaxes, images)]
            indexes = np.argsort(losses)
            image = images[indexes[0]: indexes[0] + 1]
            specimens = specimens[indexes[:10]]
            specimens = np.concatenate([
                specimens,
                specimens + np.random.normal(scale = .5, size = (10, encoding_size)),
                specimens + np.random.normal(scale = .5, size = (10, encoding_size))
            ])
            c = softmaxes[indexes[0]][label]
        batch.append(image)
    return torch.cat(batch)#, axis = 0)


def optimize_rescale(classifier, generator, batch_size = 10):
    batch = []

    n_iter = batch_size
    for i in range(n_iter):
        encoding_size = 512        
        encoding_size = 128
        c = 0.
        x = 0
        specimens = np.random.uniform(-3.3, 3.3, size = (2, encoding_size))
        label = np.random.randint(101, size = (1, 1))
        # while c < .90 and x < 3:
        for _ in range(2):
            x += 1
            encodings = specimens
            with torch.no_grad():
                images = generator(torch.tensor(
                    specimens
                ).to(device='cuda').float())
                images = torch.nn.functional.interpolate(images, size = 224)
                # print(images.shape)
                # images = torchvision.transforms.Resize(images, size = 224) # food101
                # multipliers = [.2126, .7152, .0722]
                # multipliers = np.expand_dims(multipliers, 0)
                # multipliers = np.expand_dims(multipliers, -1)
                # multipliers = np.expand_dims(multipliers, -1)
                # multipliers = np.tile(multipliers, [1, 1, 32, 32])
                # multipliers = torch.Tensor(multipliers).to(device)
                # images = images * multipliers
                # images = images.sum(axis = 1, keepdims = True)

                softmaxes = classifier(images).detach().cpu()
            losses = [loss(np.array(s), i, label) for s, i in zip(softmaxes, images)]
            indexes = np.argsort(losses)
            image = images[indexes[0]: indexes[0] + 1]
            specimens = specimens[indexes[:1]]
            specimens = np.concatenate([
                specimens,
                specimens + np.random.normal(scale = .5, size = (1, encoding_size)),
            ])
            c = softmaxes[indexes[0]][label]
        batch.append(image)
    return torch.cat(batch)#, axis = 0)

# def optimize(classifier, generator, batch_size = 20):
def optimize(classifier, generator, batch_size = 10):
    batch = []

    n_iter = batch_size
    encoding_size = 256
    if 'sngan' in str(type(generator)):
        encoding_size = 128
    encoding_size = 128
    for i in range(n_iter // 2):
        c = 0.
        x = 0
        # specimens = np.random.uniform(-3.3, 3.3, size = (2, encoding_size))
        rand_classes = [random.randint(0, 999), random.randint(0, 999)]
        while rand_classes[0] == rand_classes[1]:
            rand_classes[1] = random.randint(0, 999)

        class_vector = np.zeros(shape=(2,1000), dtype=np.float32)
        class_vector[0][rand_classes[0]] = 1.0
        class_vector[1][rand_classes[1]] = 1.0
        specimens = truncated_noise_sample(truncation=0.4, batch_size=2)
        # print(specimens.shape)

        class_vector = torch.from_numpy(class_vector).to('cuda')

        label = np.random.randint(101, size = (2, 1))

        with torch.no_grad():
            image = generator(torch.from_numpy(specimens).to('cuda'), class_vector, 0.4)

        # while c < .9 and x < 300:# and classifier.calls_made < 20:
        for _ in range(2):
            x += 1
            encodings = specimens
            with torch.no_grad():
                images = generator(
                    torch.from_numpy(specimens).to('cuda'), class_vector, 0.4
                )
                images = torchvision.transforms.Resize(
                    224,
                    interpolation=torchvision.transforms.InterpolationMode.BILINEAR
                )(images) # food101
                # multipliers = [.2126, .7152, .0722]
                # multipliers = np.expand_dims(multipliers, 0)
                # multipliers = np.expand_dims(multipliers, -1)
                # multipliers = np.expand_dims(multipliers, -1)
                # multipliers = np.tile(multipliers, [1, 1, 32, 32])
                # multipliers = torch.Tensor(multipliers).to(device)
                # images = images * multipliers
                # images = images.sum(axis = 1, keepdims = True)

                # print(f'Cate imagini sunt: {len(images)}')
                softmaxes = classifier(images).detach().cpu()
            losses = [loss(np.array(s), i, label) for s, i in zip(softmaxes, images)]
            indexes = np.argsort(losses)
            image = images[indexes[0]: indexes[0] + 5]
            specimens = specimens[indexes[:1]]
            specimens = np.concatenate([
                specimens,
                specimens + np.random.normal(scale = .5, size = (1, encoding_size)).astype(specimens.dtype)
            ])
            c = softmaxes[indexes[0]].numpy()
            c = c[label]
            # break
            batch.append(image)

        """
        batch.append(image)
        """
    
    # print(classifier.calls_made)
    # print(f'len batch: {len(batch)}')
    # print(f'len batch: {len(batch[0])}')
    print(f'shape: {torch.cat(batch).shape}')
    return torch.cat(batch)#, axis = 0)


def discrepancy_loss(teacher_predictions, student_predictions):
    teacher_softmax = np.exp(teacher_predictions) / np.sum(
        np.exp(teacher_predictions)
    )
    student_softmax = np.exp(student_predictions) / np.sum(
        np.exp(student_predictions)
    )
    return -np.sum(np.square(teacher_softmax - student_softmax))


def optimize_discrepancies(teacher, student, generator, batch_size = 16):
    batch = []

    n_iter = batch_size
    for i in range(n_iter):
        encoding_size = 128
        specimens = np.random.uniform(-3.3, 3.3, size = (30, encoding_size))
        x = 0
        while x < 10:
            x += 1
            encodings = specimens
            with torch.no_grad():
                images = generator(torch.tensor(
                    specimens
                ).float().to('cuda'))
                multipliers = [.2126, .7152, .0722]
                multipliers = np.expand_dims(multipliers, 0)
                multipliers = np.expand_dims(multipliers, -1)
                multipliers = np.expand_dims(multipliers, -1)
                multipliers = np.tile(multipliers, [1, 1, 32, 32])
                multipliers = torch.Tensor(multipliers).to('cuda')
                images = images * multipliers
                images = images.sum(axis = 1, keepdims = True)

                teacher_predictions = teacher(images).detach().cpu()
                student_predictions = student(images).detach().cpu()
            losses = [
                discrepancy_loss(np.array(s), np.array(i))
                for s, i in zip(
                    teacher_predictions, student_predictions
                )
            ]
            indexes = np.argsort(losses)
            image = images[indexes[0]: indexes[0] + 1]
            specimens = specimens[indexes[:10]]
            specimens = np.concatenate([
                specimens,
                specimens + np.random.normal(scale = .5, size = (10, encoding_size)),
                specimens + np.random.normal(scale = .5, size = (10, encoding_size))
            ])
        batch.append(image)
    return torch.cat(batch, axis = 0)

def discrepancy_loss_kl(teacher_predictions, student_predictions):
    teacher_softmax = np.exp(teacher_predictions) / np.sum(
        np.exp(teacher_predictions)
    )
    student_softmax = np.exp(student_predictions) / np.sum(
        np.exp(student_predictions)
    )
    return -np.sum(teacher_softmax * np.log(teacher_softmax) / np.log(student_softmax))

def optimize_discrepancies_kl(teacher, student, generator, batch_size = 64):
    batch = []

    n_iter = batch_size
    for i in range(n_iter):
        encoding_size = 128
        specimens = np.random.uniform(-3.3, 3.3, size = (30, encoding_size))
        x = 0
        while x < 50:
            x += 1
            encodings = specimens
            with torch.no_grad():
                images = generator(torch.tensor(
                    specimens
                ).float().to('cuda'))
                multipliers = [.2126, .7152, .0722]
                multipliers = np.expand_dims(multipliers, 0)
                multipliers = np.expand_dims(multipliers, -1)
                multipliers = np.expand_dims(multipliers, -1)
                multipliers = np.tile(multipliers, [1, 1, 32, 32])
                multipliers = torch.Tensor(multipliers).to('cuda')
                images = images * multipliers
                images = images.sum(axis = 1, keepdims = True)

                teacher_predictions = teacher(images).detach().cpu()
                student_predictions = student(images).detach().cpu()
            losses = [
                discrepancy_loss_kl(np.array(s), np.array(i))
                for s, i in zip(
                    teacher_predictions, student_predictions
                )
            ]
            indexes = np.argsort(losses)
            image = images[indexes[0]: indexes[0] + 1]
            specimens = specimens[indexes[:10]]
            specimens = np.concatenate([
                specimens,
                specimens + np.random.normal(scale = .5, size = (10, encoding_size)),
                specimens + np.random.normal(scale = .5, size = (10, encoding_size))
            ])
        batch.append(image)
    return torch.cat(batch, axis = 0)


def loss(softmax, image, label):
    softmax = np.exp(softmax) / np.sum(np.exp(softmax))
    return np.sum(np.power(
        softmax -
        np.eye(101)[label],
        2
    ))

def optimize_discrepancies_(teacher, student, generator, batch_size = 64):
    encoding_size = 128
    batch_size = 16

    with torch.no_grad():
        specimens = torch.tensor(
            np.random.uniform(-3.3, 3.3, size = (batch_size, 30, encoding_size))
        ).float().to('cuda')

        for _ in range(10):
            images = generator(specimens.view(-1, encoding_size))
            multipliers = [.2126, .7152, .0722]
            multipliers = np.expand_dims(multipliers, 0)
            multipliers = np.expand_dims(multipliers, -1)
            multipliers = np.expand_dims(multipliers, -1)
            multipliers = np.tile(multipliers, [1, 1, 32, 32])
            multipliers = torch.Tensor(multipliers).to('cuda')
            images = images * multipliers
            images = images.sum(axis = 1, keepdims = True)
            teacher_predictions = torch.softmax(
                teacher(images), axis = -1
            ).detach().cpu()
            student_predictions = torch.softmax(
                student(images), axis = -1
            ).detach().cpu()

            losses = -1. * torch.pow(
                teacher_predictions - student_predictions, 2
            ).sum(-1).view(batch_size, 30)
            indexes = torch.argsort(losses) < 10
            specimens = specimens[indexes].view(batch_size, 10, encoding_size)
            specimens = torch.cat((
                specimens,
                specimens + torch.randn(batch_size, 10, encoding_size).to('cuda'),
                specimens + torch.randn(batch_size, 10, encoding_size).to('cuda'),
            ), axis = 1)

        images = generator(specimens.view(-1, encoding_size))
        images = generator(specimens.view(-1, encoding_size))
        multipliers = [.2126, .7152, .0722]
        multipliers = np.expand_dims(multipliers, 0)
        multipliers = np.expand_dims(multipliers, -1)
        multipliers = np.expand_dims(multipliers, -1)
        multipliers = np.tile(multipliers, [1, 1, 32, 32])
        multipliers = torch.Tensor(multipliers).to('cuda')
        images = images * multipliers
        images = images.sum(axis = 1, keepdims = True)

        teacher_predictions = torch.softmax(
            teacher(images), axis = -1
        ).detach().cpu()
        student_predictions = torch.softmax(
            student(images), axis = -1
        ).detach().cpu()

        losses = -1. * torch.pow(
            teacher_predictions - student_predictions, 2
        ).sum(-1).view(batch_size, 30)

        indexes = torch.argsort(losses) < 1
        images = images.view(batch_size, 30, 1, 32, 32)[indexes]
    return images




def curriculum_loss(teacher_predictions, student_predictions, label, weight):
    teacher_softmax = np.exp(teacher_predictions) / np.sum(
        np.exp(teacher_predictions)
    )
    student_softmax = np.exp(student_predictions) / np.sum(
        np.exp(student_predictions)
    )
    return (
        np.sum(np.square(teacher_softmax - label)) -
        weight * np.sum(np.square(teacher_softmax - student_softmax))
    )


def optimize_curriculum(teacher, student, generator, epoch, batch_size = 16):
    batch = []
    weights = [0.] * 4 + list(np.linspace(0, 1., 46)) + [1.] * 200

    n_iter = batch_size
    for i in range(n_iter):
        encoding_size = 128
        specimens = np.random.uniform(-3.3, 3.3, size = (30, encoding_size))
        x = 0
        label = np.eye(10)[np.random.randint(10)]
        while x < 10:
            x += 1
            encodings = specimens
            with torch.no_grad():
                images = generator(torch.tensor(
                    specimens
                ).float().cuda())
                multipliers = [.2126, .7152, .0722]
                multipliers = np.expand_dims(multipliers, 0)
                multipliers = np.expand_dims(multipliers, -1)
                multipliers = np.expand_dims(multipliers, -1)
                multipliers = np.tile(multipliers, [1, 1, 32, 32])
                multipliers = torch.Tensor(multipliers).to('cuda')
                images = images * multipliers
                images = images.sum(axis = 1, keepdims = True)

                teacher_predictions = teacher(images).detach().cpu()
                student_predictions = student(images).detach().cpu()
            losses = [
                curriculum_loss(np.array(s), np.array(i), label, weights[epoch])
                for s, i in zip(
                    teacher_predictions, student_predictions
                )
            ]
            indexes = np.argsort(losses)
            image = images[indexes[0]: indexes[0] + 1]
            specimens = specimens[indexes[:10]]
            specimens = np.concatenate([
                specimens,
                specimens + np.random.normal(scale = .5, size = (10, encoding_size)),
                specimens + np.random.normal(scale = .5, size = (10, encoding_size))
            ])
        batch.append(image)
    return torch.cat(batch, axis = 0)
