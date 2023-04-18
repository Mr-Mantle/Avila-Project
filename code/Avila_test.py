from Py_torch.Avila_project.Avila_Project import *


if __name__ == '__main__':
    # np.random.seed(1)
    # t.manual_seed(1)

    dataset = AvilaDataset(split=True)
    dataset.generate(to_tensor=True)

    # model = AvilaAEDoubleLoss(dataset.n_input, h=9)

    # model = AvilaAESEDoubleLoss(dataset.n_input, h=9)
    # tr = TrainingProcess(model, dataset, double_loss=True, weights=(6, 1))

    model = AvilaModel(dataset.n_input)
    # tr = TrainingProcess(model, dataset)

    # model = AvilaSEModel(dataset.n_input)
    # tr = TrainingProcess(model, dataset)

    # tr.training(epochs=1000)
    # tr.evaluation()

    # tr.get_SE_weight()
    # tr.plot_dl_loss()



