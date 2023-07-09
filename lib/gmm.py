import torch
import torch.nn.functional as F

from sklearn.mixture import GaussianMixture

from lib.models import build_model


class GMM:
    def __init__(self, net, loader, aug_fn, scale=1.):
        self.net = net
        self.net.eval()

        assert -1 <= scale <= 1
        if scale <= 0:
            scale_ori = -scale
            scale_aug = 1
        else:
            scale_ori = 1
            scale_aug = scale

        loss_list = []
        with torch.no_grad():
            for batch_idx, (data_ori, label) in enumerate(loader):
                data_ori, label = data_ori.cuda(), label.cuda()

                data_aug, _ = aug_fn.trainable_aug(data_ori, label)
                if aug_fn.base_aug is not None:
                    data_aug = torch.stack([aug_fn.base_aug(_x) for _x in data_aug])
                if aug_fn.normalizer is not None:
                    data_ori = aug_fn.normalizer(data_ori)

                preds = self.net(data_ori)
                loss = F.cross_entropy(preds, label, reduction='none')
                loss_list.append(loss[:int(data_ori.size(0) * scale_ori)])

                preds = self.net(data_aug)
                loss = F.cross_entropy(preds, label, reduction='none')
                loss_list.append(loss[:int(data_aug.size(0) * scale_aug)])

            loss_list = torch.log(torch.cat(loss_list, dim=0) + 1e-10)

        self.gmm = GaussianMixture(n_components=2, max_iter=10, tol=1e-2, reg_covar=5e-4)
        self.gmm.fit(loss_list.view(-1, 1).cpu().numpy())

    def predict(self, inputs, labels):
        with torch.no_grad():
            preds = self.net(inputs)
            input_loss = F.cross_entropy(preds, labels, reduction='none')
        prob = self.gmm.predict_proba(torch.log(input_loss + 1e-10).view(-1, 1).cpu().numpy())
        prob = prob[:, self.gmm.means_.argmin()]
        return prob


def get_gmm(model, n_classes, loader, path, device, aug_fn):
    print("GMM Initialization\nLoading Model:", path)
    net = build_model(model, n_classes)
    checkpoint = torch.load(path)
    net.load_state_dict(checkpoint['model'], strict=True)
    net.to(device)

    gmm = GMM(net, loader, aug_fn)
    print('Done')
    return gmm