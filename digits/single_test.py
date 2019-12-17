import torch
import tqdm

def single_test(val_loader, extractor, classifier, device):
    extractor.eval()
    classifier.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for idx, (t_img, t_label) in tqdm.tqdm(enumerate(val_loader)):
            t_img = t_img.to(device)
            t_label = t_label.to(device)
            t_feature = extractor(t_img)
            t_pred = classifier(t_feature)
            pred = t_pred.argmax(dim=1, keepdim=True)
            correct += pred.eq(t_label.view_as(pred)).sum().item()
            total += len(t_label)

        acc = correct * 1.0 / total

    print('Accuracy: %.4f%%' % (acc * 100.))

    return acc
