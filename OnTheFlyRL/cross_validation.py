from model_env_train import *
from sklearn.model_selection import KFold
from torch.utils.data import Subset

def split(data, train_index, test_index):
    comp_data, proc_data, prop_data, elem_feature = data
    return (
        (comp_data[train_index], proc_data[train_index], prop_data[train_index], elem_feature),
        (comp_data[test_index], proc_data[test_index], prop_data[test_index], elem_feature),
    )

def cross_validation(num_training_epochs = 500,
          batch_size = 16,
          k_fold = 10,
          save_path = 'cross_validation.txt'):
    ''' util func for training_epoch_num validation '''
    d = load_data()
    d, scalers = fit_transform(d)

    norm_exp, norm_pred = [], []

    kf = KFold(n_splits = k_fold, shuffle=True, random_state=42)
    for train_index, test_index in kf.split(range(len(d[0])), range(len(d[0]))):
        train_d, val_d = split(d, train_index, test_index)

        loss_fn = torch.nn.MSELoss()
        dl = get_dataloader(train_d, batch_size)
        model = CnnDnnModel()
        # train one epoch
        for epoch in range(num_training_epochs):
            model.train()
            _batch_loss_buffer = []
            for comp, proc, prop, elem_t in dl:
                # forward pass
                out = model(comp, elem_t, proc)
                prop = prop.reshape(*(out.shape))
                l = loss_fn(out, prop)

                # backward pass
                model.optimizer.zero_grad()
                l.backward()
                model.optimizer.step()
                
                _batch_loss_buffer.append(l.item())
            
            # model.eval()
            # _batch_mean_loss = np.mean(_batch_loss_buffer)
        
        model.eval()
        dl = get_dataloader(val_d, len(val_d[0]))
        comp, proc, prop, elem_t = next(iter(dl))
        out = model(comp, elem_t, proc).detach().numpy()
        prop = prop.reshape(*(out.shape)).detach().numpy()

        norm_exp.extend(prop.flatten())
        norm_pred.extend(out.flatten())
    
    np.savetxt(
        save_path,
        np.vstack((norm_exp, norm_pred)).T,
        fmt = '%.6f',
        delimiter = '\t',
    )
    
    return model, d, scalers

cross_validation()