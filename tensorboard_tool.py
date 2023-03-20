import pandas as pd
from torch.utils.tensorboard import SummaryWriter


features = {'epoch': [], 'train_loss': [], 'val_loss_epoch': [], 'val_loss_step': []}
version = 1

tsbd_event_csv = "lightning_logs_final/version_{v}/version_{v}-{feature}.csv"
output_tsbd_event_dir = "lightning_logs_final/version_{v}/epoch".format(v=str(version))
nrows = 0
for k in features.keys():
    fpath = tsbd_event_csv.format(feature=k, v=str(version))
    df = pd.read_csv(fpath)
    rename_dict = {c: '_'.join(c.lower().split()) for c in df.columns}
    df = df.rename(rename_dict, axis=1)
    features[k] = df
    nrows = len(features[k])

step_epoch = features['epoch'][['step', 'value']]
step_epoch['step'] = step_epoch['step'].astype(int)
step_epoch = step_epoch.drop_duplicates('value', keep='last')
index = []
value = []
i = 0
for _, row in step_epoch.iterrows():
    step = int(row['step']) + 1
    index +=  list(range(i, step))
    value += [row['value'] for _ in range(step-i)]
    i = step
step_epoch = pd.DataFrame(value, index=index, columns=['value'])

writer = SummaryWriter(output_tsbd_event_dir)
for i, row in features['val_loss_epoch'].iterrows():
    print(i)
    writer.add_scalar('val_loss_epoch', row['value'], i)

train_loss_epoch = features['train_loss'].copy()
train_loss_epoch['step'] = train_loss_epoch['step'].apply(lambda x: step_epoch.loc[x]['value'])
train_loss_epoch = train_loss_epoch.drop_duplicates('step', keep='last') # keep loss for the latest epoch step
for _, row in train_loss_epoch.iterrows():
    print(int(row['step']))
    writer.add_scalar('train_loss_epoch', row['value'], int(row['step']))
print('done.')
