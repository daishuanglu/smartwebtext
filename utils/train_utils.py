import os
import glob
from pytorch_lightning import Trainer
import torch
import yaml
from GPUtil import showUtilization as gpu_usage
from pytorch_lightning.callbacks import ModelCheckpoint


MODEL_OBJ_PATH = '{logger_dir}/{model_name}.pt'
DEFAULT_LOGGER_DIR = 'default_lightning_log'
accelerator, device, num_devices = ("gpu", "cuda", torch.cuda.device_count()
                                    ) if torch.cuda.is_available() else ("cpu", "cpu", 1)
print("Use deep learning device: %s, %s, %d devices." % (accelerator,device, num_devices))

def latest_ckpt(logger_dir, model_name):
    list_of_files = glob.glob(os.path.join(logger_dir, '%s-epoch*.ckpt' % model_name))  # * means all if need specific format then *.csv
    if list_of_files:
        latest_file = max(list_of_files, key=os.path.getctime)
        print("found latest checkpoint %s" % latest_file)
    else:
        latest_file = None
        print("Not found latest checkpoint. ")
    return latest_file


def read_config(config_file):
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)
    return config


def load(model, state_dict_path):
    #model = torch.load(model_obj_path, pickle_module=dill, encoding='utf-8')
    state_dict = torch.load(state_dict_path, map_location=torch.device(device))
    if 'pytorch-lightning_version' in state_dict.keys():
        state_dict = state_dict['state_dict']
    model.load_state_dict(state_dict)
    return model


def save(model, PATH):
    torch.save(model.state_dict(), PATH)
    return


def save_best(cur, prev, config, model):
    def _monitor_criterion(cur, prev):
        return cur < prev
    #print("current:", cur, ", previous:", prev)
    if _monitor_criterion(cur, prev):
        print("%s is improved from %0.4f to %0.4f." % (
            config['monitor'], prev, cur))
        save(model, config['saved_model_path'] )
        print("Improved model %s Saved." % config['saved_model_path'])
        return True
    else:
        print("%s is not improved at %0.4f." % (config['monitor'], prev))
        return False


def training_pipeline(
        model,train_x, val_x, nepochs, model_name, resume_ckpt='',
        monitor="val_loss", logger_path=None):
    #torch.jit.script(model)
    ckpt_path = resume_ckpt if resume_ckpt else None
    print('torch jit script trainer')
    os.system("free -h")
    gpu_usage()
    if logger_path is None:
        os.makedirs(DEFAULT_LOGGER_DIR, exist_ok=True)
        logger_path = DEFAULT_LOGGER_DIR
    checkpoint_callback = ModelCheckpoint(
        monitor = monitor,
        dirpath = logger_path,
        filename = ('%s-epoch{epoch:02d}-%s{%s:.2f}' % (model_name, monitor, monitor)),
        auto_insert_metric_name = False,
        mode='min'
    )
    if ckpt_path:
        print('Resume training from check point: %s' % ckpt_path)
    trainer = Trainer(
        max_epochs=nepochs+1 if ckpt_path else nepochs,
        resume_from_checkpoint = ckpt_path,
        accelerator = accelerator,
        devices = num_devices,
        enable_checkpointing = True, # because we always have checkpoint callback
        callbacks=[checkpoint_callback])
    #print(model)
    #trainer.tune(model)
    #print("monitor metric before: ", trainer.callback_metrics[monitor].item())
    if train_x is None:
        trainer.validate(model, val_x)
        #print("monitor metric after: ", trainer.callback_metrics[monitor].item())
    else:
        trainer.fit(model, train_x, val_x)
    #torch.save(
    #    model,
    #    MODEL_OBJ_PATH.format(logger_dir=logger_path, model_name=model_name),
    #    pickle_module=dill)
    if val_x is None:
        return model, None
    elif train_x is None:
        return None, trainer.callback_metrics[monitor].item()
    else:
        return model, trainer.callback_metrics[monitor].item()