from src_files.data.get_coco import get_coco
from src_files.data.get_voc import voc
from src_files.data.uniformSample import get_uniform_voc
from src_files.data.vg import get_VG


def get_dataset(args):
    if args.dataname == 'coco' or args.dataname == 'coco17':
        train_dataset, val_dataset = get_coco(args)

    elif args.dataname == 'voc2007':
        train_dataset, val_dataset = voc(args)
        #train_dataset, val_dataset = get_uniform_voc(args)
    elif args.dataname == 'vg':
        train_dataset, val_dataset = get_VG(args)
    else:
        raise NotImplementedError("Unknown dataname %s" % args.dataname)

    print("len(train_dataset):", len(train_dataset))
    print("len(val_dataset):", len(val_dataset))
    return train_dataset, val_dataset