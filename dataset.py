from data import *

def get_train_Set(args):
    return Vimeo_train(args)

def get_validation_Set(args):
    return Vimeo_validation(args)

def get_test_Set(args):
    if args.Dataset == "vimeo":
        return Vimeo_test(args)
    elif args.Dataset =="ucf101":
        return UCF101_test(args)
    elif args.Dataset == 'SNU-FILM-all':
        return SNU_FILM_All_test(args)
    elif args.Dataset == 'Xiph_HD':
        return Xiph_HD_test(args)
    elif args.Dataset == 'X4K1000FPS':
        return X4K1000FPS_test(args)
    else:
        raise ValueError("[%s] is not available!"%(args.Dataset))        
