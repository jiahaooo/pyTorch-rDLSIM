def define_Model(opt):
    if opt['model'] in ['reconstruction']:
        from rDL_model.model_plain import ModelPlain as M
    elif opt['model'] in ['rdl']:
        from rDL_model.model_rdl import ModelRDL as M
    else:
        raise NotImplementedError

    m = M(opt)

    return m
